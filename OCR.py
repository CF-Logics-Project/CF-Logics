from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.layout_model_specs import (
    DOCLING_LAYOUT_EGRET_LARGE,
    DOCLING_LAYOUT_EGRET_XLARGE,
    DOCLING_LAYOUT_HERON,
    DOCLING_LAYOUT_V2,
)
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionVlmOptions,
    RapidOcrOptions,
    granite_picture_description,
    smolvlm_picture_description,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from huggingface_hub import snapshot_download
from langdetect import DetectorFactory, detect
from pypdf import PdfReader, PdfWriter

DEFAULT_OCR_REPO_ID = "SWHL/RapidOCR"
DEFAULT_PICTURE_DESCRIPTION_PROMPT = (
    "Extract all text, axis labels, legend items, and coordinates from this chart. "
    "Recreate the structured data in a brief text summary or key-value list."
)
DOCLING_PROGRESS_LOGGERS = (
    "docling",
    "docling.pipeline",
    "docling.document_converter",
)
QUIET_PROGRESS_LOGGERS = (
    "huggingface_hub",
    "onnxruntime",
)

# Force consistent, reproducible results from the language detector.
DetectorFactory.seed = 0


class _DoclingProgressFormatter(logging.Formatter):
    """Format Docling logs into compact progress-oriented console lines."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        message = re.sub(r"\s+", " ", record.getMessage()).strip()
        prefix = self._build_prefix(record.name, message)
        return f"{timestamp} {prefix} {message}"

    @staticmethod
    def _build_prefix(logger_name: str, message: str) -> str:
        lowercase_message = message.lower()
        progress_match = re.search(r"\b(\d+)\s*/\s*(\d+)\b", message)
        page_match = re.search(r"\bpage\s+(\d+)\b", lowercase_message)

        if page_match:
            return f"[DOCLING PAGE {page_match.group(1)}]"
        if "page" in lowercase_message and progress_match:
            return f"[DOCLING PAGE {progress_match.group(1)}/{progress_match.group(2)}]"
        if "page" in lowercase_message:
            return "[DOCLING PAGE]"
        if progress_match:
            return f"[DOCLING {progress_match.group(1)}/{progress_match.group(2)}]"

        component = logger_name.removeprefix("docling.").replace(".", "/").upper()
        if component == "DOCLING":
            return "[DOCLING]"
        return f"[DOCLING {component}]"


def configure_docling_progress_logging(
    enabled: bool = True,
    level: int = logging.INFO,
) -> None:
    """Configure opt-in console logging for Docling progress messages.

    This does not guarantee true page counters; it surfaces Docling's internal
    progress messages when the library emits them.
    """
    docling_logger = logging.getLogger("docling")
    if enabled:
        formatter = _DoclingProgressFormatter(datefmt="%H:%M:%S")

        has_progress_handler = any(
            getattr(handler, "_ocr_progress_handler", False) for handler in docling_logger.handlers
        )
        if not has_progress_handler:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            handler.setFormatter(formatter)
            handler._ocr_progress_handler = True  # type: ignore[attr-defined]
            docling_logger.addHandler(handler)

        docling_logger.setLevel(level)
        docling_logger.propagate = False

        for logger_name in DOCLING_PROGRESS_LOGGERS:
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            logger.propagate = True

        for logger_name in QUIET_PROGRESS_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        return

    for handler in list(docling_logger.handlers):
        if getattr(handler, "_ocr_progress_handler", False):
            docling_logger.removeHandler(handler)

    docling_logger.setLevel(logging.WARNING)
    docling_logger.propagate = True

    for logger_name in DOCLING_PROGRESS_LOGGERS:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
        logger.propagate = True


def verbose_print(step_name: str, message: str) -> None:
    """Print a consistently formatted progress message."""
    print(f"\n[{step_name.upper()}] ---------------------------------")
    print(message)
    print("-" * (len(step_name) + 37))


def get_latest_ocr_models(repo_id: str = DEFAULT_OCR_REPO_ID) -> dict[str, str]:
    """Resolve the latest available OCR ONNX models from Hugging Face."""
    verbose_print(
        "Model Resolution",
        f"Connecting to Hugging Face to fetch latest models from '{repo_id}'...",
    )
    snapshot_dir = Path(snapshot_download(repo_id=repo_id))
    print(f"  - Repository snapshot verified at: {snapshot_dir}")

    onnx_files = list(snapshot_dir.rglob("*.onnx"))
    categorized: dict[str, list[tuple[float, Path]]] = {"det": [], "cls": [], "rec": []}

    for file_path in onnx_files:
        path_str = str(file_path).lower()

        if "_det_" in path_str:
            role = "det"
        elif "_cls_" in path_str:
            role = "cls"
        elif "_rec_" in path_str:
            role = "rec"
        else:
            continue

        versions = re.findall(r"ocrv(\d+)|v(\d+\.\d+)|v(\d+)", path_str)
        version_val = 0.0
        if versions:
            flat_matches = [group for match in versions for group in match if group]
            if flat_matches:
                try:
                    version_val = float(flat_matches[0])
                except ValueError:
                    version_val = 1.0

        categorized[role].append((version_val, file_path))

    resolved_paths: dict[str, str] = {}
    for role, models in categorized.items():
        if not models:
            raise FileNotFoundError(f"Could not find any ONNX files for task: {role}")

        models.sort(key=lambda item: item[0], reverse=True)
        _, best_path = models[0]
        resolved_paths[role] = str(best_path)

    return resolved_paths


def generate_single_page_pdf(original_pdf_path: str, temp_pdf_path: str) -> None:
    """Slice the first PDF page into a temporary one-page PDF."""
    verbose_print(
        "PDF Slicer",
        f"Slicing Page 1 from '{original_pdf_path}' to avoid multi-page warnings...",
    )
    reader = PdfReader(original_pdf_path)
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    with open(temp_pdf_path, "wb") as file_handle:
        writer.write(file_handle)


def detect_pdf_language(
    pdf_path: str,
    det_model: str,
    rec_model: str,
    cls_model_path: str,
    default_lang: str = "en",
) -> list[str]:
    """Detect the PDF language from native text first, then fall back to OCR."""
    verbose_print("Digital Text Extraction", "Scanning PDF Page 1 metadata for native digital text...")
    try:
        reader = PdfReader(pdf_path)
        page1_text = reader.pages[0].extract_text()
        if page1_text and len(page1_text.strip()) > 50:
            detected_lang = detect(page1_text)
            verbose_print(
                "Language Detection",
                f"Native digital text found. Detected language: ['{detected_lang}']",
            )
            return [detected_lang]
    except Exception as exc:
        print(f"  - Native text check bypassed: {exc}")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf_path = temp_pdf.name

    try:
        generate_single_page_pdf(pdf_path, temp_pdf_path)
        verbose_print("Pre-Flight OCR", "Processing Page 1 through fast OCR for language inspection...")

        temp_pipeline = PdfPipelineOptions()
        temp_pipeline.ocr_options = RapidOcrOptions(
            lang=[default_lang],
            det_model_path=det_model,
            rec_model_path=rec_model,
            cls_model_path=cls_model_path,
        )
        temp_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=temp_pipeline)}
        )
        temp_result = temp_converter.convert(temp_pdf_path)
        temp_md = temp_result.document.export_to_markdown()

        detected_lang = detect(temp_md)
        verbose_print(
            "Language Detection",
            f"Pre-flight OCR completed. Detected language: ['{detected_lang}']",
        )
        return [detected_lang]
    except Exception as exc:
        verbose_print(
            "Pre-Flight Failed",
            f"Language pre-flight aborted: {exc}. Defaulting to ['{default_lang}'].",
        )
        return [default_lang]
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


def run_dynamic_docling_pipeline(
    pdf_path: str,
    det_model_path: str,
    rec_model_path: str,
    cls_model_path: str,
    layout_spec=DOCLING_LAYOUT_HERON,
    default_lang: str = "en",
    force_full_page_ocr: bool = False,
    do_table_structure: bool = True,
    do_picture_description: bool = False,
    vlm_type: str = "smolvlm",
    custom_vlm_repo: Optional[str] = None,
    picture_description_prompt: Optional[str] = None,
    images_scale: float = 2.0,
    enable_progress_logging: bool = False,
) -> str:
    """Run the Docling OCR pipeline and return markdown output."""
    configure_docling_progress_logging(enabled=enable_progress_logging)

    detected_langs = detect_pdf_language(
        pdf_path=pdf_path,
        det_model=det_model_path,
        rec_model=rec_model_path,
        cls_model_path=cls_model_path,
        default_lang=default_lang,
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.ocr_options = RapidOcrOptions(
        lang=detected_langs,
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        cls_model_path=cls_model_path,
        force_full_page_ocr=force_full_page_ocr,
    )
    pipeline_options.layout_options.model_spec = layout_spec
    pipeline_options.do_table_structure = do_table_structure

    if do_picture_description:
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = images_scale
        pipeline_options.do_picture_description = True

        if vlm_type.lower() == "smolvlm":
            pipeline_options.picture_description_options = smolvlm_picture_description
            vlm_selection_log = "SmolVLM-256M (lightweight / low VRAM)"
        elif vlm_type.lower() == "granite":
            pipeline_options.picture_description_options = granite_picture_description
            vlm_selection_log = "Granite-Vision-2B (heavy / high accuracy)"
        elif vlm_type.lower() == "custom" and custom_vlm_repo:
            pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
                repo_id=custom_vlm_repo
            )
            vlm_selection_log = f"Custom HF model ({custom_vlm_repo})"
        else:
            raise ValueError(
                f"Unknown VLM type selection: '{vlm_type}' or missing custom repository."
            )

        if picture_description_prompt:
            pipeline_options.picture_description_options.prompt = picture_description_prompt
    else:
        vlm_selection_log = "Disabled"

    verbose_print(
        "Configuring Final Pipeline",
        f"- OCR Detection Model: {Path(det_model_path).name}\n"
        f"- OCR Recognition Model: {Path(rec_model_path).name}\n"
        f"- OCR Classifier Model: {Path(cls_model_path).name}\n"
        f"- Target Languages: {detected_langs}\n"
        f"- Layout Model Engine: {layout_spec.name}\n"
        f"- Force Full Page OCR: {force_full_page_ocr}\n"
        f"- Do Table Parsing: {do_table_structure}\n"
        f"- Chart VLM Model Selected: {vlm_selection_log}\n"
        f"- Image Scale Multiplier: {images_scale if do_picture_description else 'N/A'}\n"
        f"- Internal Progress Logging: {enable_progress_logging}",
    )

    final_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    verbose_print("Structural Analysis", "Running layout mapping, OCR, and VLM analysis...")
    final_result = final_converter.convert(pdf_path)

    verbose_print("Conversion Success", "Final output generation completed successfully.")
    return final_result.document.export_to_markdown()