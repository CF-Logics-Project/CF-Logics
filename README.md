# CF-Logics

CF-Logics measures Community Foundations' (CF) institutional logics in annual reports with a hybrid AI, human-in-the-loop workflow.&#32;

First, annual reports are transcribed, PII tagged, chunked, and screened. Then, both humans and LLMs perform qualitative coding, and finally, the coding results are compared to evaluate agreement and disagreement.&#32;

The workflow applies deterministic tools when possible, LLMs for tasks necessitating semantic understanding, and sandboxed agent LLMs for qualitative coding.

The repository accompanies the proposal *Measuring Constructs in Organizational Language: A Hybrid AI & Human-in-the-Loop Approach* (Dressel, Gomez Teijeiro & LePere-Schloop). It holds a working prototype of the workflow and the sample of three annual reports we provide results for (from our corpus of more than 2,000).&#32;

Our codebook instructs questions to identify four logics, in text chunks belonging to one or several elements.

We prepared an interface for human coding: [human\_labeling\_interface.html](https://github.com/cdressel77/CF-Logics/blob/main/human_labeling_interface.html). For LLM agents, the codebook is  embedded in the script [`inference_revised.py`](https://github.com/cdressel77/CF-Logics/blob/main/inference_revised.py).&#32;

**Logics.** Each logic is an ideal type: a coherent set of values about community and philanthropy, and of practices a CF follows.

| Logic | Definition |
| --- | --- |
| Philanthropic Community Trust (PPG) | The CF acts as a community charitable bank. It grows and preserves unrestricted endowments and bequests, builds a community legacy and responds to emerging local needs. Community means the shared identity, values and traditions of people from one place. |
| Donor Services (DS) | The CF acts as a philanthropic wealth advisor. It mobilizes donors and stewards their funds, mainly donor-advised funds, so that each donor can leave a personal legacy. Community means the interests, values and identities of individual donors. |
| Public-Private Governance (PPG) | The CF acts as a community leader and convener. It provides private solutions to public problems through cross-sector collaboration and research-based strategic investment. Community means the interdependence and shared fate of people in one place. |
| Social Change (SG) | The CF acts as an advocate for the marginalized. It works to dismantle barriers that perpetuate disparity and inequality, through participatory, trust-based grantmaking. Community is a socially constructed and contested process. |

**Elements.** An element is a type of content in an annual report through which a logic becomes visible. A chunk belongs to an element when it describes:

| Element | What the chunk describes |
| --- | --- |
| Community | A community, a group of people, a "we", or the ideal of how people build and maintain community |
| Donor (ideal of philanthropy) | The ideal of philanthropy, the importance of charitable giving, donations to CF funds, or support for causes through the CF |
| Activity | The activities of the CF, the value it provides, or the way it works with other organizations |
| Board | The CF's board or committee members |
| Grantee | The causes, organizations and grantees that receive funding or grants from the CF |

## Reproducing the Pipeline:&#32;

Run the steps in order from anywhere: every script resolves its paths from the repository root, and each step reads the output folder of the step before it. Intermediate outputs for the sample reports are stored in `out/`, one folder per step.

| Step | Script | What it does | Reads | Writes |
| --- | --- | --- | --- | --- |
| 1 · OCR | `python OCR.py [--report <report_id>]` | Converts the unstructured text of each PDF report into a Markdown file. | `Sample Annual Report PDFs/<report_id>_AR.pdf` | `out/ocr/out_<report_id>_AR.md` |
| 2 · Entity tagging | `python tagging.py [--report <report_id>]` | Replaces the names of individual people and community foundations with standardized entity tags (Gemini API: set `GEMINI_API_KEY`). | `out/ocr/` | `out/tagging/tagging_out_<report_id>_AR.md` and a `.csv` log of substitutions |
| 3 · Chunking | `python report_chunking.py` | Splits the text into sentences with an NLP sentence splitter and regroups similar adjacent sentences using a cosine similarity threshold. Prompts for the report identifier. | `out/tagging/` | `out/chunking/annual_report_chunks_<report_id>_<version>.csv`: one row per chunk |
| 4 · Screening | `python screen_chunks_regex.py` | Flags irrelevant chunks for removal from the analysis (e.g., lists of names, or text in another language). | `out/chunking/` | `out/screening/screening_annual_report_chunks_<report_id>_<version>.csv`: four yes/no flags; `yes` excludes the chunk |
| 5a · LLM coding | `python inference_revised.py [--provider anthropic\|google] [--report <report_id>]` | Runs the LLM coding of CF logics on the chunks that pass screening; shows how we translated the codebook into questions based on its elements (set `ANTHROPIC_API_KEY` or `GEMINI_API_KEY`). | `out/screening/` | `out/out_coding/ai/output_<report_id>_<version>.csv` and `.json` |
| 5b · Human coding | `human_labeling_interface.html` | Interface the expert coders use to code the same chunks with the same codebook questions; open it in a browser and upload a screening file. | `out/screening/` | `out/out_coding/human/` (CSV export) |
| 6 · Agreement | `repro_scripts/` | Scores the human and LLM codes and compares them. | `out/out_coding/` | Human–LLM agreement statistics; method in `logic_scoring_evaluation.md` |

## A note on pipeline development

This pipeline results from a human-AI iterative effort to build a computational companion for qualitative coding that started from an original expert human coding of CF annual reports `Sample Human Labeling/original labeling/`.&#32;

The iterative journey informed limitations that we addressed in the next attempt and resulted in our current pipeline. `Sample Human Labeling/Fisher_AI_Dressel.pptx`documents the intermediate attempts that informed the development of the current protocol (original human coding, unsupervised and supervised learning on text embeddings, TF-IDF narratives, and one-short classification both with one-shot inference and codebook graphRAG contextualized inference).
 
