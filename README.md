# CF-Logics

CF-Logics measures Community Foundations' (CF) institutional logics in annual reports with a hybrid AI, human-in-the-loop workflow.&#32;

First, annual reports are transcribed, PII tagged, chunked, and screened. Then, both humans and LLMs perform qualitative coding, and finally, the coding results are compared to evaluate agreement and disagreement.&#32;

The workflow applies deterministic tools when possible, LLMs for tasks necessitating semantic understanding, and sandboxed agent LLMs for qualitative coding.

The repository accompanies the proposal *Measuring Constructs in Organizational Language: A Hybrid AI & Human-in-the-Loop Approach* (Dressel, Gomez Teijeiro & LePere-Schloop). It holds a working prototype of the workflow and the sample of three annual reports we provide results for (from our corpus of more than 2,000).&#32;

Our codebook instructs questions to identify four logics, in text chunks belonging to one or several elements.

We prepared an interface for human coding: [human\_labeling\_interface.html](https://github.com/CF-Logics-Project/CF-Logics/blob/main/human_labeling_interface.html), which you can [open directly in your browser](https://cf-logics-project.github.io/CF-Logics/human_labeling_interface.html). For LLM agents, the codebook is  embedded in the script [`inference_revised.py`](https://github.com/CF-Logics-Project/CF-Logics/blob/main/inference_revised.py).&#32;

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
| 3 · Chunking | `python report_chunking.py` | Splits the text into sentences with an NLP sentence splitter and regroups similar adjacent sentences using a cosine similarity threshold. Prompts for the report identifier. | `out/tagging/` | `out/chunking/annual_report_chunks_<report_id>.csv`: one row per chunk |
| 4 · Screening | `python screen_chunks_regex.py` | Flags irrelevant chunks for removal from the analysis (e.g., lists of names, or text in another language). | `out/chunking/` | `out/screening/screening_annual_report_chunks_<report_id>.csv`: four yes/no flags; `yes` excludes the chunk |
| 5a · LLM coding | for direct inference (not what we did) `python inference_revised.py [--provider anthropic\|google] [--report <report_id>]`, the adopted use is script provision as agentic tool use for claude opus 5.5 | Runs the LLM coding of CF logics on the chunks that pass screening; shows how we translated the codebook into questions based on its elements (set `ANTHROPIC_API_KEY` or `GEMINI_API_KEY`). | `out/screening/` | `out/out_coding/ai/output_<report_id>.csv` and `.json` |
| 5b · Human coding | `human_labeling_interface.html` | Interface the expert coders use to code the same chunks with the same codebook questions; open it in a browser and upload a screening file. | `out/screening/` | `out/out_coding/human/` (CSV export) |
| 6 · Agreement | `python agreement_analysis.py`, then `python agreement_xlsx.py` and `python agreement_dashboard.py` | Scores the human and LLM codes and compares them (requires pandas, numpy, scikit-learn; openpyxl for the workbook); the reports to compare are listed in `out/out_coding/manifest.csv`. The dashboard is [published online](https://cf-logics-project.github.io/CF-Logics/agreement_dashboard.html). | `out/out_coding/` | `out/agreement/`: agreement statistics per report and pooled (`agreement_bundle.json`) and a workbook with live formulas; `agreement_dashboard.html`. Method in `logic_scoring_evaluation.md` |

For the sample reports, steps 2 (entity tagging) and 5a (LLM coding) were executed by a code agent (Claude opus 5.5) that read the input (chunks and report) and followed, respectively, the tagging instructions below and the questions in `inference_revised.py`. `tagging.py` and `inference_revised.py` run the same instructions through an API, so non agentic one-shot inference is a less performant alternative.

## A note on pipeline development

This pipeline results from a human-AI iterative effort to build a computational companion for qualitative coding that started from an original expert human coding of CF annual reports `Sample Human Labeling/original labeling/`.&#32;

<details>
<summary>Tagging instructions given to the agent</summary>

> You are a behaving agent, you can directly execute tasks directly, you are not just a code generator. Execute directly this task: read the file out/ocr/out_25-1650208_2021_AR.md, identify the name of the community foundation that writes the report, if the name is preceded by a determined article such as "The", don't consider the article to be part of the name. Also identify if any accronym is used to refer the community foundation and treat it as synonym of the name. You'll need to keep this information in mind for the tagging process. You need to execute all the tagging tasks and then save the new version of the .md file in the folder out/tagging, add to filename the prefix "tagging_" so it inherits the original name of the file with the prefix. The tagging tasks are the following: 1) Identify and replace with "[community_foundation_tag]" the language that the community foundation uses to refer to itself (e.g. [name], acronym of [name], "community foundation", "the foundation", "we", etc.). Be sure not replace with [community_foundation_tag] any funds managed by the organization. 2) Individual tag prompting: Identify and replace with "[individual_tag]" the names of all individual people in this report. Do this replacement only if the name has the purpose of mentioning an individual but not if the name of the individual is a part of the name of a foundation or a fund or any other entity, organization, company. 3) Identifiy and replace with [financial_table_tag] ocurrences of table structures containing only finance related terms and numbers. Replace the entire table with the tag [financial_table_tag]. 4) identify if there is a table of contents in the report, if so replace with [table_of_contents] the entire opening table of contents of the report.

</details>



The iterative journey informed limitations that we addressed in the next attempt and resulted in our current pipeline. `Sample Human Labeling/Fisher_AI_Dressel.pptx`documents the intermediate attempts that informed the development of the current protocol (original human coding, unsupervised and supervised learning on text embeddings, TF-IDF narratives, and one-short classification both with one-shot inference and codebook graphRAG contextualized inference).
 
