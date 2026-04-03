# Prompt Engineering Research Report
**Model:** MBZUAI-IFM/K2-Think-v2  
**Experiments completed:** 8/8

---

## Abstract

This lab investigates the empirical impact of prompt design strategies on large language model output quality. Eight controlled experiments isolate individual variables using K2-Think-v2, with outputs evaluated on a four-axis rubric (accuracy, clarity, format adherence, task relevance) scored 1–10 by an independent model judge.

---

## Methodology

- **Model:** MBZUAI-IFM/K2-Think-v2, temperature 0.8
- **Design:** Controlled A/B pairs — one variable changed per experiment
- **Evaluation:** 4-axis rubric scored by judge model call
- **Metrics:** Composite quality score, response latency, word count

---

## Results

### E01: Zero-Shot vs Few-Shot Prompting

**Category:** Learning Paradigm  
**Task:** Sentiment classification of a mixed movie review  
**Variable:** Presence of in-context examples  
**Hypothesis:** Providing labelled examples (few-shot) produces more consistently formatted and accurate outputs than an instruction-only prompt (zero-shot).

| Metric | Zero-Shot | Few-Shot |
|--------|---------|--------|
| Total Score | 36/10 | 36/10 |
| Accuracy | 9 | 9 |
| Clarity | 8 | 8 |
| Format | 9 | 9 |
| Relevance | 10 | 10 |
| Latency (ms) | 7316 | 28434 |
| Word Count | 1 | 1 |

**Winner:** Tie

**Analysis:** The hypothesis that few-shot prompting yields more consistently formatted and accurate results is not supported by this single comparison, as both outputs are identical and equally correct. The only qualitative difference is the prompting style, which did not affect the returned label. A design insight is that for simple binary classification tasks without explicit formatting requirements, adding examples adds little observable benefit over a clear instruction.

---

### E02: Vague vs Specific Instructions

**Category:** Instruction Quality  
**Task:** Product description for wireless headphones  
**Variable:** Instruction specificity and constraint density  
**Hypothesis:** Highly specific prompts with explicit audience, tone, format, and length constraints produce more useful and targeted outputs than open-ended requests.

| Metric | Vague | Specific |
|--------|---------|--------|
| Total Score | 35/10 | 32/10 |
| Accuracy | 8 | 8 |
| Clarity | 9 | 7 |
| Format | 9 | 10 |
| Relevance | 9 | 7 |
| Latency (ms) | 3884 | 5599 |
| Word Count | 252 | 59 |

**Winner:** Vague

**Analysis:** The hypothesis is partially upheld: the specific prompt yielded a concise, audience‑focused copy that fully met the format constraint, but the vague prompt produced a more comprehensive and clear description. The main qualitative difference lies in scope versus targeting – Output A offers extensive feature detail and logical organization, while Output B delivers a short, on‑point message for remote professionals. This suggests that explicit audience and format constraints improve targeting, yet omitting depth can reduce overall usefulness for a generic product description. A balanced prompt should combine audience focus with a request for key feature coverage.

---

### E03: Direct Answer vs Chain-of-Thought

**Category:** Reasoning Strategy  
**Task:** Multi-step word problem (trains, distance, speed)  
**Variable:** Presence of reasoning scaffold  
**Hypothesis:** Providing an explicit step-by-step reasoning scaffold (chain-of-thought) significantly improves accuracy and transparency on multi-step logical problems.

| Metric | Direct | Chain-of-Thought |
|--------|---------|--------|
| Total Score | 37/10 | 27/10 |
| Accuracy | 9 | 8 |
| Clarity | 9 | 6 |
| Format | 10 | 5 |
| Relevance | 9 | 8 |
| Latency (ms) | 5453 | 11149 |
| Word Count | 2 | 3 |

**Winner:** Direct

**Analysis:** The hypothesis that chain-of-thought improves accuracy and transparency was not supported here. Output A delivered a precise, correctly formatted time with high clarity, whereas Output B added a vague qualifier and deviated from the expected format, reducing readability and fidelity. A key insight is that simply prompting for chain-of-thought does not guarantee the model will actually generate stepwise reasoning; the scaffold must be explicit enough to enforce detailed intermediate steps.

---

### E04: Generic Request vs Expert Persona

**Category:** Role Prompting  
**Task:** Explain compound interest to a 22-year-old  
**Variable:** Persona assignment and audience specification  
**Hypothesis:** Assigning a specific expert persona with defined audience produces more domain-appropriate, pedagogically sound, and audience-tailored responses.

| Metric | No Persona | Expert Persona |
|--------|---------|--------|
| Total Score | 30/10 | 36/10 |
| Accuracy | 9 | 9 |
| Clarity | 7 | 9 |
| Format | 7 | 9 |
| Relevance | 7 | 9 |
| Latency (ms) | 3716 | 5698 |
| Word Count | 167 | 533 |

**Winner:** Expert Persona

**Analysis:** The hypothesis is supported: assigning an expert persona with a defined audience yields a more domain‑appropriate and pedagogically sound answer. Output B is conversational, includes comparative simple‑interest context, tables, and actionable advice tailored to a 22‑year‑old, whereas Output A is a concise textbook definition with minimal engagement. The key design insight is that explicit role and audience instructions steer the model toward richer, audience‑focused structuring and tone. This suggests persona prompts improve relevance and clarity for targeted explanations.

---

### E05: Implicit vs Explicit Output Format

**Category:** Output Control  
**Task:** Extract structured data from a job posting  
**Variable:** Output format specification  
**Hypothesis:** Explicitly specifying the output format (e.g., JSON schema) produces machine-readable, consistently structured outputs versus prose or ad-hoc structure.

| Metric | Implicit | Explicit JSON |
|--------|---------|--------|
| Total Score | 32/10 | 34/10 |
| Accuracy | 10 | 9 |
| Clarity | 9 | 6 |
| Format | 5 | 10 |
| Relevance | 8 | 9 |
| Latency (ms) | 12305 | 3306 |
| Word Count | 40 | 36 |

**Winner:** Explicit JSON

**Analysis:** Explicit JSON format produced a machine-readable structure, confirming the hypothesis. B's strict schema and separated fields gave higher format and relevance scores, whereas A's prose bullets were clearer for humans but less consistent for downstream parsing. When downstream consumption matters, prompt designers should prescribe a concrete output schema.

---

### E06: Minimal vs Rich System Context

**Category:** Context Engineering  
**Task:** Customer support response for a delayed order  
**Variable:** System context depth and specificity  
**Hypothesis:** Providing rich system context (company persona, policies, tone guidelines, escalation rules) produces responses that are more on-brand, policy-compliant, and customer-appropriate.

| Metric | Minimal Context | Rich Context |
|--------|---------|--------|
| Total Score | 30/10 | 32/10 |
| Accuracy | 6 | 8 |
| Clarity | 9 | 7 |
| Format | 7 | 8 |
| Relevance | 8 | 9 |
| Latency (ms) | 6904 | 12249 |
| Word Count | 228 | 137 |

**Winner:** Rich Context

**Analysis:** The hypothesis is supported: the richer system context yielded a response that is more on‑brand and aligns with typical compensation policies, while the minimal context version was slightly more verbose but less realistic. Output B’s use of brand name, discount code, and realistic timeline improves relevance and accuracy, whereas Output A’s overly optimistic promises (hourly update, overnight replacement) hurt accuracy and format completeness. Prompt design insight: include explicit policy references and brand voice cues to guide the model toward realistic, compliant solutions rather than generic placeholders.

---

### E07: Positive vs Negative Instruction Framing

**Category:** Instruction Polarity  
**Task:** Write a persuasive cover letter opening paragraph  
**Variable:** Instruction polarity (do vs don't)  
**Hypothesis:** Positively-framed instructions ("do X") produce more complete, confident outputs than negatively-framed instructions ("avoid Y"), which may cause the model to be overly cautious.

| Metric | Negative Framing | Positive Framing |
|--------|---------|--------|
| Total Score | 34/10 | 36/10 |
| Accuracy | 9 | 9 |
| Clarity | 7 | 9 |
| Format | 9 | 9 |
| Relevance | 9 | 9 |
| Latency (ms) | 4764 | 3482 |
| Word Count | 78 | 68 |

**Winner:** Positive Framing

**Analysis:** The hypothesis holds: Output B, generated from a positively‑framed instruction, is more confident and includes concrete performance metrics, making it more persuasive, while Output A from a negative framing is more cautious and generic. The main qualitative difference is B’s vivid quantitative claim and enthusiastic tone versus A’s hedging language and less specific achievements. This suggests that framing instructions affirmatively elicits assertive, detail‑rich content from the model.

---

### E08: Single-Step vs Decomposed Task

**Category:** Task Architecture  
**Task:** Analyse and improve a weak business pitch  
**Variable:** Task decomposition and subtask specification  
**Hypothesis:** Decomposing a complex task into explicit subtasks produces more thorough, structured, and actionable outputs than issuing the same task as a single compound instruction.

| Metric | Single-Step | Decomposed |
|--------|---------|--------|
| Total Score | 29/10 | 19/10 |
| Accuracy | 5 | 6 |
| Clarity | 9 | 6 |
| Format | 9 | 3 |
| Relevance | 6 | 4 |
| Latency (ms) | 4200 | 4204 |
| Word Count | 589 | 141 |

**Winner:** Single-Step

**Analysis:** The hypothesis was not supported in this case; the single‑step prompt produced a more thorough and better‑structured pitch than the decomposed one. Output A organized the content into clear sections and sub‑points, while Output B remained a dense paragraph lacking explicit headings or actionable recommendations. This suggests that when the target output is a polished pitch rather than a step‑by‑step analysis, a single directed instruction can be more efficient than breaking the task into granular subtasks. For future experiments, align the decomposition granularity with the desired deliverable format to avoid unnecessary fragmentation.

---

## Conclusions

- Hypothesis confirmation rate: **50%** (4/8 experiments)
- Specificity, format constraints, and context are strong prompt engineering levers, but outcomes vary by task
- Chain-of-thought and task decomposition were mixed in this run; direct and single-step prompts won in E03 and E08
- Deliberate prompt design still produces measurable gains when the strategy matches the deliverable

## Reproducibility & Verification

- Export and attach the raw JSON logs (prompts, outputs, rubric scores, latency, and word counts) from the lab UI for independent review.
- Re-run all experiments with the same model/settings to verify whether the same winners and margins replicate.
