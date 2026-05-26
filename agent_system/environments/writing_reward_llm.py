import json
import os
import re
import requests
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

import ray


def remove_think_tag(text):
    start_tag = "<think>"
    end_tag = "</think>\n\n"
    
    if text.startswith(start_tag):
        start_idx = text.find(start_tag)
        end_idx = text.rfind(end_tag)
        
        if end_idx != -1 and end_idx > start_idx:
            return text[end_idx + len(end_tag):]
    return text


def build_references_list(data: dict) -> str:
    """
    Convert a dict with indices and infos into a references list string.
    """
    idx2title = {
        int(item["idx"]): item["title"]
        for item in data.get("infos", [])
    }

    lines = []
    for idx in data.get("indices", []):
        if idx in idx2title:
            lines.append(f"[{idx}] {idx2title[idx]}")
    
    return "\n".join(lines)


class APIClient:
    api_key = None
    base_url = None

    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url

    def create(self, model, messages, temperature=None, max_tokens=None):
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = requests.post(
            self.base_url,
            json=payload,
            headers=headers,
            verify=False
        ).json()

        response["choices"][0]["message"]["content"] = remove_think_tag(
            response["choices"][0]["message"]["content"]
        )
        return response



@dataclass
class WritingData:
    graph: Any
    outputs: str                # GENERATED introduction
    paper_content: str          # ORIGINAL introduction
    references: List[Any]

    @classmethod
    def from_raw(cls, graph, outputs, paper_content, references):
        return cls(
            graph=graph,
            outputs=str(outputs).strip(),
            paper_content=str(paper_content).strip(),
            references=references
        )




SYSTEM_PROMPT = """You are an expert researcher and academic paper reviewer.
Your task is to evaluate the quality of a generated academic paper Introduction.

You will compare a GENERATED Introduction with the ORIGINAL Introduction
and evaluate it according to academic writing and research standards.

Your evaluation must be strict, objective, and reproducible.
You must NOT assume any information beyond the given inputs.
You must penalize missing, incorrect, or misleading use of references.
"""

USER_PROMPT_TEMPLATE = """Please evaluate the GENERATED Introduction by comparing it against the ORIGINAL Introduction.

Score the GENERATED Introduction on the following dimensions.
Each dimension must be scored with an INTEGER from 1 to 5.

Scoring scale:
- 1 = very poor
- 2 = poor
- 3 = acceptable
- 4 = good
- 5 = excellent

Evaluation dimensions:

1. Background & Context  
   Does the introduction provide sufficient and appropriate background for understanding the research area?

2. Problem Clarity  
   Is the research problem clearly and precisely defined?

3. Motivation & Significance  
   Does the introduction convincingly explain why the problem is important and worth studying?

4. Related Work Positioning  
   Does it properly situate the work within existing research, without distortion or omission?

5. Contribution Clarity  
   Are the main contributions of the paper clearly and concretely stated?

6. Logical Structure  
   Does the introduction follow a clear and reasonable academic structure (e.g., background → gap → contribution)?

7. Coherence & Flow  
   Are the ideas well connected across sentences and paragraphs?

8. Consistency with Original Introduction  
   Is the content logically consistent with the original introduction, without introducing contradictory or incompatible claims?

9. Coverage of Key Points  
   Does it cover the key ideas, arguments, and contributions present in the original introduction?

10. References Usage Correctness  
    Are the references from the provided list used correctly, appropriately, and without omission or misuse?

11. Academic Writing Quality  
    Is the writing formal, precise, and appropriate for an academic paper?

Strictness Instructions:
- When in doubt or if any minor issue, give a lower score rather than a higher score.
- Be conservative in scoring: prefer to under-rate rather than over-rate.
- Only give a high score (4 or 5) if the dimension is clearly excellent, accurate, complete, and academically well-written.

---

After scoring all dimensions, answer the following question:

"Overall, is the GENERATED Introduction better than the ORIGINAL Introduction in terms of academic quality and clarity?"

Answer strictly with:
- "Yes" or
- "No"

---

Output format requirements:
- Output MUST be a Python-style dictionary.
- Keys MUST exactly match the variable names specified below.
- Values for scores MUST be integers from 1 to 5.
- Do NOT include explanations, justifications, or extra text.

Required output format:

{{
  "background_context": int,
  "problem_clarity": int,
  "motivation_significance": int,
  "related_work_positioning": int,
  "contribution_clarity": int,
  "logical_structure": int,
  "coherence_flow": int,
  "consistency_with_original": int,
  "coverage_key_points": int,
  "references_usage_correctness": int,
  "academic_writing_quality": int,
  "generated_better_than_original": "Yes" or "No"
}}

--------------------------------------------------
The inputs are provided below.
[Original Introduction]
{ORIGINAL_INTRODUCTION}

[References List]
{REFERENCES_LIST}

[Generated Introduction to Evaluate]
{GENERATED_INTRODUCTION}
/no_think
"""



def compute_llm_as_judge(
    d: WritingData,
    client: APIClient,
    model: str = "qwen3-235b",
) -> Tuple[float, Dict[str, Any]]:

    score_keys = [
        "background_context",
        "problem_clarity",
        "motivation_significance",
        "related_work_positioning",
        "contribution_clarity",
        "logical_structure",
        "coherence_flow",
        "consistency_with_original",
        "coverage_key_points",
        "references_usage_correctness",
        "academic_writing_quality",
    ]

    MIN_SCORE = 0.0
    MAX_SCORE = 5.0
    DEFAULT_SCORE = 1.0

    try:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            ORIGINAL_INTRODUCTION=d.paper_content,
            REFERENCES_LIST=build_references_list(d.references),
            GENERATED_INTRODUCTION=d.outputs
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        response = client.create(
            model=model,
            messages=messages,
        )

        content = response["choices"][0]["message"]["content"]
        raw_result = json.loads(content)

        clipped_scores = []

        for k in score_keys:
            v = raw_result.get(k, DEFAULT_SCORE)

            try:
                v = float(v)
            except Exception:
                v = DEFAULT_SCORE

            v = max(MIN_SCORE, min(MAX_SCORE, v))

            clipped_scores.append(v)

        score_reward = sum(clipped_scores) / 5.0

        better_flag = raw_result.get("generated_better_than_original", "No")
        better_reward = 1.0 if better_flag == "Yes" else 0.0

        reward = score_reward + better_reward

        judge_result = {}

        for k, v in zip(score_keys, clipped_scores):
            judge_result["llm_" + k] = v / 5.0

        judge_result["llm_generated_better_than_original"] = better_reward

        return reward, judge_result

    except Exception as e:
        print(f"[LLM Judge Fatal Error] {e}")

        fallback_score = DEFAULT_SCORE
        fallback_reward = fallback_score * len(score_keys) / 5.0

        judge_result = {
            "llm_" + k: fallback_score / 5.0
            for k in score_keys
        }
        judge_result["llm_generated_better_than_original"] = 0.0

        return fallback_reward, judge_result


@ray.remote(num_cpus=0.01)
def compute_writing_reward_llm(
    d: WritingData,
) -> Tuple[float, Dict[str, Any]]:

    api_key = os.environ.get("LLM_JUDGE_API_KEY", "")
    base_url = os.environ.get("LLM_JUDGE_BASE_URL", "http://localhost:8001/v1/chat/completions")

    client = APIClient(api_key=api_key, base_url=base_url)

    reward, judge_result = compute_llm_as_judge(d, client)

    return reward, judge_result


def compute_writing_reward_llm_no_ray(
    d: WritingData,
) -> Tuple[float, Dict[str, Any]]:

    api_key = os.environ.get("LLM_JUDGE_API_KEY", "")
    base_url = os.environ.get("LLM_JUDGE_BASE_URL", "http://localhost:8001/v1/chat/completions")

    client = APIClient(api_key=api_key, base_url=base_url)

    reward, judge_result = compute_llm_as_judge(d, client)

    return reward, judge_result


def compute_writing_reward_llm_batch(
    graph,
    outputs,          # generated_introductions
    paper_content,    # original_introductions
    references,
) -> Tuple[List[float], List[Dict[str, Any]]]:


    futures = [compute_writing_reward_llm.remote(WritingData.from_raw(g, o, p, r)) for g, o, p, r in zip(graph, outputs, paper_content, references)]
    results = ray.get(futures)
    values, details = zip(*results)
    return values, details
