import json
import os
from dataclasses import dataclass, field
from typing import List, Any, Dict, Tuple, Optional

import evaluate
import pydot
import nltk
import yake
from sentence_transformers import SentenceTransformer, util
import re

from deepeval.models.summac_model import SummaCModels

from agent_system.environments.bleu.bleu import Bleu

import ray

nltk.download = lambda *args, **kwargs: True

bleu_metric = Bleu()
sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qwen3_embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

summac_model = SummaCModels(model_name="mnli-base", granularity="document")


def remove_think_tag(text):
    start_tag = "<think>"
    end_tag = "</think>\n\n"
    
    if text.startswith(start_tag):
        start_idx = text.find(start_tag)
        end_idx = text.rfind(end_tag)
        
        if end_idx != -1 and end_idx > start_idx:
            return text[end_idx + len(end_tag):]
    return text

def extract_dot_graph_from_text(text: str) -> str:
    text = remove_think_tag(text)
    if "```dot" in text:
        parts = text.split("```dot")
        if len(parts) > 1:
            return parts[1].split("```")[0].strip()

    if "```" in text and "digraph" in text:
        lines = text.split("\n")
        in_block, buf = False, []
        for line in lines:
            if line.strip().startswith("```"):
                if in_block:
                    break
                in_block = True
                continue
            if in_block:
                buf.append(line)
        return "\n".join(buf)

    return text


def parse_dot_graph(dot_str: str) -> Optional[pydot.Dot]:
    try:
        graphs = pydot.graph_from_dot_data(dot_str)
        return graphs[0] if graphs else None
    except Exception:
        return None





@dataclass
class WritingData:
    graph: Optional[pydot.Dot]
    outputs: str
    paper_content: str
    references: List[Any]

    output_text: str = field(init=False)
    paper_kps: List[str] = field(init=False)
    output_kps: List[str] = field(init=False)
    graph_kps: List[str] = field(init=False)

    graph_sentences: List[str] = field(init=False)
    graph_text: str = field(init=False)

    @classmethod
    def from_raw(cls, graph, outputs, paper_content, references):
        dot_graph = parse_dot_graph(extract_dot_graph_from_text(str(graph)))
        outputs = remove_think_tag(outputs)
        return cls(dot_graph, str(outputs), str(paper_content), references)

    def __post_init__(self):
        self.output_text = self.outputs.strip()
        self._extract_graph_sentences()
        self._extract_key_phrases()

    def _remove_reasoning_prefixes(self, text: str) -> str:
        prefixes = [
            r'^Deduction reasoning:\s*',
            r'^Induction reasoning:\s*',
            r'^Abduction reasoning:\s*',
            r'^演绎推理:\s*',
            r'^归纳推理:\s*',
            r'^溯因推理:\s*'
        ]
        for p in prefixes:
            text = re.sub(p, '', text, flags=re.IGNORECASE)
        return text.strip()

    def _extract_graph_sentences(self):
        self.graph_sentences = []
        if self.graph:
            for node in self.graph.get_nodes():
                label = node.get_label()
                if not label:
                    continue
                label = str(label).strip('"')

                m = re.match(r'Source:\s*\(.*?\)\s*[\n\r]*(.*)', label, re.DOTALL)
                if not m:
                    m = re.match(r'\(.*?\)\s*(.*)', label)
                if m:
                    label = m.group(1).strip()

                label = self._remove_reasoning_prefixes(label)
                if label:
                    self.graph_sentences.append(label)

        self.graph_text = " ".join(self.graph_sentences)

    def _extract_key_phrases(self):
        extractor = yake.KeywordExtractor(lan="en", n=3, top=50)
        self.paper_kps = [k for k, _ in extractor.extract_keywords(self.paper_content)]
        self.output_kps = [k for k, _ in extractor.extract_keywords(self.outputs)]
        self.graph_kps = (
            [k for k, _ in extractor.extract_keywords(self.graph_text)]
            if self.graph_text else []
        )


def semantic_coverage(src_kps, tgt_kps, threshold=0.7) -> float:
    if not src_kps or not tgt_kps:
        return 0.0
    e1 = sentence_model.encode(src_kps, convert_to_tensor=True)
    e2 = sentence_model.encode(tgt_kps, convert_to_tensor=True)
    sim = util.cos_sim(e1, e2)
    return (sim >= threshold).any(dim=1).float().mean().item()


def parse_citation_numbers(text):
    """
    Parse all citation numbers from text. Supports:
    [1], [1,2], [1-3], [1–3], [1,3–5]
    Returns set(str).
    """
    citation_pattern = re.compile(r"\[([^\]]+)\]")
    number_set = set()

    matches = citation_pattern.findall(text)
    for match in matches:
        parts = re.split(r"[,\s]+", match)
        for part in parts:
            part = part.strip()
            if not part:
                continue

            if "-" in part or "–" in part:
                sep = "-" if "-" in part else "–"
                try:
                    start, end = part.split(sep)
                    for i in range(int(start), int(end) + 1):
                        number_set.add(str(i))
                except ValueError:
                    continue
            else:
                if part.isdigit():
                    number_set.add(part)

    return number_set


def compute_lexical_similarity_reward(d):
    try:
        r = bleu_metric.compute(predictions=[d.output_text], references=[[d.paper_content]])
        return {"lexical_bleu": r["bleu"]}
    except Exception as e:
        print(f"[ERROR] compute_lexical_similarity_reward failed: {e}")
        return {"lexical_bleu": 0.0}


def compute_semantic_similarity_reward(d):
    try:
        query_emb = qwen3_embedding_model.encode(
            d.output_text,
            prompt_name="query",
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        doc_emb = qwen3_embedding_model.encode(
            d.paper_content,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        sim = qwen3_embedding_model.similarity(query_emb, doc_emb).item()
        sim = (sim + 1)/2
        return {"semantic_similarity": sim}

    except Exception as e:
        print(f"[ERROR] compute_semantic_similarity_reward failed: {e}")
        import traceback
        traceback.print_exc()
        return {"semantic_similarity": 0.0}



def compute_contextual_relevance_reward(d):
    try:
        if not d.graph_sentences:
            return {"contextual_relevance": 0.0}
        eo = sentence_model.encode(d.output_text, convert_to_tensor=True)
        eg = sentence_model.encode(d.graph_sentences, convert_to_tensor=True)
        score = util.cos_sim(eo, eg).mean().item()
        score = (score + 1) / 2
        return {"contextual_relevance": score}
    except Exception as e:
        print(f"[ERROR] compute_contextual_relevance_reward failed: {e}")
        return {"contextual_relevance": 0.0}


def compute_coverage_reward(d):
    try:
        return {
            "paper_coverage": semantic_coverage(d.output_kps, d.paper_kps),
            "graph_coverage": semantic_coverage(d.output_kps, d.graph_kps),
        }
    except Exception as e:
        print(f"[ERROR] compute_coverage_reward failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "paper_coverage": 0.0,
            "graph_coverage": 0.0,
        }


def compute_kp_faithfulness_reward(d):
    try:
        return {
            "kp_faithfulness_graph": semantic_coverage(d.graph_kps, d.output_kps)
        }
    except Exception as e:
        print(f"[ERROR] compute_kp_faithfulness_reward failed: {e}")
        return {
            "kp_faithfulness_graph": 0.0
        }


def compute_kp_consistency_reward(d):
    try:
        return {
            "kp_consistency_paper": semantic_coverage(d.paper_kps, d.output_kps)
        }
    except Exception as e:
        print(f"[ERROR] compute_kp_consistency_reward failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "kp_consistency_paper": 0.0
        }


def compute_entailment_faithfulness_reward(d):
    try:
        if not d.graph_text:
            return {"entailment_faithfulness": 0.0}
        s = summac_model._call(d.output_text, d.graph_text)["score"]
        return {"entailment_faithfulness": (s + 1) / 2}
    except Exception as e:
        print(f"[ERROR] compute_entailment_faithfulness_reward failed: {e}")
        return {"entailment_faithfulness": 0.0}


def compute_entailment_consistency_reward(d):
    try:
        s = summac_model._call(d.output_text, d.paper_content)["score"]
        return {"entailment_consistency": (s + 1) / 2}
    except Exception as e:
        print(f"[ERROR] compute_entailment_consistency_reward failed: {e}")
        import traceback
        traceback.print_exc()
        return {"entailment_consistency": 0.0}

def compute_reference_reward(d):
    """
    Extract citation numbers from outputs and compute recall against d.references['indices'].
    """
    try:
        pred_set_str = parse_citation_numbers(d.outputs)
        pred_set = set(int(x) for x in pred_set_str)

        gt_indices = d.references.get("indices", [])
        gt_set = set(int(x) for x in gt_indices)

        if len(gt_set) == 0:
            return {"reference_recall": 0.0}

        hit = pred_set & gt_set
        recall = len(hit) / len(gt_set)
        return {"reference_recall": recall}

    except Exception as e:
        return {"reference_recall": 0.0}


@ray.remote(num_cpus=0.01)
def compute_writing_reward(data) -> Tuple[float, Dict[str, float]]:
    reward_dict = {}

    reward_fns = [
        compute_lexical_similarity_reward,
        compute_semantic_similarity_reward,
        compute_contextual_relevance_reward,
        compute_coverage_reward,
        compute_kp_faithfulness_reward,
        compute_kp_consistency_reward,
        compute_entailment_faithfulness_reward,
        compute_entailment_consistency_reward,
        compute_reference_reward,
    ]

    for fn in reward_fns:
        reward_dict.update(fn(data))

    return sum(reward_dict.values()), reward_dict

def compute_writing_reward_no_ray(data) -> Tuple[float, Dict[str, float]]:
    reward_dict = {}

    reward_fns = [
        compute_lexical_similarity_reward,
        compute_semantic_similarity_reward,
        compute_contextual_relevance_reward,
        compute_coverage_reward,
        compute_kp_faithfulness_reward,
        compute_kp_consistency_reward,
        compute_entailment_faithfulness_reward,
        compute_entailment_consistency_reward,
        compute_reference_reward,
    ]

    for fn in reward_fns:
        reward_dict.update(fn(data))

    return sum(reward_dict.values()), reward_dict


def compute_writing_reward_batch(graph, outputs, paper_content, references):
    futures = [compute_writing_reward.remote(WritingData.from_raw(g, o, p, r)) for g, o, p, r in zip(graph, outputs, paper_content, references)]
    results = ray.get(futures)
    values, details = zip(*results)
    return values, details