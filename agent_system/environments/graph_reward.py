import os
import sys
import json
import re
import glob
import networkx as nx
import pydot
import openai
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import ray

dot_parse_lock = threading.Lock()


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
    """Extract DOT graph from response text"""
    text = remove_think_tag(text)
    if "```dot" in text:
        parts = text.split("```dot")
        if len(parts) > 1:
            end_parts = parts[1].split("```")
            return end_parts[0].strip()
    elif "```" in text and "digraph" in text:
        lines = text.split('\\n')
        in_code_block = False
        dot_lines = []
        for line in lines:
            if line.strip().startswith("```"):
                if in_code_block:
                    break
                else:
                    in_code_block = True
                    continue
            if in_code_block:
                dot_lines.append(line)
        return '\\n'.join(dot_lines)
    return text

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
        
        response = requests.post(self.base_url, json=payload, verify=False, headers=headers)
        
        response = response.json()

        response['choices'][0]['message']['content'] = remove_think_tag(response['choices'][0]['message']['content'])
        return response

def get_api_config(model_name: str) -> tuple[str, str]:
    api_key = os.environ.get("LLM_JUDGE_API_KEY", "")
    base_url = os.environ.get("LLM_JUDGE_BASE_URL", "http://localhost:8001/v1/chat/completions")
    return api_key, base_url

EVALUATION_MODELS = {
    "qwen3": "qwen3-235b",
}


class GraphEvaluator:
    def __init__(self, graph, core_idea, entities, clients):
        
        self.standard_edge_types = {
            'deduction-rule', 'deduction-case',
            'induction-case', 'induction-common', 
            'abduction-phenomenon', 'abduction-knowledge'
        }
        
        self.reasoning_pairs = {
            'deductive': ('deduction-rule', 'deduction-case'),
            'inductive': ('induction-case', 'induction-common'),
            'abductive': ('abduction-phenomenon', 'abduction-knowledge')
        }
        
        self.dot_content = extract_dot_graph_from_text(graph)
        self.core_idea_entities = entities

        self.clients = clients
        

    def load_data(self):
        try:
            with dot_parse_lock:
                dot_content = self.dot_content
                graphs = pydot.graph_from_dot_data(dot_content)
                if not graphs or len(graphs) == 0:
                    return False
                
                self.graph = graphs[0]
            
            node_count = len(self.graph.get_nodes()) if self.graph.get_nodes() else 0
            if node_count == 0:
                return False
                
        except Exception as parse_error:
            return False
        return True


    def calculate_entity_coverage_from_correct_reasoning(self, reasoning_validation_results: Dict) -> float:
        
        if not self.core_idea_entities:
            return 0.0
        
        all_steps, valid_steps = self.filter_valid_reasoning_steps()
        
        correct_step_nodes = set()
        
        step_id = 1
        all_step_targets = []
        
        edges_by_target = {}
        for edge in self.graph.get_edges():
            source_raw = edge.get_source()
            target_raw = edge.get_destination()
            label_raw = edge.get_label()
            
            if hasattr(source_raw, 'strip'):
                source = source_raw.strip('"')
            else:
                source = str(source_raw).strip('"')
                
            if hasattr(target_raw, 'strip'):
                target = target_raw.strip('"')
            else:
                target = str(target_raw).strip('"')
                
            if label_raw:
                if hasattr(label_raw, 'strip'):
                    edge_label = label_raw.strip('"')
                else:
                    edge_label = str(label_raw).strip('"')
            else:
                edge_label = ""
            
            if edge_label in self.standard_edge_types:
                if target not in edges_by_target:
                    edges_by_target[target] = []
                edges_by_target[target].append((source, edge_label))
        
        for target_node, edges in edges_by_target.items():
            source_nodes = [source for source, _ in edges]
            edge_types = [label for _, label in edges]
            all_step_targets.append((target_node, source_nodes, edge_types))
        
        for i, (target_node, source_nodes, edge_types) in enumerate(all_step_targets, 1):
            step_result = reasoning_validation_results.get(str(i), reasoning_validation_results.get(i))
            if step_result == "correct":

                correct_step_nodes.add(target_node)
                correct_step_nodes.update(source_nodes)
        
        if not correct_step_nodes:
            return 0.0
        
        reasoning_entities = set()
        processed_nodes = 0
        
        for node in self.graph.get_nodes():
            name_raw = node.get_name()
            if hasattr(name_raw, 'strip'):
                node_name = name_raw.strip('"')
            else:
                node_name = str(name_raw).strip('"')
                
            if node_name and node_name != 'node' and node_name in correct_step_nodes:
                processed_nodes += 1
                label_raw = node.get_label()
                if label_raw:
                    if hasattr(label_raw, 'strip'):
                        label = label_raw.strip('"')
                    else:
                        label = str(label_raw).strip('"')
                else:
                    label = None
                
                if label:
                    source_x = source_y = source_z = None
                    
                    match = re.match(r'Source:\s*\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)\s*[\n\r]*(.*)', label, re.DOTALL)
                    if match:
                        source_x, source_y, source_z = int(match.group(1)), int(match.group(2)), int(match.group(3))
                        original_content, _ = self._get_original_content_from_source_with_node(source_x, source_y, source_z, node.get_name())
                    else:
                        match = re.match(r'\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)\s*(.*)', label)
                        if match:
                            source_x, source_y, source_z = int(match.group(1)), int(match.group(2)), int(match.group(3))
                            original_content, _ = self._get_original_content_from_source_with_node(source_x, source_y, source_z, node.get_name())
                        else:
                            match = re.match(r'\((-?\d+),\s*(-?\d+)\)\s*(.*)', label)
                            if match:
                                source_x, source_y = int(match.group(1)), int(match.group(2))
                                original_content, _ = self._get_original_content_from_source_with_node(source_x, source_y, node.get_name())
                            else:
                                original_content, _ = self._get_original_content_from_source_with_node(0, 0, 0, node.get_name())
                    
                    if original_content and not original_content.startswith('['):
                        single_words = re.findall(r'\b[A-Za-z]+\b', original_content)
                        compound_terms = re.findall(r'\b[A-Za-z]+(?:[-\s][A-Za-z]+)+\b', original_content)
                        all_terms = single_words + compound_terms
                        
                        stop_words = {
                            'the', 'and', 'are', 'for', 'with', 'that', 'this', 'can', 'may', 'will', 
                            'has', 'have', 'been', 'more', 'such', 'also', 'used', 'use', 'than', 
                            'these', 'they', 'from', 'into', 'over', 'under', 'their', 'there', 
                            'where', 'when', 'what', 'which', 'while', 'through', 'but', 'not',
                            'all', 'any', 'both', 'each', 'few', 'most', 'other', 'some', 'such',
                            'only', 'own', 'same', 'so', 'then', 'very', 'just', 'now', 'how',
                            'its', 'our', 'out', 'way', 'many', 'could', 'would', 'should'
                        }
                        
                        filtered_words = [
                            word.strip() for word in all_terms 
                            if len(word.strip()) >= 2 and 
                            word.lower().strip() not in stop_words
                        ]
                        reasoning_entities.update(filtered_words)
        if not reasoning_entities:
            return 0.0
        
        core_entities_lower = [entity.lower() for entity in self.core_idea_entities]
        reasoning_entities_lower = [entity.lower() for entity in reasoning_entities]
        
        covered_entities = []
        for core_entity in core_entities_lower:
            for reasoning_entity in reasoning_entities_lower:
                if core_entity in reasoning_entity or reasoning_entity in core_entity:
                    covered_entities.append(core_entity)
                    break
        
        coverage_rate = len(covered_entities) / len(self.core_idea_entities) * 100
        
        return coverage_rate

    def filter_valid_reasoning_steps(self) -> Tuple[List[Tuple], List[Tuple]]:
 
        edges_by_target = {}
        non_standard_edges = []
        
        for edge in self.graph.get_edges():
            source_raw = edge.get_source()
            target_raw = edge.get_destination()
            label_raw = edge.get_label()
            
            if hasattr(source_raw, 'strip'):
                source = source_raw.strip('"')
            else:
                source = str(source_raw).strip('"')
                
            if hasattr(target_raw, 'strip'):
                target = target_raw.strip('"')
            else:
                target = str(target_raw).strip('"')
                
            if label_raw:
                if hasattr(label_raw, 'strip'):
                    edge_label = label_raw.strip('"')
                else:
                    edge_label = str(label_raw).strip('"')
            else:
                edge_label = ""
            
            if target not in edges_by_target:
                edges_by_target[target] = []
            
            if edge_label not in self.standard_edge_types:
                original_label = edge_label
                if edge_label == "":
                    edge_label = "unlabeled_edge"
                elif "style=dashed" in str(edge_label) or edge_label == "dashed" or edge_label == "style=dashed":
                    edge_label = "dashed_edge" 
                elif edge_label == "solid" or "style=solid" in str(edge_label):
                    edge_label = "unlabeled_solid_edge" 
                else:
                    edge_label = f"invalid_edge:{edge_label}" 
                
                non_standard_edges.append((source, target, edge_label))
            
            edges_by_target[target].append((source, edge_label))
        
        
        all_steps = []
        valid_steps = []
        
        for target_node, edges in edges_by_target.items():
            source_nodes = [source for source, _ in edges]
            edge_types = [label for _, label in edges]
            
            step_info = (target_node, source_nodes, edge_types)
            all_steps.append(step_info)
            
            has_non_standard_edges = any(edge_type not in self.standard_edge_types for edge_type in edge_types)
            
            if has_non_standard_edges:
                non_standard_types = [et for et in edge_types if et not in self.standard_edge_types]

            elif len(edges) >= 2:
                if self._validate_edge_pairing(edge_types):
                    valid_steps.append(step_info)
        
        return all_steps, valid_steps

    def _validate_edge_pairing(self, edge_types: List[str]) -> bool:
        edge_set = set(edge_types)
        edge_counts = {edge_type: edge_types.count(edge_type) for edge_type in edge_set}
    
        if 'deduction-rule' in edge_set and 'deduction-case' in edge_set:
            if (edge_counts.get('deduction-rule', 0) == 1 and 
                edge_counts.get('deduction-case', 0) == 1 and
                len(edge_types) == 2):
                return True
        
        if 'abduction-phenomenon' in edge_set and 'abduction-knowledge' in edge_set:
            if (edge_counts.get('abduction-phenomenon', 0) == 1 and 
                edge_counts.get('abduction-knowledge', 0) == 1 and
                len(edge_types) == 2):
                return True
        
        if 'induction-common' in edge_set and 'induction-case' in edge_set:
            if (edge_counts.get('induction-common', 0) == 1 and 
                edge_counts.get('induction-case', 0) >= 1 and
                edge_counts.get('induction-common', 0) + edge_counts.get('induction-case', 0) == len(edge_types)):
                return True
        
        return False

    def _get_reasoning_type(self, edge_types: List[str]) -> str:
        edge_set = set(edge_types)
        
        for reasoning_type, (type1, type2) in self.reasoning_pairs.items():
            if type1 in edge_set and type2 in edge_set:
                return reasoning_type
        
        return "unknown"
    
    def _remove_reasoning_prefixes(self, text: str) -> str:
        reasoning_prefixes = [
            "Deduction reasoning: ", "Induction reasoning: ", "Abduction reasoning: ",
            "deduction-reasoning: ", "induction-reasoning: ", "abduction-reasoning: ",
            "Phenomenon: ", "Currently there is evidence that ", "Currently there is knowledge that ",
            "Currently, ", "Currently "
        ]
        
        for prefix in reasoning_prefixes:
            if text.startswith(prefix):
                return text[len(prefix):].strip()
        
        return text

    def _get_original_content_from_source_with_node(self, source_x: int, source_y: int, source_z: int = None, node_name: str = None) -> Tuple[str, bool]:
        try:
            if node_name:

                for node in self.graph.get_nodes():
                    name_raw = node.get_name()
                    if hasattr(name_raw, 'strip'):
                        current_node_name = name_raw.strip('"')
                    else:
                        current_node_name = str(name_raw).strip('"')
                    if current_node_name == node_name:
                        label_raw = node.get_label()
                        if label_raw:
                            if hasattr(label_raw, 'strip'):
                                node_label = label_raw.strip('"')
                            else:
                                node_label = str(label_raw).strip('"')
                        else:
                            node_label = ''

                        transcription = node_label
                        
                        match = re.match(r'Source:\s*\(.*?\)\s*[\n\r]*(.*)', node_label, re.DOTALL)
                        if match:
                            transcription = match.group(1).strip()
                        else:
                            match = re.match(r'\(.*?\)\s*(.*)', node_label)
                            if match:
                                transcription = match.group(1).strip()
                        
                        transcription = self._remove_reasoning_prefixes(transcription)
                        return transcription, True
            return "Supplementary content/background knowledge", True
                
        except Exception as e:
            return "Error retrieving content", False

    def generate_reasoning_validation_prompts_for_steps(self, steps: List[Tuple]) -> List[Dict]:
        prompts = []
        
        def get_node_label(node_name):
            for node in self.graph.get_nodes():
                name_raw = node.get_name()
                if hasattr(name_raw, 'strip'):
                    current_node_name = name_raw.strip('"')
                else:
                    current_node_name = str(name_raw).strip('"')
                if current_node_name == node_name:
                    label_raw = node.get_label()
                    if label_raw:
                        if hasattr(label_raw, 'strip'):
                            return label_raw.strip('"')
                        else:
                            return str(label_raw).strip('"')
                    else:
                        return ''
            return ''
        
        def get_edge_label(source_node, target_node):
            for edge in self.graph.get_edges():
                source_raw = edge.get_source()
                target_raw = edge.get_destination()
                label_raw = edge.get_label()

                if hasattr(source_raw, 'strip'):
                    edge_source = source_raw.strip('"')
                else:
                    edge_source = str(source_raw).strip('"')
                    
                if hasattr(target_raw, 'strip'):
                    edge_target = target_raw.strip('"')
                else:
                    edge_target = str(target_raw).strip('"')
                
                if edge_source == source_node and edge_target == target_node:
                    if label_raw:
                        if hasattr(label_raw, 'strip'):
                            return label_raw.strip('"')
                        else:
                            return str(label_raw).strip('"')
                    else:
                        return 'unknown'
            return 'unknown'
        
        for i, (target_node, source_nodes, edge_types) in enumerate(steps, 1):
            try:
                target_label = get_node_label(target_node)
                
                target_source_match = re.search(r'Source:\s*\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)', target_label)
                if target_source_match:
                    target_source_x, target_source_y, target_source_z = int(target_source_match.group(1)), int(target_source_match.group(2)), int(target_source_match.group(3))
                    target_content, _ = self._get_original_content_from_source_with_node(target_source_x, target_source_y, target_source_z, target_node)
                else:
                    target_source_match = re.search(r'\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)', target_label)
                    if target_source_match:
                        target_source_x, target_source_y, target_source_z = int(target_source_match.group(1)), int(target_source_match.group(2)), int(target_source_match.group(3))
                        target_content, _ = self._get_original_content_from_source_with_node(target_source_x, target_source_y, target_source_z, target_node)
                    else:
                        target_source_match = re.search(r'\((-?\d+),\s*(-?\d+)\)', target_label)
                        if target_source_match:
                            target_source_a, target_source_b = int(target_source_match.group(1)), int(target_source_match.group(2))
                            target_content, _ = self._get_original_content_from_source_with_node(target_source_a, target_source_b, node_name=target_node)
                        else:
                            target_content = target_label
                
                source_contents = []
                actual_edge_types = []

                for source_node in source_nodes:
                    source_label = get_node_label(source_node)

                    source_match = re.search(r'Source:\s*\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)', source_label)
                    if source_match:
                        source_x, source_y, source_z = int(source_match.group(1)), int(source_match.group(2)), int(source_match.group(3))
                        content, _ = self._get_original_content_from_source_with_node(source_x, source_y, source_z, source_node)
                        source_contents.append(content)
                    else:
                        source_match = re.search(r'\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)', source_label)
                        if source_match:
                            source_x, source_y, source_z = int(source_match.group(1)), int(source_match.group(2)), int(source_match.group(3))
                            content, _ = self._get_original_content_from_source_with_node(source_x, source_y, source_z, source_node)
                            source_contents.append(content)
                        else:
                            source_match = re.search(r'\((-?\d+),\s*(-?\d+)\)', source_label)
                            if source_match:
                                source_a, source_b = int(source_match.group(1)), int(source_match.group(2))
                                content, _ = self._get_original_content_from_source_with_node(source_a, source_b, node_name=source_node)
                                source_contents.append(content)
                            else:
                                source_contents.append(source_label)
                    
                    edge_type = get_edge_label(source_node, target_node)
                    actual_edge_types.append(edge_type)
                
                reasoning_type = self._get_reasoning_type(actual_edge_types)
                
                premise_descriptions = []
                
                for content, edge_type in zip(source_contents, actual_edge_types):
                    if reasoning_type == "deductive":
                        if edge_type == "deduction-rule":
                            premise_descriptions.append(f"General principle/rule ({edge_type}): {content}")
                        elif edge_type == "deduction-case":
                            premise_descriptions.append(f"Specific observation/case ({edge_type}): {content}")
                    elif reasoning_type == "inductive":
                        if edge_type == "induction-case":
                            premise_descriptions.append(f"Specific case/observation ({edge_type}): {content}")
                        elif edge_type == "induction-common":
                            premise_descriptions.append(f"Common pattern/generalization ({edge_type}): {content}")
                    elif reasoning_type == "abductive":
                        if edge_type == "abduction-phenomenon":
                            premise_descriptions.append(f"Observed phenomenon ({edge_type}): {content}")
                        elif edge_type == "abduction-knowledge":
                            premise_descriptions.append(f"Background knowledge ({edge_type}): {content}")
                    else:
                        premise_descriptions.append(f"Supporting evidence ({edge_type}): {content}")
                
                validation_prompt = f"""Please evaluate this {reasoning_type} reasoning step for logical correctness.

REASONING STRUCTURE:
{chr(10).join([f"{j+1}. {desc}" for j, desc in enumerate(premise_descriptions)])}

CONCLUSION:
{target_content}

EVALUATION CRITERIA:
1. Logical validity: Does the conclusion logically follow from the premises?
2. Scientific soundness: Is the reasoning scientifically appropriate?
3. Completeness: Are the premises sufficient to support the conclusion?
4. Consistency: Is the reasoning internally consistent?

Please respond in JSON format:
{{
    "result": "correct" or "wrong",
    "reason": "Brief explanation of your evaluation focusing on logical structure and scientific validity"
}} /no_think"""

                
                prompt_info = {
                    "reasoning_id": i,
                    "target_node": target_node,
                    "source_nodes": source_nodes,
                    "edge_types": edge_types,
                    "actual_edge_types": actual_edge_types,
                    "reasoning_type": reasoning_type,
                    "target_content": target_content,
                    "source_contents": source_contents,
                    "premise_descriptions": premise_descriptions,
                    "validation_prompt": validation_prompt
                }

                prompts.append(prompt_info)
                
            except Exception as e:
                continue
        
        return prompts

    def _validate_single_model(self, model_key: str, model_name: str, prompt_info: Dict) -> Dict:
        reasoning_id = prompt_info['reasoning_id']
        validation_prompt = prompt_info['validation_prompt']
        
        system_message = "You are an expert at evaluating logical reasoning in scientific contexts. Always respond in valid JSON format."
        
        try:
            create_params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": validation_prompt}
                ]
            }
            

            create_params.update({
                "temperature": 0,
                "max_tokens": 200
            })
            
            response = self.clients[model_key].create(**create_params)
            result_text = response['choices'][0]['message']['content'].strip()
            
            try:
                result_json = json.loads(result_text)
                result = result_json.get("result", "error").lower()
                reason = result_json.get("reason", "No reason provided")
            except:
                try:
                    cleaned_text = result_text.replace("```json", "").replace("```", "").strip()
                    result_json = json.loads(cleaned_text)
                    result = result_json.get("result", "error").lower()
                    reason = result_json.get("reason", "No reason provided")
                except:
                    result = "error"
                    reason = "JSON parsing failed"
            
            model_response = {
                "model_key": model_key,
                "model_name": model_name,
                "result": result,
                "reason": reason,
                "raw_response": result_text,
                "success": True
            }
            
            return model_response
            
        except Exception as e:
            return {
                "model_key": model_key,
                "model_name": model_name,
                "result": "error",
                "reason": f"API call failed: {e}",
                "raw_response": "",
                "success": False
            }

    def _vote_on_results(self, model_results: Dict[str, str]) -> Tuple[Dict, str]:

        vote_counts = {}
        for result in model_results.values():
            vote_counts[result] = vote_counts.get(result, 0) + 1
        
        vote_breakdown = {
            "votes": model_results,
            "counts": vote_counts,
            "total_models": len(model_results)
        }
        
        valid_results = {k: v for k, v in vote_counts.items() if k != "error"}
        
        if not valid_results:
            final_result = "error"
            vote_breakdown["decision"] = "All models failed"
        elif len(valid_results) == 1:
            final_result = list(valid_results.keys())[0]
            vote_breakdown["decision"] = f"Unanimous: {final_result}"
        else:
            max_votes = max(valid_results.values())
            winners = [result for result, count in valid_results.items() if count == max_votes]
            
            if len(winners) == 1:
                final_result = winners[0]
                vote_breakdown["decision"] = f"Majority: {final_result} ({max_votes}/{len(model_results)})"
            else:
                if "wrong" in winners:
                    final_result = "wrong"
                    vote_breakdown["decision"] = f"Tie broken in favor of 'wrong'"
                else:
                    final_result = winners[0] 
                    vote_breakdown["decision"] = f"Tie: defaulted to {final_result}"
        
        return vote_breakdown, final_result

    def validate_reasoning_steps_with_llm(self) -> Tuple[float, Dict]:

        all_steps, valid_steps = self.filter_valid_reasoning_steps()
        
        if not all_steps:
            return 0.0, {}
        
        reasoning_validation_results = {}
        
        if valid_steps:
            prompts = self.generate_reasoning_validation_prompts_for_steps(valid_steps)
            inner_max_workers = min(100, len(prompts) * len(EVALUATION_MODELS))
            with ThreadPoolExecutor(max_workers=inner_max_workers) as executor:
                future_to_task = {}
                
                for prompt_info in prompts:
                    reasoning_id = prompt_info['reasoning_id']
                    for model_key, model_name in EVALUATION_MODELS.items():
                        future = executor.submit(self._validate_single_model, model_key, model_name, prompt_info)
                        future_to_task[future] = (reasoning_id, model_key)
                
                all_model_results = {}
                for future in as_completed(future_to_task):
                    reasoning_id, model_key = future_to_task[future]
                    try:
                        model_response = future.result(timeout=60)
                        
                        if reasoning_id not in all_model_results:
                            all_model_results[reasoning_id] = {}
                        all_model_results[reasoning_id][model_key] = model_response
                        
                    except Exception as e:
                        
                        if reasoning_id not in all_model_results:
                            all_model_results[reasoning_id] = {}
                        all_model_results[reasoning_id][model_key] = {
                            "model_key": model_key,
                            "model_name": EVALUATION_MODELS[model_key],
                            "result": "error",
                            "reason": f"Task failed: {e}",
                            "raw_response": "",
                            "success": False
                        }
                
                for reasoning_id, model_responses in all_model_results.items():
                    model_results = {k: v["result"] for k, v in model_responses.items()}
                    vote_result, final_result = self._vote_on_results(model_results)
                    
                    reasoning_validation_results[reasoning_id] = final_result

                    prompt_info = next(p for p in prompts if p['reasoning_id'] == reasoning_id)
                    vote_summary = {
                        "reasoning_id": reasoning_id,
                        "target_node": prompt_info['target_node'],
                        "source_nodes": prompt_info['source_nodes'],
                        "edge_types": prompt_info['edge_types'],
                        "actual_edge_types": prompt_info['actual_edge_types'],
                        "reasoning_type": prompt_info['reasoning_type'],
                        "target_content": prompt_info['target_content'],
                        "source_contents": prompt_info['source_contents'],
                        "premise_descriptions": prompt_info['premise_descriptions'],
                        "model_results": model_results,
                        "model_responses": model_responses,
                        "vote_breakdown": vote_result,
                        "final_result": final_result
                    }
                    
        
        invalid_steps = [step for step in all_steps if step not in valid_steps]
        invalid_count = len(invalid_steps)
        
        current_id = len(valid_steps) + 1
        for i, (target_node, source_nodes, edge_types) in enumerate(invalid_steps):
            reasoning_id = current_id + i
            
            has_non_standard_edges = any(edge_type not in self.standard_edge_types for edge_type in edge_types)
            
            if has_non_standard_edges:

                reasoning_validation_results[str(reasoning_id)] = "non_standard_edge_error"
                non_standard_types = [et for et in edge_types if et not in self.standard_edge_types]
            elif len(edge_types) == 1:
                reasoning_validation_results[str(reasoning_id)] = "single_edge_error"
            else:
                reasoning_validation_results[str(reasoning_id)] = "format_error"
        
        correct_count = sum(1 for result in reasoning_validation_results.values() if result == "correct")
        total_count = len(reasoning_validation_results)
        correctness_rate = (correct_count / total_count * 100) if total_count > 0 else 0
        
        single_edge_count = sum(1 for result in reasoning_validation_results.values() if result == "single_edge_error")
        format_error_count = sum(1 for result in reasoning_validation_results.values() if result == "format_error")
        non_standard_edge_count = sum(1 for result in reasoning_validation_results.values() if result == "non_standard_edge_error")
        llm_verified_count = len(valid_steps)
        
        
        return correctness_rate, reasoning_validation_results

    def calculate_accuracy_score(self, reasoning_validation_results: Dict) -> Dict:
        
        total_steps = len(reasoning_validation_results)
        if total_steps == 0:
            return {
                "total_steps": 0,
                "valid_steps": 0,
                "accuracy_score": 0,
                "details": {}
            }
        
        valid_steps = sum(1 for result in reasoning_validation_results.values() if result == "correct")
        
        accuracy_score = valid_steps / total_steps
        
        accuracy_result = {
            "total_steps": total_steps,
            "valid_steps": valid_steps,
            "invalid_steps": total_steps - valid_steps,
            "accuracy_score": accuracy_score,
            "details": reasoning_validation_results
        }
        
        return accuracy_result


    def evaluate(self):

        self.load_data()
        correctness_rate, reasoning_validation_results = self.validate_reasoning_steps_with_llm()

        coverage_rate = self.calculate_entity_coverage_from_correct_reasoning(reasoning_validation_results)
        
        coverage_result = {
            "coverage_rate": coverage_rate / 100,  # Convert to decimal
        }

        accuracy_result = self.calculate_accuracy_score(reasoning_validation_results)
        
        return coverage_result["coverage_rate"], accuracy_result["accuracy_score"]


@ray.remote(num_cpus=0.01) # Let Ray schedule CPU uniformly
def compute_graph_quality_reward(graph, core_idea, entities):

    clients = {}
    for model_key, model_name in EVALUATION_MODELS.items():
        api_key, base_url = get_api_config(model_name)
        clients[model_key] = APIClient(
            api_key=api_key,
            base_url=base_url)
    try:
        evaluator=GraphEvaluator(graph, core_idea, entities, clients)
        coverage, accuracy = evaluator.evaluate()

        return coverage, accuracy
    except:
        return 0.0, 0.0

def compute_graph_quality_reward_no_ray(graph, core_idea, entities):

    clients = {}
    for model_key, model_name in EVALUATION_MODELS.items():
        api_key, base_url = get_api_config(model_name)
        clients[model_key] = APIClient(
            api_key=api_key,
            base_url=base_url)
    try:
        evaluator=GraphEvaluator(graph, core_idea, entities, clients)
        coverage, accuracy = evaluator.evaluate()
        return coverage, accuracy
    except:
        return 0.0, 0.0







def compute_graph_quality_reward_batch(graph, core_idea, entities):
    futures = [compute_graph_quality_reward.remote(g, i, e) for g, i, e in zip(graph, core_idea, entities)]
    results = ray.get(futures) # Still runs in parallel
    coverage_list, accuracy_list = zip(*results)
    return list(coverage_list), list(accuracy_list)

