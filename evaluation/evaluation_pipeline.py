import os
import json
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
import random
import concurrent.futures
from functools import partial
import shutil

from openai import OpenAI

from data_preprocess.NC_Physics import (
    load_jsonl,
    build_references_list,
    load_example,
    build_paper2graph_prompt,
    build_introduction_writing_prompt,
    build_paper_content
)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from agent_system.environments.writing_reward import WritingData
    from agent_system.environments.writing_reward import compute_writing_reward_no_ray as compute_writing_reward
    from agent_system.environments.writing_reward_llm import compute_writing_reward_llm_no_ray as compute_writing_reward_llm
    from agent_system.environments.graph_reward import compute_graph_quality_reward_no_ray as compute_graph_quality_reward
    EVALUATION_AVAILABLE = True
except ImportError as e:
    EVALUATION_AVAILABLE = False
    print(f"Warning: Evaluation modules not available: {e}")


def compute_writing_reward_batch_concurrent(graph, outputs, paper_content, references, max_workers=1):
    """Compute evaluation results in parallel using concurrent.futures."""
    def compute_single_reward(g, o, p, r):
        try:
            d = WritingData.from_raw(g, o, p, r)
            return compute_writing_reward(d)
        except Exception as e:
            print(f"Error computing reward: {e}")
            return 0.0, {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(compute_single_reward, g, o, p, r): i
                   for i, (g, o, p, r) in enumerate(zip(graph, outputs, paper_content, references))}

        results = [None] * len(futures)
        if TQDM_AVAILABLE:
            with tqdm(total=len(futures), desc="Computing writing rewards") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    index = futures[future]
                    results[index] = future.result()
                    pbar.update(1)
        else:
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                results[index] = future.result()

    values, details = zip(*results)
    return values, details


def compute_writing_llm_reward_batch_concurrent(graph, outputs, paper_content, references, max_workers=1):
    """Compute writing_llm evaluation results in parallel."""
    def compute_single_reward(g, o, p, r):
        try:
            d = WritingData.from_raw(g, o, p, r)
            return compute_writing_reward_llm(d)
        except Exception as e:
            print(f"Error computing writing_llm reward: {e}")
            return 0.0, {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_single_reward, g, o, p, r): i
                   for i, (g, o, p, r) in enumerate(zip(graph, outputs, paper_content, references))}

        results = [None] * len(futures)
        if TQDM_AVAILABLE:
            with tqdm(total=len(futures), desc="Computing writing_llm rewards") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    index = futures[future]
                    results[index] = future.result()
                    pbar.update(1)
        else:
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                results[index] = future.result()

    values, details = zip(*results)
    return values, details


def load_and_process_data(data_file: str, example_dir: str = None, inference_folder: str = None) -> List[Dict[str, Any]]:
    """Load and process data."""
    print(f"Loading data from {data_file}")
    raw_data = load_jsonl(data_file)

    processed_data = []
    for idx, example in enumerate(raw_data):
        paper_content = build_paper_content(example)
        introduction_content = ""
        if example.get('sections') and len(example['sections']) > 0:
            introduction_content = example['sections'][0].get('content', '')
        references = example['sections'][0]['references']
        reference_list = build_references_list(references)

        example_intro = None
        example_graph = None
        if example_dir:
            example_data = load_example(example_dir)
            if example_data["input"] and example_data["output"]:
                if example_data["input"].get("introduction", {}).get("content"):
                    example_intro = example_data["input"]["introduction"]["content"]
                    example_graph = example_data["output"]

        paper2graph_prompt = build_paper2graph_prompt(example_intro, example_graph, paper_content)

        if inference_folder:
            data = {
                "index": idx,
                "unique_id": example.get('unique_id'),
                "paper_content": paper_content,
            }
            processed_data.append(data)
            continue

        data = {
            "index": idx,
            "unique_id": example.get('unique_id'),
            "subfield": example.get('subfield'),
            "core_idea": example.get('core_idea'),
            "entities": example.get('entities'),
            "paper_content": paper_content,
            "introduction": introduction_content,
            "references": references,
            "reference_list": reference_list,
            "step1_input": paper2graph_prompt
        }
        processed_data.append(data)

    print(f"Processed {len(processed_data)} samples")
    return processed_data


class LLMInference:
    """LLM inference client."""
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "gpt-4o", max_tokens: int = None, temperature: float = None):
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL")
        )
        self.model = model
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature

    def call(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        try:
            call_params = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if max_tokens is not None or self.default_max_tokens is not None:
                call_params["max_tokens"] = max_tokens or self.default_max_tokens
            if temperature is not None or self.default_temperature is not None:
                call_params["temperature"] = temperature or self.default_temperature

            response = self.client.chat.completions.create(**call_params)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM API call failed: {e}")
            return ""


def mock_llm_inference(prompt: str) -> str:
    """Mock LLM inference."""
    if "Graphviz DOT" in prompt or "digraph" in prompt:
        return "This is a mock introduction output based on the graph and references provided."
    else:
        return """digraph ReasoningGraph {
    A [label="Source: (1, 0, 0)\\nOriginal research shows interesting phenomena."];
    B [label="Source: (2, 0, 0)\\nCurrent methods have limitations."];
    C [label="Source: (0, 0, 0)\\nDeduction: If we address the limitations, we can achieve better results."];
    A -> C [label="deduction-rule"];
    B -> C [label="deduction-case"];
}"""


def copy_inference_results(src_folder: str, dst_folder: str):
    """Copy inference folder to a new directory."""
    os.makedirs(dst_folder, exist_ok=True)
    for fname in os.listdir(src_folder):
        src_path = os.path.join(src_folder, fname)
        dst_path = os.path.join(dst_folder, fname)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
    print(f"Copied inference results from {src_folder} to {dst_folder}")


def load_inference_results(inference_folder: str, processed_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Load inference results from folder."""
    loaded_data = []
    for data in processed_data:
        unique_id = data.get("unique_id", f"sample_{data['index']}")
        inference_file = os.path.join(inference_folder, f"{unique_id}.json")
        if os.path.exists(inference_file):
            with open(inference_file, 'r', encoding='utf-8') as f:
                loaded_sample = json.load(f)
            loaded_data.append(loaded_sample)
        else:
            print(f"Warning: Inference file not found: {inference_file}")
            loaded_data.append(data)
    print(f"Loaded {len(loaded_data)} inference results")
    return loaded_data


def run_inference_parallel(
        processed_data: List[Dict[str, Any]],
        use_mock: bool = True,
        llm_config: Dict[str, Any] = None,
        results_folder: str = None,
        reward_types: List[str] = None,
        max_workers: int = 4,
        run_mode: str = '2step',
        inference_folder: str = None
) -> List[Dict[str, Any]]:
    """Parallel inference supporting step1 / step2 / 2step modes."""
    print("Running inference with parallel processing...")
    llm = None
    if not use_mock:
        llm_config = llm_config or {}
        llm = LLMInference(**llm_config)

    def process_single_sample(data):
        try:
            if run_mode in ['step1', '2step']:
                step1_input = data["step1_input"]
                if use_mock:
                    step1_output = mock_llm_inference(step1_input)
                else:
                    step1_output = llm.call(step1_input)
                data["step1_output"] = step1_output

            if run_mode in ['step2', '2step']:
                if run_mode == 'step2':
                    unique_id = data.get("unique_id", f"sample_{data['index']}")
                    graph_file = os.path.join(inference_folder, f"{unique_id}.json")
                    with open(graph_file, 'r', encoding='utf-8') as f:
                        loaded = json.load(f)
                    step1_output = loaded["step1_output"]
                    data["step1_output"] = step1_output

                step2_input = build_introduction_writing_prompt(
                    graph=data["step1_output"],
                    references=data["reference_list"]
                )
                if use_mock:
                    step2_output = mock_llm_inference(step2_input)
                else:
                    step2_output = llm.call(step2_input)
                data["step2_output"] = step2_output

            return data
        except Exception as e:
            print(f"Error processing sample {data.get('unique_id', data.get('index', 'unknown'))}: {e}")
            return data

    print(f"Processing {len(processed_data)} samples with {max_workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_data = {executor.submit(process_single_sample, data): data for data in processed_data}
        if TQDM_AVAILABLE:
            results = []
            with tqdm(total=len(processed_data), desc="Processing samples") as pbar:
                for future in concurrent.futures.as_completed(future_to_data):
                    results.append(future.result())
                    pbar.update(1)
        else:
            results = [future.result() for future in concurrent.futures.as_completed(future_to_data)]

    processed_data = sorted(results, key=lambda x: x.get('index', 0))

    if results_folder:
        print("Saving inference results...")
        for data in processed_data:
            unique_id = data.get("unique_id", f"sample_{data['index']}")
            result_file = os.path.join(results_folder, f"{unique_id}.json")
            save_data = {k: v for k, v in data.items() if k != "evaluation"}
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

    print("Inference completed")
    return processed_data


def run_evaluation(processed_data: List[Dict[str, Any]], max_workers: int = 4, results_folder: str = None, reward_types: List[str] = None) -> List[Dict[str, Any]]:
    """Run evaluation with incremental saving and resume support."""
    if not EVALUATION_AVAILABLE:
        print("Evaluation modules not available, skipping evaluation")
        for data in processed_data:
            data["evaluation"] = {
                "total_reward": 0.0,
                "details": {},
                "error": "Evaluation modules not available"
            }
        return processed_data

    if not reward_types:
        reward_types = ['writing']

    print(f"Running evaluation for reward types: {reward_types}")

    to_evaluate = []
    to_evaluate_indices = []
    for i, data in enumerate(processed_data):
        if "evaluation" in data and data["evaluation"].get("details"):
            uid = data.get("unique_id", f"sample_{data.get('index', i)}")
            print(f"Skipping already evaluated: {uid}")
        elif not data.get("step2_output", ""):
            uid = data.get("unique_id", f"sample_{data.get('index', i)}")
            print(f"Skipping empty step2_output: {uid}")
            data["evaluation"] = {
                "total_reward": 0.0,
                "details": {},
                "error": "Empty step2_output"
            }
            if results_folder:
                _save_single_evaluation(data, results_folder)
        else:
            to_evaluate.append(data)
            to_evaluate_indices.append(i)

    print(f"Total: {len(processed_data)}, To evaluate: {len(to_evaluate)}, Already done: {len(processed_data) - len(to_evaluate)}")

    if not to_evaluate:
        print("All samples already evaluated")
        return processed_data

    evaluation_results = {}

    if 'writing' in reward_types:
        print("Computing writing rewards...")
        graphs = [data["step1_output"] for data in to_evaluate]
        outputs = [data["step2_output"] for data in to_evaluate]
        introduction_contents = [data["introduction"] for data in to_evaluate]
        references = [data["references"] for data in to_evaluate]

        try:
            values, details = compute_writing_reward_batch_concurrent(graphs, outputs, introduction_contents, references, max_workers)
            evaluation_results["writing"] = {
                "individual_rewards": values,
                "details": details
            }
        except Exception as e:
            print(f"Writing evaluation failed: {e}")
            evaluation_results["writing"] = {
                "individual_rewards": [0.0] * len(to_evaluate),
                "details": [{}] * len(to_evaluate),
                "error": str(e)
            }

    if 'writing_llm' in reward_types:
        print("Computing writing_llm rewards...")
        graphs = [data["step1_output"] for data in to_evaluate]
        outputs = [data["step2_output"] for data in to_evaluate]
        introduction_contents = [data["introduction"] for data in to_evaluate]
        references = [data["references"] for data in to_evaluate]

        try:
            values, details = compute_writing_llm_reward_batch_concurrent(graphs, outputs, introduction_contents, references, max_workers)
            evaluation_results["writing_llm"] = {
                "individual_rewards": values,
                "details": details
            }
        except Exception as e:
            print(f"Writing_llm evaluation failed: {e}")
            evaluation_results["writing_llm"] = {
                "individual_rewards": [0.0] * len(to_evaluate),
                "details": [{}] * len(to_evaluate),
                "error": str(e)
            }

    if 'graph_quality' in reward_types:
        print("Computing graph quality rewards for step2...")
        try:
            def compute_single_graph_quality_reward(data):
                """Compute graph quality reward for a single sample."""
                try:
                    coverage, accuracy = compute_graph_quality_reward(
                        data["step1_output"],  # graph
                        data.get("core_idea", ""),  # core_idea
                        data.get("entities", [])  # entities
                    )
                    return {"coverage": coverage, "accuracy": accuracy}
                except Exception as e:
                    print(f"Error computing graph quality reward for sample {data.get('unique_id', data.get('index', 'unknown'))}: {e}")
                    return {"coverage": 0.0, "accuracy": 0.0}

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(compute_single_graph_quality_reward, data): i for i, data in enumerate(to_evaluate)}

                step2_graph_quality_rewards = [None] * len(to_evaluate)
                if TQDM_AVAILABLE:
                    with tqdm(total=len(futures), desc="Computing graph quality rewards for step2") as pbar:
                        for future in concurrent.futures.as_completed(futures):
                            index = futures[future]
                            step2_graph_quality_rewards[index] = future.result()
                            pbar.update(1)
                else:
                    for future in concurrent.futures.as_completed(futures):
                        index = futures[future]
                        step2_graph_quality_rewards[index] = future.result()

            evaluation_results["graph_quality_step2"] = {
                "rewards": step2_graph_quality_rewards
            }
        except Exception as e:
            print(f"Arche evaluation for step2 failed: {e}")
            evaluation_results["graph_quality_step2"] = {
                "rewards": [{"coverage": 0.0, "accuracy": 0.0} for _ in to_evaluate],
                "error": str(e)
            }

    for i, data in enumerate(to_evaluate):
        data["evaluation"] = {
            "total_reward": 0.0,
            "details": {}
        }

        total_reward = 0.0

        if "writing" in evaluation_results:
            writing_reward = evaluation_results["writing"]["individual_rewards"][i]
            writing_details = evaluation_results["writing"]["details"][i]
            total_reward += writing_reward
            data["evaluation"]["details"].update(writing_details)

        if "writing_llm" in evaluation_results:
            writing_llm_reward = evaluation_results["writing_llm"]["individual_rewards"][i]
            writing_llm_details = evaluation_results["writing_llm"]["details"][i]
            total_reward += writing_llm_reward
            data["evaluation"]["details"].update(writing_llm_details)

        if "graph_quality_step2" in evaluation_results:
            gq_reward = evaluation_results["graph_quality_step2"]["rewards"][i]
            gq_coverage = gq_reward["coverage"]
            gq_accuracy = gq_reward["accuracy"]
            gq_total = gq_coverage + gq_accuracy
            total_reward += gq_total
            data["evaluation"]["details"]["gq_coverage"] = gq_coverage
            data["evaluation"]["details"]["gq_accuracy"] = gq_accuracy

        data["evaluation"]["total_reward"] = total_reward

        if results_folder:
            _save_single_evaluation(data, results_folder)
            uid = data.get("unique_id", f"sample_{data.get('index', i)}")
            print(f"Saved evaluation for {uid}")

    print("Evaluation completed")
    return processed_data


def _save_single_evaluation(data: Dict[str, Any], results_folder: str):
    """Incrementally save evaluation result for a single sample."""
    unique_id = data.get("unique_id", f"sample_{data['index']}")
    result_file = os.path.join(results_folder, f"{unique_id}.json")

    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = {k: v for k, v in data.items() if k != "evaluation"}

    if "evaluation" in data:
        existing_data["evaluation"] = data["evaluation"]

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)


def update_results_files_with_evaluation(processed_data: List[Dict[str, Any]], results_folder: str):
    """Update evaluation results into result files."""
    for data in processed_data:
        unique_id = data.get("unique_id", f"sample_{data['index']}")
        result_file = os.path.join(results_folder, f"{unique_id}.json")

        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        else:
            print(f"Warning: Result file not found: {result_file}")
            continue

        if "evaluation" in data:
            existing_data["evaluation"] = data["evaluation"]

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)


def save_results(processed_data: List[Dict[str, Any]], timestamp_dir: str):
    """Save summary results."""
    evaluation_summary_dir = os.path.join(timestamp_dir, "evaluation_summary")
    os.makedirs(evaluation_summary_dir, exist_ok=True)

    print(f"Saving summary to {evaluation_summary_dir}")

    summary = generate_summary(processed_data)
    summary_path = os.path.join(evaluation_summary_dir, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Summary saved successfully")


def generate_summary(processed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics."""
    if not processed_data:
        return {"error": "No data to summarize"}

    all_rewards = []
    reward_keys = set()

    for data in processed_data:
        eval_data = data.get("evaluation", {})
        if "details" in eval_data:
            reward_details = eval_data["details"]
            reward_keys.update(reward_details.keys())
            all_rewards.append(reward_details)
    summary = {
        "total_samples": len(processed_data),
        "timestamp": datetime.now().isoformat(),
        "metrics": {}
    }

    if all_rewards:
        for key in sorted(reward_keys):
            values = [r.get(key, 0.0) for r in all_rewards if key in r]
            if values:
                summary["metrics"][key] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }

        total_rewards = [data.get("evaluation", {}).get("total_reward", 0.0) for data in processed_data]
        summary["total_reward"] = {
            "mean": sum(total_rewards) / len(total_rewards),
            "min": min(total_rewards),
            "max": max(total_rewards),
            "count": len(total_rewards)
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluation Pipeline for NC Physics")
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--example_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./evaluation_results')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--use_mock', action='store_true', default=False)
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--base_url', type=str, default=None)
    parser.add_argument('--max_workers', type=int, default=4)
    parser.add_argument('--max_tokens', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--inference_folder', type=str, default=None)
    parser.add_argument('--reward_types', type=str, nargs='+', default=['writing'],
                        choices=['graph_quality', 'writing', 'writing_llm'])
    parser.add_argument('--parallel_inference', action='store_true', default=False)
    parser.add_argument('--run_mode', type=str, default='2step',
                        choices=['step1', 'step2', '2step', "eval"])
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--copy_inference', action='store_true', default=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(timestamp_dir, exist_ok=True)
    inference_results_dir = os.path.join(timestamp_dir, "inference_results")
    os.makedirs(inference_results_dir, exist_ok=True)
    
    processed_data = load_and_process_data(args.data_file, args.example_dir, args.inference_folder)
    if args.max_samples:
        processed_data = processed_data[:args.max_samples]

    if args.inference_folder:
        inference_src = args.inference_folder
        sub_dir = os.path.join(inference_src, "inference_results")
        if os.path.exists(sub_dir):
            inference_src = sub_dir
        if args.copy_inference:
            copy_inference_results(inference_src, inference_results_dir)
        processed_data = load_inference_results(inference_src, processed_data)

    llm_config = {
        'model': args.model,
        'api_key': args.api_key,
        'base_url': args.base_url,
        'max_tokens': args.max_tokens,
        'temperature': args.temperature
    }
    
    if not args.run_mode == "eval":
        if args.parallel_inference:
            processed_data = run_inference_parallel(
                processed_data,
                use_mock=args.use_mock,
                llm_config=llm_config,
                results_folder=inference_results_dir,
                reward_types=args.reward_types,
                max_workers=args.max_workers,
                run_mode=args.run_mode,
                inference_folder=args.inference_folder
            )
        else:
            from functools import partial
            processed_data = run_inference_parallel(
                processed_data,
                use_mock=args.use_mock,
                llm_config=llm_config,
                results_folder=inference_results_dir,
                reward_types=args.reward_types,
                max_workers=1,
                run_mode=args.run_mode,
                inference_folder=args.inference_folder
            )

    if args.eval:
        processed_data = run_evaluation(
            processed_data,
            max_workers=args.max_workers,
            results_folder=inference_results_dir,
            reward_types=args.reward_types
        )
    else:
        print("Skip evaluation (--eval not set)")

    save_results(processed_data, timestamp_dir)
    print("Evaluation pipeline completed successfully!")


if __name__ == '__main__':
    main()
