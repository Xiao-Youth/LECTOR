import re
import os
import datasets
import json
from verl.utils.hdfs_io import copy, makedirs
import argparse
import random


def get_introduction_writing_prompt_template() -> str:
    """Get introduction writing prompt template."""
    return r"""
You are given a Graphviz DOT file that encodes the complete logical reasoning structure behind a scientific research idea.
This DOT graph represents a single-rooted reasoning tree derived from the main body of a scientific article, capturing the end-to-end logical structure of the entire paper.
The graph is intended to serve as the structural and logical basis for writing the Introduction section.
Each node corresponds to a specific piece of information (original sentence, referenced opinion, or implicit knowledge),
and each pair of edges represents a reasoning step following Peirce's three modes of reasoning: deduction, abduction, and induction.

In addition to the DOT graph, you are also given a references list.
The references list contains bibliographic entries that may be cited in the Introduction, each associated with a unique index.

Your task is to use the reasoning graph as the sole source of scientific content and the provided references list as the sole source of citations,
and write a complete Introduction section for a scientific paper.

You must fully preserve, use, and cover all concepts, relations, and reasoning steps contained in the DOT graph, translating them into coherent academic prose, and insert citations at appropriate positions where prior work,
background knowledge, or related studies are involved.

All citations MUST strictly follow this format: [idx], where idx is the index of the corresponding reference
in the given references list. Do not invent, modify, or omit reference indices.

---

Swales' CARS Model Requirement:

The Introduction must explicitly follow Swales' CARS (Create A Research Space) model, which consists of three rhetorical moves:

1. Move 1 — Establishing the territory:
   Describe the broad research area and its importance. Summarize the established background knowledge relevant to the DOT graph.

2. Move 2 — Establishing the niche:
   Identify gaps, unresolved issues, limitations, or unanswered questions implied by the reasoning graph.

3. Move 3 — Occupying the niche:
   Present the central research idea, method, or proposal (corresponding to the root node of the graph), showing how it logically follows from the reasoning chain.

Your Introduction must integrate all nodes, reasoning relations, and required citations into a clear, natural,
and academically polished narrative.

---

Writing Requirements:

- Use clear, natural, academic English appropriate for a top-tier scientific journal.
- Ensure the Introduction fully reflects the logical structure encoded in the Graphviz DOT file.
- Ensure all citations come exclusively from the provided references list and use the required [idx] format.
- Use Markdown code formatting for all math symbols and equations.
- Do not omit any important information represented anywhere in the graph.
- DO NOT include commentary, explanations of the DOT file, or any text outside the required structure.
- DO NOT add references that are not explicitly provided.

---

Output Format (MANDATORY):

Your output must include only the Introduction content.
Do not include any titles, headings, sections, notes, or explanations—just the text of the Introduction itself.

---

Final Instruction:

Now, given the following Graphviz DOT code and the corresponding references list, write the required Introduction.

GRAPHVIZ DOT:
{graph}

REFERENCES:
{references}
"""



def build_introduction_writing_prompt(graph: str, references: str) -> str:
    """Build introduction writing prompt."""
    template = get_introduction_writing_prompt_template()
    return template.format(graph=graph, references=references)

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



def build_paper_content(example):
    """
    Build paper content: title + all sections except Introduction.
    
    """
    content_parts = []

    if 'title' in example and example['title']:
        content_parts.append(f"Title: {example['title']}")

    for i, section in enumerate(example.get('sections', [])):
        if i == 0:
            continue

        section_name = section.get('section', f'Section {i}')
        section_content = section.get('content', '')

        if section_content.strip():
            content_parts.append(f"{section_name}: {section_content}")

    return '\n\n'.join(content_parts)


def load_example(example_dir):
    """Load example input/output."""
    example_data = {"input": None, "output": None}
    if example_dir is not None:
        example_input_file = os.path.join(example_dir, "input_data.json")
        example_output_file = os.path.join(example_dir, "final_clean_graph.dot")

        if os.path.exists(example_input_file):
            with open(example_input_file, 'r', encoding='utf-8') as f:
                example_data["input"] = json.load(f)
        if os.path.exists(example_output_file):
            with open(example_output_file, 'r', encoding='utf-8') as f:
                example_data["output"] = f.read()
    return example_data


def build_paper2graph_prompt(example_intro: str = None, example_graph: str = None, paper_content: str = None) -> str:
    """Build paper2graph prompt."""
    has_example = (example_intro is not None and example_intro.strip() and
                   example_graph is not None and example_graph.strip())
    use_toy_example = False
    if not has_example:
        use_toy_example = True
        example_graph = """
digraph G {\n  n1 [label=\"Backward laser tuning decreases the effective detuning toward the low-detuning boundary in Kerr microresonators.\"];\n  n2 [label=\"If a procedure moves a system into a parameter region where a phenomenon exists, it provides deterministic access to that phenomenon.\"];\n  n3 [label=\"Backward laser tuning provides deterministic access to breathing dissipative solitons.\"];\n  n4 [label=\"Periodic oscillations of intracavity power with RF sidebands are observed during backward tuning in Si₃N₄ microresonators.\"];\n  n5 [label=\"Such oscillations and sidebands are characteristic signatures of breathing dissipative solitons.\"];\n  n6 [label=\"The Si₃N₄ microresonator hosts breathing dissipative solitons under backward tuning.\"];\n  n7 [label=\"Similar oscillations and sidebands are observed during backward tuning in MgF₂ microresonators.\"];\n  n8 [label=\"The MgF₂ microresonator hosts breathing dissipative solitons under backward tuning.\"];\n  n9 [label=\"Breathing dissipative solitons occur universally in Kerr microresonators of different platforms.\"];\n  n10 [label=\"Numerical simulations based on the Lugiato-Lefever equation predict a linear increase of breathing frequency with effective detuning.\"];\n  n11 [label=\"Experiments measure a linear increase of breathing frequency with effective detuning.\"];\n  n12 [label=\"The agreement between simulations and experiments validates the Lugiato-Lefever equation for breathing soliton dynamics.\"];\n  n13 [label=\"Backward laser tuning deterministically accesses the universal breathing soliton phenomenon in Kerr microresonators.\"];\n  n14 [label=\"Backward laser tuning is a robust, universally applicable method to generate and study breathing dissipative solitons in optical microresonators, confirming the Lugiato-Lefever model.\"];\n\n  n2 -> n3 [label=\"deduction-rule\"];\n  n1 -> n3 [label=\"deduction-case\"];\n\n  n4 -> n6 [label=\"abduction-phenomenon\"];\n  n5 -> n6 [label=\"abduction-knowledge\"];\n\n  n7 -> n8 [label=\"abduction-phenomenon\"];\n  n5 -> n8 [label=\"abduction-knowledge\"];\n\n  n6 -> n9 [label=\"induction-case\"];\n  n8 -> n9 [label=\"induction-common\"];\n\n  n3 -> n13 [label=\"induction-case\"];\n  n9 -> n13 [label=\"induction-common\"];\n\n  n11 -> n12 [label=\"induction-case\"];\n  n10 -> n12 [label=\"induction-common\"];\n\n  n12 -> n14 [label=\"induction-case\"];\n  n13 -> n14 [label=\"induction-common\"];\n}
"""

    paper2graph_prompt = """
Charles S. Peirce, a member of the National Academy of Sciences of the United States, pointed out that all valid reasoning is either deductive, inductive, or hypothetic; or else it combines two or more of these characters. 

Now, I provide the main body of a scientific article (excluding the Introduction section). Please extract its core scientific research proposal or idea, and use the above three types of reasoning to explicitly reconstruct the reasoning process leading to that idea.

Please output the complete logical reasoning chain in Graphviz DOT syntax as a single graph. Your output must consist of the complete Graphviz DOT graph only. Do NOT include any explanations, comments, natural language descriptions, or any text outside the DOT code block.

Please strictly abide by the following requirements:
Overall goal: Extract the logical structure behind the scientific research from the original text. Its structure should reflect the end-to-end logical organization of the entire paper, so that the resulting reasoning graph can serve as a structural basis for writing the Introduction section. The graph describes how the paper's core research idea is motivated, justified, and formed from the main body content. Specifically, you will build a single-rooted reasoning tree, and the following is the specific definition.

1. Each node should be written as a complete sentence (Transcription) that explicitly reflects a step in the reasoning process.

The information expressed in the Transcription may originate from one of the following situations:
- An original sentence from the paper
- An original viewpoint or claim inferred from the paper
- An opinion or statement derived from referenced work
- Implicit information or background knowledge used for reasoning

You do NOT need to explicitly label which situation a node belongs to. However, each node must correspond to exactly one reasoning unit or atomic piece of information. Do not combine multiple independent reasoning units into a single node. If multiple pieces of information are needed, create separate nodes and connect them through reasoning edges.

2. The edge type can only be one of the following 6 types: deduction-rule, deduction-case, abduction-phenomenon, abduction-knowledge, induction-case, induction-common

3. CRITICAL CONSTRAINT - Edge Pairing Requirements: Every reasoning conclusion must be reached by exactly two edges of specific paired types pointing to the same target node. The valid pairs are:
   - For deductive reasoning: One "deduction-rule" edge and one "deduction-case" edge must both point to the same target node
   - For abductive reasoning: One "abduction-phenomenon" edge and one "abduction-knowledge" edge must both point to the same target node
   - For inductive reasoning: One "induction-case" edge and one "induction-common" edge must both point to the same target node

   This means if you have a conclusion node reached by deductive reasoning, there must be exactly one incoming "deduction-rule" edge from one source node and exactly one incoming "deduction-case" edge from another source node, both targeting the same conclusion node.

4. Other constraints:
a. If there is multi-hop reasoning in the reasoning chain (or a single reasoning/induction/deduction is not enough to explain clearly, such compound reasoning is common in scientific literature), please introduce intermediate nodes (as implicit information) to break down the logical path into multiple clear reasoning steps. The intermediate nodes must also be written as complete sentences (Transcription). The more detailed the reasoning and the more nodes there are, the higher the score.

b. Domain consensus can be used as implicit information, but it cannot be written directly in the form of a conclusion, and must be written as callable background knowledge.

c. A single node may serve simultaneously as the conclusion of one reasoning step and as the argument (premise) of a subsequent reasoning step. In such cases, use a single Transcription for the node and connect it to different reasoning edges according to its roles.

d. The reasoning tree must have exactly one root node with input edges only. This root node represents the determination of the core scientific research idea or method, summarizing the overall research logic of the entire paper in a form that can be naturally articulated and motivated in the Introduction section.

e. You should aim to construct a connected, multi-step Reasoning Tree that gradually leads to this root node. Avoid linking most nodes directly to the root node; instead, intermediate conclusions should be reused as premises for subsequent reasoning steps, forming deeper and more coherent reasoning paths. All nodes must be part of this logical backbone, with no abandoned or isolated nodes.

f. Graph size constraint: The total number of nodes in the graph must NOT exceed 50. If the reasoning process would naturally exceed this limit, you must prioritize the most essential reasoning steps, merge highly similar or redundant intermediate nodes when logically possible, and preserve the main multi-hop reasoning backbone that leads to the root node.
"""

    if has_example:
        paper2graph_prompt += f"""
I will give you:
(1) an example paper main body content (excluding the Introduction),
(2) a correct logical reasoning graph extracted from it,
and then
(3) a new paper main body content (excluding the Introduction).

PAPER CONTENT (Example):
{example_intro}

EXPECTED OUTPUT (Example):
{example_graph}

PAPER CONTENT:
{paper_content}
"""
    elif use_toy_example:
        paper2graph_prompt += f"""
Here is a simple example graph. You can refer to its structure, but note that the graph you generate may be more complex.
EXAMPLE GRAPH:
{example_graph}

PAPER CONTENT:
{paper_content}
"""

    else:
        paper2graph_prompt += f"""

PAPER CONTENT:
{paper_content}
"""



    return paper2graph_prompt

def make_map_fn(example_dir=None):

    example_data = load_example(example_dir)

    has_valid_example = (example_data["input"] is not None and
                        example_data["output"] is not None and
                        example_data["input"].get("introduction", {}).get("content"))

    example_intro = None
    example_graph = None
    if has_valid_example:
        example_intro = example_data["input"]["introduction"]["content"]
        example_graph = example_data["output"]

    def process_fn(example, idx):
        paper_content = build_paper_content(example)

        introduction_content = ""
        if example.get('sections') and len(example['sections']) > 0:
            introduction_content = example['sections'][0].get('content', '')

        references = example['sections'][0]['references']
        reference_list = build_references_list(references)

        paper2graph_prompt = build_paper2graph_prompt(example_intro, example_graph, paper_content)

        data = {
            "data_source": "NC_physics",
            "prompt": [{
                "role": "user",
                "content": '',
            }],
            "ability": "paper2graph_writing",
            "reward_model": {
                "style": "rule",
                "ground_truth": introduction_content
            },
            "extra_info": {
                'index': idx,
                'subfield': example.get('subfield'),
                "unique_id": example.get('unique_id'),
            },
            "env_kwargs": {
                "paper2graph_prompt": [{
                "role": "user",
                "content": paper2graph_prompt,
            }],
                "introduction_writing_prompt": [{
                    "role": "user",
                    "content": get_introduction_writing_prompt_template(),
                }],
                "paper_content": paper_content,
                "introduction": introduction_content,
                "core_idea":  example.get('core_idea'),
                "entities": example.get('entities'),
                "references": references,
                "reference_list": reference_list
            }
        }
        return data
    return process_fn

def load_jsonl(file_path):
    """Load jsonl file and return as list."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/')
    parser.add_argument('--example_dir', default='', help='Directory containing example input/output')
    parser.add_argument('--name', default='LECTOR', help='Prefix for saved parquet files')
    parser.add_argument('--train_size', type=int, default=10000, help='Number of samples for training split')
    parser.add_argument('--val_size', type=int, default=100, help='Number of samples for validation split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()

    data_file = os.path.join(args.local_dir, "NC_Physics_trainval.jsonl")
    
    data = load_jsonl(data_file)

    dataset_all = datasets.Dataset.from_list(data)
    
    random.seed(args.seed)
    
    shuffled_indices = list(range(len(dataset_all)))
    random.shuffle(shuffled_indices)
    
    train_indices = shuffled_indices[:args.train_size]
    val_indices = shuffled_indices[args.train_size:args.train_size + args.val_size]
    
    splits = {
        'train': dataset_all.select(train_indices),
        'val': dataset_all.select(val_indices),
    }
    
    for split_name, dataset in splits.items():
        dataset = dataset.map(function=make_map_fn(example_dir=args.example_dir), with_indices=True)
        save_path = os.path.join(args.local_dir, f'{args.name}_{split_name}.parquet')
        dataset.to_parquet(save_path)
        print(f"Saved {split_name} dataset to {save_path}")
