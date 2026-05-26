# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Dict, Any
import numpy as np
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.environments.graph_reward import compute_graph_quality_reward_batch
from agent_system.environments.writing_reward import compute_writing_reward_batch
from agent_system.environments.writing_reward_llm import compute_writing_reward_llm_batch
from agent_system.memory import SimpleMemory

class PaperEnvironmentManager(EnvironmentManagerBase):
    """
    Minimal two-step environment for paper-style tasks.

    Step 1:
        prompt = paper2graph_prompt
    Step 2:
        prompt = introduction_writing_prompt + step1_output
        (NO step-1 context)

    Episode ends after step 2.
    """

    def __init__(self, envs=None, projection_f=None, config=None):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs=None):
        batch_size = self.config.data.train_batch_size * self.config.env.rollout.n
        self.step_counter = [0] * batch_size

        self.env_kwargs = kwargs

        paper2graph_prompt = [
            item.get("paper2graph_prompt", None) for item in self.env_kwargs
        ]

        self.step1_outputs = [None] * batch_size
        self.memory.reset(batch_size=batch_size)

        text_obs = [prompt[0]["content"] for prompt in paper2graph_prompt]

        observations = {
            "text": text_obs,
            "image": None,
            "anchor": text_obs.copy(),
        }

        infos = [{"won": False, "is_action_valid": True} for _ in range(batch_size)]
        return observations, infos

    def step(self, text_actions: List[str]):
        batch_size = len(text_actions)

        next_text_obs = [""] * batch_size
        rewards = np.zeros(batch_size, dtype=np.float32)
        dones = np.zeros(batch_size, dtype=bool)
        infos: List[Dict[str, Any]] = [{} for _ in range(batch_size)]

        env_kwargs = self.env_kwargs
        writing_prompt = [
            item.get("introduction_writing_prompt", [{"role": "user", "content": ""}])
            for item in env_kwargs
        ]
        introduction = [item.get("introduction", None) for item in env_kwargs]
        core_idea = [item.get("core_idea", None) for item in env_kwargs]
        entities = [item.get("entities", None) for item in env_kwargs]
        reference_list = [item.get("reference_list", None) for item in env_kwargs]
        references = [item.get("references", None) for item in env_kwargs]

        step1_indices = [i for i in range(batch_size) if self.step_counter[i] == 0]
        step2_indices = [i for i in range(batch_size) if self.step_counter[i] == 1]

        if step1_indices:
            step1_outputs = [text_actions[i] for i in step1_indices]
            self.step1_outputs = step1_outputs

            coverage_list, accuracy_list = compute_graph_quality_reward_batch(
                step1_outputs,
                [core_idea[i] for i in step1_indices],
                [entities[i] for i in step1_indices],
            )

            for idx, cov, acc in zip(step1_indices, coverage_list, accuracy_list):
                self.step1_outputs[idx] = text_actions[idx]

                rewards[idx] = cov + acc

                prompt = self._build_step2_prompt(
                    text_actions[idx],
                    writing_prompt[idx][0]["content"],
                    reference_list[idx]
                )
                next_text_obs[idx] = prompt
                self.step_counter[idx] = 1

                infos[idx] = {
                    "won": False,
                    "is_action_valid": True,
                    "coverage": float(cov),
                    "accuracy": float(acc),
                }

        if step2_indices:
            step2_outputs = [text_actions[i] for i in step2_indices]

            writing_rewards, writing_reward_details = compute_writing_reward_batch(
                self.step1_outputs, step2_outputs, introduction, references
            )

            writing_reward_llm, writing_reward_llm_details = compute_writing_reward_llm_batch(
                self.step1_outputs, step2_outputs, introduction, references
            )
            writing_rewards = [
                llm + base for llm, base in zip(writing_reward_llm, writing_rewards)
            ]

            writing_reward_details = [
                {**llm_detail, **base_detail}
                for llm_detail, base_detail in zip(writing_reward_llm_details, writing_reward_details)
            ]
            for idx, r, r_details in zip(step2_indices, writing_rewards, writing_reward_details):
                rewards[idx] = r
                next_text_obs[idx] = "Episode finished."
                self.step_counter[idx] = 2
                dones[idx] = True

                infos[idx] = {
                    "won": True,
                    "is_action_valid": True,
                    "writing_reward": float(r),
                    **{k: float(v) for k, v in r_details.items()}  # Ensure all values are float
                }

        for i in range(batch_size):
            if self.step_counter[i] >= 2 and not dones[i]:
                next_text_obs[i] = "Episode finished."
                dones[i] = True
                infos[i] = {
                    "won": True,
                    "is_action_valid": True,
                }

        observations = {
            "text": next_text_obs,
            "image": None,
            "anchor": text_actions.copy(),
        }

        return observations, to_numpy(rewards), to_numpy(dones), infos

    def _build_step2_prompt(self, step1_output: str, introduction_template: str, references) -> str:
        return introduction_template.format(graph=step1_output, references=references)

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        """
        success = info['won'] at last active step.
        """
        success['coverage'].append(total_infos[batch_idx][0]['coverage'])
        success['accuracy'].append(total_infos[batch_idx][0]['accuracy'])
        all_keys = [
            "writing_reward",
            "lexical_bleu",
            "semantic_similarity",
            "contextual_relevance",
            "paper_coverage",
            "graph_coverage",
            "kp_faithfulness_graph",
            "kp_consistency_paper",
            "entailment_faithfulness",
            "entailment_consistency",
            "reference_recall",
            "llm_background_context",
            "llm_problem_clarity",
            "llm_motivation_significance",
            "llm_related_work_positioning",
            "llm_contribution_clarity",
            "llm_logical_structure",
            "llm_coherence_flow",
            "llm_consistency_with_original",
            "llm_coverage_key_points",
            "llm_references_usage_correctness",
            "llm_academic_writing_quality",
            "llm_generated_better_than_original"
        ]

        for key in all_keys:
            success[key].append(total_infos[batch_idx][1][key])
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item["active_masks"]:
                info = total_infos[batch_idx][i]
                success["success_rate"].append(float(info["won"]))
                return


def make_envs(config):
    if "paper" not in config.env.env_name.lower():
        raise ValueError(f"Unsupported environment: {config.env.env_name}")
    envs = PaperEnvironmentManager(envs=None, config=config)
    val_envs = PaperEnvironmentManager(envs=None, config=config)
    return envs, val_envs

