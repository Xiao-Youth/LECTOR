from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory
from typing import List, Dict, Any

class TwoStepEnvironmentManager(EnvironmentManagerBase):
    """
    Fixed two-step task environment:
    - Step 1: generate prompt from input
    - Step 2: generate prompt from step 1 output (without step 1 context)
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs=None):
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.memory.reset(batch_size=len(obs))
        self.step_counter = [0] * len(obs)

        text_obs = [self.build_step1_prompt(o) for o in obs]

        observations = {
            'text': text_obs,
            'image': None,
            'anchor': obs.copy()
        }
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': next_obs, 'action': actions})

        text_obs = []
        for i, o in enumerate(next_obs):
            if self.step_counter[i] == 0:
                prompt = self.build_step2_prompt(o)
            else:
                prompt = f"Final observation: {o}"
            text_obs.append(prompt)
            self.step_counter[i] += 1

        next_observations = {
            'text': text_obs,
            'image': None,
            'anchor': next_obs.copy()
        }

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        return next_observations, rewards, dones, infos

    def build_step1_prompt(self, obs):
        return f"Step 1: Process input data: {obs}"

    def build_step2_prompt(self, step1_output):
        return f"Step 2: Process result of first step: {step1_output}"
