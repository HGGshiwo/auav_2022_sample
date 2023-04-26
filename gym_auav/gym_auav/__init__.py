from gymnasium.envs.registration import register

register(
    id="gym_auav/TrialWorld-v0",
    entry_point="gym_auav.envs:TrialWorldEnv",
)