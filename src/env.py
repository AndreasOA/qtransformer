import metaworld
import random
import numpy as np
import torch



class MetaworldEnvironment:
    def __init__(self, env_name='door-open-v2', render=False):
        # Load MetaWorld Env
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ml1 = metaworld.ML1(env_name)
        self.env = self.ml1.train_classes[env_name]()
        self.render = render
        if render:
            self.env.render_mode = "human"
        task = random.choice(self.ml1.train_tasks)
        self.env.set_task(task)
        # Reset environment to its initial state
        self.reset()

    def reset(self):
        return torch.tensor(self.env.reset()[0], device=self.device, dtype=torch.float32)


    def forward(self, action):
        if torch.is_tensor(action):
            action = action.cpu().numpy()
        next_state, reward, done, truncated, _ = self.env.step(action)
        if self.render:
            self.env.render()
        return torch.tensor(reward, device=self.device, dtype=torch.float32), \
               torch.tensor(next_state, device=self.device, dtype=torch.float32), \
               torch.tensor(done, device=self.device, dtype=torch.bool), \
               torch.tensor(truncated, device=self.device, dtype=torch.bool)


if __name__ == "__main__":
    env_name = 'pick-place-v2'  
    render = True  

    # Instantiate the MetaWorldEnv class
    meta_env = MetaworldEnvironment(env_name, render=render)

    # Reset the environment
    obs = meta_env.reset()
    print("Initial Observation:", obs)

    # Perform random actions
    for _ in range(10000):  # Perform 10 random actions
        action = meta_env.env.action_space.sample()  # Sample a random action
        if random.random() > 0.5:  # Randomly convert action to tensor for testing
            action = torch.tensor(action)
        #print(action)
        reward, next_state, done, truncated = meta_env.forward(action)
        print(reward)

        if truncated or done:
            meta_env.reset()