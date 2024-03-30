from pettingzoo.mpe import simple_adversary_v3
env = simple_adversary_v3.parallel_env()

for i in range(10):
    
    observations, infos = env.reset(seed=42)
    print(observations)
    while env.agents:
      
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()