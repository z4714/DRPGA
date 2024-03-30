from pettingzoo.mpe import simple_adversary_v3
import json
import numpy as np
import os
from ..src.maddpg_utils import SFSA_generator
from ..models.maddpg.sa_maddpg import MADDPG


directory = "././data/SFSA_0200"


max_cycle = 200
seed = -1

test_env = simple_adversary_v3.parallel_env(max_cycles=max_cycle)


test_agents = ['adversary_0', 'agent_0', 'agent_1']

dialogues_and_order = {
    "adversary_0(Thief)": {"dialogues": [[] for _ in range(4)]},
    "agent_0(Elf Wizard)": {"dialogues": [[] for _ in range(4)]},
    "agent_1(Human Warrior)": {"dialogues": [[] for _ in range(4)]},
    "order": [[] for _ in range(4)]
}



maddpg = MADDPG()


SFSA_generator.sfsa_generator(test_env, test_agents, seed, maddpg, max_cycle, directory, dialogues_and_order)


