from pettingzoo.mpe import simple_adversary_v3
import json
import numpy as np
import os
import tools

def sfsa_generator(test_env, test_agents, seed, model, max_cycle, directory, dialogues_and_order):


    test_state, info = test_env.reset() if seed == -1 else test_env.reset(seed=seed)

    record_dict = {}

    cur_state = test_state

    last_actions = {agent:-1 for agent  in test_agents}

    for i in range(max_cycle):

        test_actions = model.take_action(cur_state, explore=False)

        test_env_actions = {t_agent : np.argmax(t_action) for t_agent, t_action in zip(test_agents, test_actions)}

        if test_env_actions == last_actions and i != max_cycle-1:
            continue

        t_next_state,  t_reward, t_done, t_truncations, t_infos = test_env.step(test_env_actions)

        round_data = {
            f"state_{i}": cur_state,
            f"actions_{i}": test_env_actions,
            f"reward_{i}": t_reward,
            f"state_{i+1}": t_next_state,
        }

        record_dict[str(i)] = round_data

        cur_state = t_next_state

        last_actions = test_env_actions

        if t_reward == {}:
            print('done:',t_done)
            print('truncations:',t_truncations)
            break

    
    
    for round_key in record_dict.keys():
        record_dict[round_key].update(dialogues_and_order)

    record_dict_cov = tools.convert_ndarray(record_dict)

    norm_directory = os.path.normpath(directory)
    base_filename = os.path.basename(norm_directory)

    filename = tools.generate_filename(directory, base_filename)


    with open(filename, 'w') as json_file:
        json.dump(record_dict_cov, json_file, indent=2)


    return 


