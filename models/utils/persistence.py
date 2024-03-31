from datetime import datetime
import os
import torch



def save_actor_critic(maddpg, save_dir):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  
    model_name = f'{current_time}.pth'
    

    actor_save_dir = f'./././parameters/weights/{save_dir}/{current_time}/actor'
    critic_save_dir = f'./././parameters/weights/{save_dir}/{current_time}/critic'
    
    os.makedirs(actor_save_dir, exist_ok=True)
    os.makedirs(critic_save_dir, exist_ok=True)

    for i, ddpgs in enumerate(maddpg.agents):
    
        torch.save(ddpgs.actor.state_dict(), os.path.join(actor_save_dir,f'actor_{i}_{model_name}'))
        torch.save(ddpgs.critic.state_dict(), os.path.join(critic_save_dir,f'critic_{i}_{model_name}'))
     
        print(f"Model parameters saved as '{model_name}'")