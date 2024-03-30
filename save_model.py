from datetime import datetime
import os
import torch



def save_actor_critic(maddpg):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  
    model_name = f'{current_time}.pth'

    save_dir = 'saved_models/maddpg/'
    os.makedirs(save_dir, exist_ok=True)

    for ddpgs in maddpg.agents:
    
        torch.save(ddpgs.actor.state_dict(), os.path.join(save_dir,'actor'+model_name))
        torch.save(ddpgs.critic.state_dict(), os.path.join(save_dir,'critic'+model_name))
     
        print(f"Model parameters saved as '{model_name}'")