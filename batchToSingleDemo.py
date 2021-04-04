import numpy as np

import glob, os


def get_files_directories(folder:str):
    return sorted(glob.glob(os.path.join(folder, '*.npz')))

def united_all_files_and_save(file_directories:list):
    #Get first file in folder
    first_directory = file_directories[0]

    data = np.load(first_directory)

    #Matrices
    actions = data['actions']
    obs = data['obs']

    #Arrays
    #episode_returns = data['episode_returns'] #Not needed right now
    episode_starts = data['episode_starts']
    rewards = data['rewards']

    del data



    
    #Remove first element from list
    file_directories = file_directories[1:] 

    for file in file_directories:
        data = np.load(file)

        #Matrices
        actions = np.vstack((actions, data['actions']))
        obs = np.vstack((obs, data['obs']))

        #Arrays
        #episode_returns = data['episode_returns'] #Not needed right now
        episode_starts = np.append(episode_starts, data['episode_starts']) 
        rewards = np.append(rewards, data['rewards'])

        del data
    
    episode_returns = np.array([np.sum(rewards)])

<<<<<<< HEAD
    np.savez("./gameDemoComplete.npz", actions=actions, obs=obs, episode_starts=episode_starts, rewards=rewards, episode_returns=episode_returns)

def main():
    united_all_files_and_save(get_files_directories("./Demo_Batches"))


main()
=======
    np.savez("./gameDemoComplete.npz", actions=actions, obs=obs, episode_starts=episode_starts, rewards=rewards, episode_returns=episode_returns)
>>>>>>> 644ace173affc82f75f90e3aaa0f71f46c8f65ce
