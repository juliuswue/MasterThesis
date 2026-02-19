import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from datetime import datetime

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import json

import time
import os
from multiprocessing import Pool
from PongSimulator import PongSimulator

import numpy as np

N_NETWORKS = 80
N_RUNS = 200

FOLDER = f'/Users/juliuswuerzler/Documents/Uni/Master/MasterThesis/results/{datetime.now().strftime("%m_%d_%H_%M")}_IS_CONTROL'

os.makedirs(FOLDER, exist_ok=True)
results = np.zeros((N_NETWORKS, N_RUNS))
for i in range(N_NETWORKS):
    np.random.seed(i)
    dt_sim = 0.1 #! CHANGED 
    simulator = PongSimulator(dt_sim, i)
    simulator.reset()
    
    n_trials = 0
    game_state = 'running'
    
    simulation_state_fname = os.path.join(FOLDER, f"network_{i}_simulation_states.csv")
    with open(simulation_state_fname, 'w') as f:
        f.write("trial,ball_x,ball_y,paddle_y,pong_state\n")
    
        while n_trials < N_RUNS:
            if game_state == 'running':
                pong_state = simulator.get_simulation()
                if np.random.rand() < 0.5:
                    game_state = simulator.simulate('up')
                else:
                    game_state = simulator.simulate('down')
                    
            elif game_state == 'hit':
                results[i][n_trials] = 1
                
                pong_state = simulator.get_simulation()
                game_state = 'running'
                n_trials += 1
                
            elif game_state == 'miss':
                results[i][n_trials] = 0
                simulator.reset()
                pong_state = simulator.get_simulation()
                game_state = 'running'
                n_trials += 1
            else:
                raise "unknown game_state {game_state}!"
        
            #----- save data -----
            # if RECORD:
            f.write(f"{n_trials},{pong_state['ball_x']},{pong_state['ball_y']},{pong_state['paddle_y']},{game_state}\n")    

    # store results
    fname = os.path.join(FOLDER, f"network_{i}_results.csv")
    np.savetxt(fname, results[i], delimiter=",")

save_data = {
    "results": np.mean(results[:, -50:], axis=1).tolist() ,
    "results_mean": np.mean(results[:, -50:]),
}

# Save file
save_path = os.path.join(FOLDER, "summary.json")
with open(save_path, "w") as f:
    json.dump(save_data, f, indent=4)