import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from datetime import datetime

import json

import time
import os
from multiprocessing import Pool
from PongSimulator import PongSimulator

from brian2 import *

# ----------------------- Models -----------------------
# --- Brian Equations ---
eqs_neurons = '''
r = clip(u, 0 * Hz, 100 * Hz) : Hz
dr_slow/dt = 1 / Tau_slow * (-r_slow + r**2) : Hz*Hz
du/dt = 1 / Tau * (-u + u_syn + u_r + r_ext) : Hz
du_r/dt = - Theta_ur * (u_r - U_r0) + (Sigma_ur * xi)*Hz : Hz
r_ext : Hz
u_syn : Hz
sum_w : 1
X : metre (constant)
Y : metre (constant)
U_r0 : Hz (constant)
Theta_ur : Hz (constant)
Sigma_ur : Hz**0.5 (constant)
Tau : second (constant)
Tau_slow : second (constant)
'''

eqs_syn = '''
dw = r_pre/Hz * r_post/Hz * (r_post - theta) : Hz (constant over dt)
ddelta_w/dt = 1 / Tau_W * (dw - delta_w) : Hz (clock-driven)
dw/dt = Lr * clip(
    delta_w * (int(delta_w < 0*Hz) * int(w > 0.01) + int(delta_w > 0*Hz) * int(w < 0.5)), 
    -30*17*Hz, 30*17*Hz)
    - 1 / C * int(sum_w_pre > W_sum_max * DEG) * int(w > 0.01) * (sum_w_pre - W_sum_max * DEG)
: 1 (clock-driven)
theta = r_slow_post / R_0 : Hz (constant over dt)
u_syn_post = w * r_pre : Hz (summed)
sum_w_pre = w : 1 (summed)
Lr : 1 (constant)
W_sum_max : 1 (constant)
R_0 : Hz (constant)
C : second (constant)
Tau_W : second (constant)
DEG : 1 (constant)
'''

# ----------------------- Parameters -----------------------
# --- Culture  ---
WIDTH = 3.85*mm
HEIGHT = 2.1*mm
NEURON_DENSITY = 400 / (WIDTH * HEIGHT)
DEGREE = 10

ELECTRODE_COLUMNS = 220
ELECTRODE_ROWS = 120
N_PER_ELECTRODE = 3

mm_per_electrode = WIDTH / ELECTRODE_COLUMNS

SENSORY_POSITIONS_X = np.arange(25,220, 25) * mm_per_electrode
SENSORY_POSITIONS_Y = np.array([20, 40] * 4) * mm_per_electrode
SENSORY_RADIUS = 200*um

MOTOR_LENGTH = 30 * mm_per_electrode
MOTOR_POSITIONS_X = np.array([20,50, 140, 170]) * mm_per_electrode
MOTOR_POSITIONS_Y = 80 * mm_per_electrode

n_neurons = int(WIDTH * HEIGHT * NEURON_DENSITY)

# --- Experiemnt ---
N_RUNS = 200
T_INIT = 300
N_CPU_CORES = 8
N_NETWORKS_PER_PARAM_SET = 40

RECORD = False
LOAD_WEIGHTS = False
WEIGHT_PATH = ''

EXPERIMENT = "NoFB" # ONLYRSTFB - NoFB - RST

# ----------------------- Functions -----------------------
# --- plot ---
def plot_culture_areas(ax):
    # chip area
    rect = patches.Rectangle((0, 0), float(WIDTH / mm), float(HEIGHT / mm), fill=False, lw=2, edgecolor='black')
    ax.add_patch(rect)

    # # motor areas
    # for i, x in enumerate(MOTOR_POSITIONS_X):
    #     c = 'dodgerblue'
    #     if i % 2 == 0:
    #         c ='orchid'
    #     rect = patches.Rectangle((float(x / mm), float(MOTOR_POSITIONS_Y / mm)), float(MOTOR_LENGTH / mm), float(MOTOR_LENGTH / mm), fill=False, lw=2, edgecolor=c)
    #     ax.add_patch(rect)
        
    # sensory area
    cmap = plt.get_cmap('autumn')  # returns RGBA values between 0 and 1
    colors = [cmap(i/8) for i in range(8)]  # 8 shades
    for x, y, c in zip (SENSORY_POSITIONS_X, SENSORY_POSITIONS_Y, colors):
        circ = patches.Circle((float(x / mm), float(y / mm)), radius=float(SENSORY_RADIUS / mm), alpha=0.2, fc=c)
        ax.add_patch(circ)
        
def plot_network(network):
    neurons = network["neurons"]
    synapses = network["synapses"]
    motor_ids_U = network["motor_ids_U"]
    motor_ids_D = network["motor_ids_D"]
    stimulation_ids = network["stimulation_ids"]
    
    fig, ax = plt.subplots(figsize=(6,6))

    plt.scatter(neurons.X / mmetre, neurons.Y / mmetre, s=10, color='black')
    plot_culture_areas(ax)

    network_info = f'total: {n_neurons} neurons and {len(synapses.i)} synapses \navg syn per neuron: {len(synapses.i) / n_neurons:.1f} synapses\nest. p: {len(synapses.i) / (n_neurons * n_neurons):.3f} synapses'
    plt.text(0, 2.3, network_info, fontsize='x-small')
    
    stimulation_neurons_info = ''
    for i, idx in enumerate(stimulation_ids):
        plt.scatter(neurons[idx].X / mmetre, neurons[idx].Y / mmetre, s=30, color='coral', alpha=0.5)
        
        # outgoing synapses for this neuron
        mask = synapses.i[:] == idx
        targets = synapses.j[:][mask]
        # draw lines to targets
        for tgt in targets:
            if isin(tgt, motor_ids_U):
                color = 'orchid'
            elif isin(tgt, motor_ids_D):
                color = 'dodgerblue'
            else:
                color = 'lightgray'
                
            plt.plot([neurons.X[idx]/mmetre, neurons.X[tgt]/mmetre],
                    [neurons.Y[idx]/mmetre, neurons.Y[tgt]/mmetre],
                    color=color, alpha=0.3, linewidth=1.5)
            
        stimulation_neurons_info += f"stim. neuron {i} (id:{idx}): {sum(isin(targets, motor_ids_U))} direct U & {sum(isin(targets, motor_ids_D))} direct D conn.\n"
        
    for i, idx in enumerate(motor_ids_U):
        plt.scatter(neurons[idx].X / mmetre, neurons[idx].Y / mmetre, s=30, color='orchid', alpha=0.5)
    for i, idx in enumerate(motor_ids_D):
        plt.scatter(neurons[idx].X / mmetre, neurons[idx].Y / mmetre, s=30, color='dodgerblue', alpha=0.5)
    
    
    plt.text(0, -3., stimulation_neurons_info, fontsize='x-small')

    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.axis('equal')
    
    return fig

# --- stimulation ----
def gameplay_stimulation(network, stim_id, ball_x, dt_stim):
    if EXPERIMENT == "RST":
        network["net"].run(dt_stim*second)
    else:
        for k in range(N_PER_ELECTRODE):
            stim_neuron_id = network["stimulation_ids"][N_PER_ELECTRODE*stim_id+k]
            network["neurons"][stim_neuron_id:stim_neuron_id+1].r_ext = (6 * ball_x + 4)*Hz
        network["net"].run(dt_stim*second)
        for k in range(N_PER_ELECTRODE):
            stim_neuron_id = network["stimulation_ids"][N_PER_ELECTRODE*stim_id+k]
            network["neurons"][stim_neuron_id:stim_neuron_id+1].r_ext = 0  * Hz
    
    r_U = mean([network["neurons"].r[x] for x in network["motor_ids_U"]])
    r_D = mean([network["neurons"].r[x] for x in network["motor_ids_D"]])

    return r_U, r_D

def random_stimulation(network):
    if EXPERIMENT != "RST" and EXPERIMENT !="NoFB" and EXPERIMENT != "ONLYRSTFB":
        for idx in network["stimulation_ids"]:
            network["neurons"][idx:idx+1].r_ext = 5*Hz
        network["net"].run(4*second)
        
        for idx in network["stimulation_ids"]:
            network["neurons"][idx:idx+1].r_ext = 0*Hz
        network["net"].run(4*second)
    
def sync_stimulation(network):
    if EXPERIMENT != "RST" and EXPERIMENT !="NoFB" and EXPERIMENT != "ONLYRSTFB":
        for idx in network["stimulation_ids"]:
            network["neurons"][idx:idx+1].r_ext = network["args"]["eff"]*Hz
        network["net"].run(0.1*second)
        
        for idx in  network["stimulation_ids"]:
            network["neurons"][idx:idx+1].r_ext = 0*Hz

# --- network ---
def create_network(args):
    seed(args["random_seed"])
    
    neurons = NeuronGroup(n_neurons, eqs_neurons, method='euler')
    synapses = Synapses(neurons, neurons, eqs_syn, method='euler')
    
    neurons.X = 'rand() * WIDTH'
    neurons.Y = 'rand() * HEIGHT'    
    
    # set neuron parameters
    neurons.U_r0 = args["U_r0"]*Hz
    neurons.u_r = args["U_r0"]*Hz
    neurons.Theta_ur = pow(args["Sigma_ur"],2) / (2 * args["Var_ur"]) *Hz
    neurons.Sigma_ur = args["Sigma_ur"] *Hz**0.5
    neurons.Tau_slow = args["Tau_slow"]*ms
    neurons.Tau = args["Tau"]*ms
    neurons.u = args["R_0"]*Hz
    
    # set motor and sensory ids
    sensory_ids = []
    for x,y in zip(SENSORY_POSITIONS_X, SENSORY_POSITIONS_Y):
        ids = argsort(sqrt(pow(neurons.X - x, 2) + pow(neurons.Y - y, 2)))
        for _ in range(N_PER_ELECTRODE):
            for idx in ids:
                if idx not in sensory_ids:
                    sensory_ids.append(idx)
                    break
    
    motor_ids = []
    n_per_area = round(n_neurons * 0.2 / 2)
    
    motor_ids_U = np.zeros(n_per_area, dtype=int32) - 1
    motor_ids_D = np.zeros(n_per_area, dtype=int32) - 1
    all_motor_ids = []
    
    for idx in range(n_per_area):
        # U
        _candidate = -1
        while(_candidate == -1 or _candidate in sensory_ids or _candidate in all_motor_ids):
            _candidate = randint(n_neurons)
        motor_ids_U[idx] = _candidate
        all_motor_ids.append(_candidate)
        
        #D
        _candidate = -1
        while(_candidate == -1 or _candidate in sensory_ids or _candidate in all_motor_ids):
            _candidate = randint(n_neurons)
        motor_ids_D[idx] = _candidate
        all_motor_ids.append(_candidate)
    
    # FIX INDEGREE
    neuron_ids = arange(n_neurons)
    for idx in range(n_neurons):
        pre_ids = choice(neuron_ids[neuron_ids!=idx], size=DEGREE, replace=False)
        synapses.connect(i=pre_ids, j=idx)
    
    stimulation_ids = array(sensory_ids)
            
    synapses.w[:] = np.clip(np.random.normal(0.05, 0.02, size=synapses.w[:].size), 0.01, 0.5)
        
    for pre in range(n_neurons):
        ids = isin(synapses.i, pre)
        synapses.DEG[:][ids] = sum(ids)
        
    # set synapse paramters
    synapses.Lr = args["Lr"]
    synapses.W_sum_max = args["W_sum_max"]
    synapses.C = args["C"]*ms
    synapses.R_0 = args["R_0"]*Hz
    synapses.theta = args["R_0"]*Hz
    synapses.Tau_W = args["Tau_W"]*ms
    
    
    dt_record = 0.1*second
    M = StateMonitor(neurons, ['r'], record=RECORD, dt=dt_record)
    S = StateMonitor(synapses, ['w'], record=RECORD, dt=dt_record)
    
    net = Network(neurons, synapses, M, S)
    
    if LOAD_WEIGHTS:
        for file in os.listdir(WEIGHT_PATH):
            file_name = os.path.join(WEIGHT_PATH, file)
                
            if f"_{args["random_seed"]}_weights.csv" in file_name:
                loaded_data = np.genfromtxt(file_name, delimiter=',', names=True, dtype=None, encoding='utf-8')
                
                synapses.w[:] = loaded_data['w']
                print("Weights sucessfully loaded!")
                break
    else:
        net.run(T_INIT*second)
    
    return {
        "net": net,
        "neurons": neurons,
        "synapses": synapses,
        "M": M,
        "S": S,
        "sensory_ids": sensory_ids,
        "motor_ids": motor_ids,
        "motor_ids_U": motor_ids_U,
        "motor_ids_D": motor_ids_D,
        "stimulation_ids": stimulation_ids,
        "args": args
    }

def run_single_network(args):
    network = create_network(args)
    # plot network 
    fig = plot_network(network)
    fname = os.path.join(args["outdir"], f"network_{args["random_seed"]}_layout.png")
    simulation_state_fname = os.path.join(args["outdir"], f"network_{args["random_seed"]}_simulation_states.csv")
    fig.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close(fig)
    del fig
    
    # store initial weights
    df = pd.DataFrame({
        "pre":  network["synapses"].i[:],
        "post": network["synapses"].j[:],
        "w":  network["synapses"].w[:]
    })
    df.to_csv(os.path.join(args["outdir"], f"network_{args['random_seed']}_initial_weights.csv"), index=False)
    
    # run simulation
    results = zeros(N_RUNS)
    dt_sim = 0.1
    simulator = PongSimulator(dt_sim, args["random_seed"])
    simulator.reset()
    pong_state = simulator.get_simulation()
    
    n_trials = 0
    game_state = 'running'
    with open(simulation_state_fname, 'w') as f:
        f.write("trial,ball_x,ball_y,paddle_y,rel_ball_x,rel_ball_y,stim_id,pong_state,network_time\n")
        f.write(f"{n_trials},{pong_state['ball_x']},{pong_state['ball_y']},{pong_state['paddle_y']},{pong_state['rel_ball_x']},{pong_state['rel_ball_y']},{pong_state['stim_id']},{game_state},{network['net'].t_}\n")
        
        while n_trials < N_RUNS:
            if game_state == 'running':
                r_U, r_D = gameplay_stimulation(network, pong_state['stim_id'], pong_state['rel_ball_x'], dt_sim)
                p_U = 1 / (1+exp(clip(-(r_U - r_D) / network["args"]["readout_acc"], -20, 20)))
                if rand() < p_U:
                    game_state = simulator.simulate('up')
                else:
                    game_state = simulator.simulate('down')
                pong_state = simulator.get_simulation()
                f.write(f"{n_trials},{pong_state['ball_x']},{pong_state['ball_y']},{pong_state['paddle_y']},{pong_state['rel_ball_x']},{pong_state['rel_ball_y']},{pong_state['stim_id']},{game_state},{network['net'].t_}\n")
                    
            if game_state == 'hit':
                sync_stimulation(network)
                results[n_trials] = 1
                game_state = 'running'
                n_trials += 1
                
            elif game_state == 'miss':
                random_stimulation(network)
                results[n_trials] = 0
                game_state = 'running'
                n_trials += 1
                
                if EXPERIMENT !="NoFB":
                    simulator.reset()
                    pong_state = simulator.get_simulation()
                    f.write(f"{n_trials},{pong_state['ball_x']},{pong_state['ball_y']},{pong_state['paddle_y']},{pong_state['rel_ball_x']},{pong_state['rel_ball_y']},{pong_state['stim_id']},{game_state},{network['net'].t_}\n")
        
    # store final weights
    df = pd.DataFrame({
        "pre":  network["synapses"].i[:],
        "post": network["synapses"].j[:],
        "w":  network["synapses"].w[:]
    })
    df.to_csv(os.path.join(args["outdir"], f"network_{args['random_seed']}_final_weights.csv"), index=False)
    
    # store rates + weights over time
    if RECORD:
        w_final = network["synapses"].w[:]  # numpy array
        i = network["synapses"].i[:]
        j = network["synapses"].j[:]

        df = pd.DataFrame({
            "pre": i,
            "post": j,
            "w": w_final
        })
        
        S = network["S"]
        np.save(os.path.join(args["outdir"],
                            f"network_{args['random_seed']}_weights_time.npy"),
                S.w)
        
        M = network["M"]
        np.save(os.path.join(args["outdir"],
                            f"network_{args['random_seed']}_rates_time.npy"),
                M.r)

        fname = os.path.join(args["outdir"],
                            f"network_{args['random_seed']}_weights.csv")
        df.to_csv(fname, index=False)
        
        # ---- save neuron IDs ----
        ids = {
            "motor_ids_U": [int(x) for x in network["motor_ids_U"]],
            "motor_ids_D": [int(x) for x in network["motor_ids_D"]],
            "sensory_ids": [int(x) for x in network["sensory_ids"]],
            "stimulation_ids": [int(x) for x in network["stimulation_ids"]],
        }

        fname = os.path.join(
            args["outdir"],
            f"network_{args['random_seed']}_neuron_ids.json"
        )

        with open(fname, "w") as f:
            json.dump(ids, f, indent=4)

    # store results
    fname = os.path.join(args["outdir"], f"network_{args['random_seed']}_results.csv")
    np.savetxt(fname, results, delimiter=",")
    
    return mean(results[-50:])
    
def run_one_paramter_set():
    args_list = []
    
    outdir = f"results/{datetime.datetime.now().strftime("%m_%d_%H_%M")}_trial_NT_OLDTP_Var0_05_30s_RO01_NoFB"
    os.makedirs(outdir, exist_ok=True)
        
    for random_seed in range(N_NETWORKS_PER_PARAM_SET):
        args_list.append(
            {
                "random_seed": random_seed,
                "outdir": outdir,
                # neuron parameters
                "U_r0": 1,
                "Var_ur": 0.05,
                "Sigma_ur": 0.0577,
                "Tau_slow": 1_000,
                "Tau": 10,
                # synapse params
                "Lr": 2e-4,
                "W_sum_max": 0.12,
                "R_0": 2,
                "C": 2_000,
                "Tau_W": 100,
                # set-up params
                "eff": 30,
                "readout_acc": 0.01,
            }
        )

    results = []
    
    with Pool(processes=N_CPU_CORES) as pool:
        results = pool.map(run_single_network, args_list)
        
    # Prepare data to save
    save_data = {
        "args": args_list[0],
        "results": results,
        "results_mean": mean(results),
        "Experiment": EXPERIMENT
    }

    # Save file
    save_path = os.path.join(outdir, "summary.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=4)
        
    return mean(results)


def main():
    print(f"Starting at {datetime.datetime.now().strftime("%H:%M:%S")}")
    start_time = time.time()
    
    run_one_paramter_set()
    
    print(f"Ended at {datetime.datetime.now().strftime("%H:%M:%S")}. Took {(time.time()-start_time):.0f}s to run")
    
if __name__ == '__main__':
    main()