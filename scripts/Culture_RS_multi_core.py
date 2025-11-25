import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from datetime import datetime

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import time
import os
from multiprocessing import Pool

from brian2 import *

# ----------------------- models -----------------------
# --- Brian Equations ---
eqs_neurons = '''
r = u * Hz : Hz
dr_slow/dt = 1 / Tau_slow * (-r_slow + r) : Hz
du/dt = 1 / Tau * (-u + u_syn + u_r + r_ext) : 1
du_r/dt = - Theta_ur * (u_r - U_r0) + Sigma_ur * xi: 1
r_ext : 1
u_syn : 1
sum_w : 1
X : metre (constant)
Y : metre (constant)
U_r0 : 1 (constant)
Theta_ur : Hz (constant)
Sigma_ur : Hz**0.5 (constant)
Tau : second (constant)
Tau_slow : second (constant)
'''

eqs_syn = '''
delta_w = r_pre/Hz * r_post/Hz * (r_post - theta) : Hz
dw/dt = Lr * (
    delta_w * (int(delta_w < 0*Hz) * int(w > 0.01) * A_ltd + int(delta_w > 0*Hz) * int(w < 0.7))
    - C * int(sum_w_pre > W_sum_max) * int(w > 0.01) * (sum_w_pre - W_sum_max) * Hz
) : 1 (clock-driven)
theta = (r_slow_post)**2 / R_0 : Hz (constant over dt)
u_syn_post = w * r_pre / Hz : 1 (summed)
sum_w_pre = w : 1 (summed)
Lr : 1
W_sum_max : 1
A_ltd : 1
R_0 : Hz
C : 1
'''

# ----------------------- Parameters -----------------------
# --- Culture  ---
WIDTH = 3.85*mm
HEIGHT = 2.1*mm
NEURON_DENSITY = 100 / (WIDTH * HEIGHT)
DEGREE = 10

ELECTRODE_COLUMNS = 220
ELECTRODE_ROWS = 120

mm_per_electrode = WIDTH / ELECTRODE_COLUMNS

SENSORY_POSITIONS_X = np.arange(25,220, 25) * mm_per_electrode
SENSORY_POSITIONS_Y = np.array([20, 40] * 4) * mm_per_electrode
SENSORY_RADIUS = 200*um

MOTOR_LENGTH = 30 * mm_per_electrode
MOTOR_POSITIONS_X = np.array([20,50, 140, 170]) * mm_per_electrode
MOTOR_POSITIONS_Y = 80 * mm_per_electrode

n_neurons = int(WIDTH * HEIGHT * NEURON_DENSITY)

# --- Experiemnt ---
N_RUNS = 300
T_INIT = 30
N_PARAM_SETS = 1_000
N_CPU_CORES = 80
N_NETWORKS_PER_PARAM_SET = 20

pbounds = {
    'p_Var': (1e-4, 2e-1),
    'p_Sigma': (0, 2e-1),
    'p_C': (1e-1, 1e2),
    'p_W_sum': (0.2, 0.9),
    'p_A_ltd': (0.7, 5),
    'p_tau_slow': (200, 400),
    'p_eff': (0.4, 0.8),
    'p_readout_acc': (1e-4, 1)
    }

# ----------------------- Funcitons -----------------------
# --- plot functions ---
def plot_culture_areas(ax):
    # chip area
    rect = patches.Rectangle((0, 0), float(WIDTH / mm), float(HEIGHT / mm), fill=False, lw=2, edgecolor='black')
    ax.add_patch(rect)

    # motor areas
    for i, x in enumerate(MOTOR_POSITIONS_X):
        c = 'dodgerblue'
        if i % 2 == 0:
            c ='orchid'
        rect = patches.Rectangle((float(x / mm), float(MOTOR_POSITIONS_Y / mm)), float(MOTOR_LENGTH / mm), float(MOTOR_LENGTH / mm), fill=False, lw=2, edgecolor=c)
        ax.add_patch(rect)
        
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
    plt.text(0, 2.3, network_info)
    
    stimulation_neurons_info = ''
    for idx in stimulation_ids.values():
        # todo fixme
        if idx.size == 1:
            idx = idx[0]
            plt.scatter(neurons[idx].X / mmetre, neurons[idx].Y / mmetre, s=20, color='coral', alpha=0.5)
            # outgoing synapses for this neuron
            mask = synapses.i[:] == idx
            targets = synapses.j[:][mask]
            # Draw lines to targets
            for tgt in targets:
                if isin(tgt, motor_ids_U):
                    color = 'orchid'
                elif isin(tgt, motor_ids_D):
                    color = 'dodgerblue'
                else:
                    color = 'lightgray'
                    
                plt.plot([neurons.X[idx]/mmetre, neurons.X[tgt]/mmetre],
                        [neurons.Y[idx]/mmetre, neurons.Y[tgt]/mmetre],
                        color=color, alpha=0.6, linewidth=1.5)
            
            if isin(idx, stimulation_ids["U"]):
                stimulation_neurons_info += f"U stimulation neuron has {sum(isin(targets, motor_ids_U))} direct U and {sum(isin(targets, motor_ids_D))} direct D connections\n"
            if isin(idx, stimulation_ids["D"]):
                stimulation_neurons_info += f"D stimulation neuron has {sum(isin(targets, motor_ids_U))} direct U and {sum(isin(targets, motor_ids_D))} direct D connections\n"
                
        else:
            for x in idx:
                plt.scatter(neurons[x].X / mmetre, neurons[x].Y / mmetre, s=20, color='coral', alpha=0.5)
                # outgoing synapses for this neuron
                mask = synapses.i[:] == x
                targets = synapses.j[:][mask]
                # Draw lines to targets
                for tgt in targets:
                    if isin(tgt, motor_ids_U):
                        color = 'orchid'
                    elif isin(tgt, motor_ids_D):
                        color = 'dodgerblue'
                    else:
                        color = 'lightgray'
                        
                    plt.plot([neurons.X[x]/mmetre, neurons.X[tgt]/mmetre],
                            [neurons.Y[x]/mmetre, neurons.Y[tgt]/mmetre],
                            color=color, alpha=0.6, linewidth=1.5)
                
                if isin(x, stimulation_ids["U"]):
                    stimulation_neurons_info += f"U stimulation neuron has {sum(isin(targets, motor_ids_U))} direct U and {sum(isin(targets, motor_ids_D))} direct D connections\n"
                if isin(x, stimulation_ids["D"]):
                    stimulation_neurons_info += f"D stimulation neuron has {sum(isin(targets, motor_ids_U))} direct U and {sum(isin(targets, motor_ids_D))} direct D connections\n"
    
    plt.text(0, -0.7, stimulation_neurons_info)

    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.axis('equal')
    
    return fig

def plot_run(network):
    fig, axs = plt.subplots(2,5, figsize=(12, 6))

    t_mask_all = network["M"].t > T_INIT*second
    t_mask_start = (network["M"].t > T_INIT*second) & (network["M"].t < T_INIT*second + 60*second)
    t_mask_end = network["M"].t > max(network["M"].t) - 60*second

    # plot rates 
    for i, t_mask in enumerate([t_mask_start, t_mask_end]):
        axs[0][i].plot(network["M"].t[t_mask], network["M"].r[:, t_mask].T, alpha=0.3, linewidth=1.5, color='lightgray')
        axs[0][i].plot(network["M"].t[t_mask], network["M"].r[network["stimulation_ids"]["U"][0]][t_mask].T, alpha=0.5, linewidth=1.5, color='orchid')
        axs[0][i].plot(network["M"].t[t_mask], network["M"].r[network["stimulation_ids"]["D"][0]][t_mask].T, alpha=0.5, linewidth=1.5, color='dodgerblue')

        axs[1][i].plot(network["M"].t[t_mask], mean(network["M"].r[:, t_mask][network['motor_ids_U']], axis=0), alpha=0.4, linewidth=1.5, color='purple')
        axs[1][i].plot(network["M"].t[t_mask], mean(network["M"].r[:, t_mask][network['motor_ids_D']], axis=0).T, alpha=0.4, linewidth=1.5, color='darkblue')
        
        axs[0][i].set_ylabel('r [Hz]')
        axs[1][i].set_ylabel('r [Hz]')

    # plot weights
    for i, t_mask in enumerate([t_mask_start, t_mask_end, t_mask_all]):
        axs[0][i+2].plot(network["S"].t[t_mask], network["S"].w[(isin(network["synapses"].j, network["motor_ids_U"]) & (isin(network["synapses"].i, network["stimulation_ids"]["U"][0])))][:, t_mask].T, alpha=0.3, linewidth=1.5, color='darkgreen')
        axs[0][i+2].plot(network["S"].t[t_mask], network["S"].w[(isin(network["synapses"].j, network["motor_ids_D"]) & (isin(network["synapses"].i, network["stimulation_ids"]["U"][0])))][:, t_mask].T, alpha=0.3, linewidth=1.5, color='darkred')
        
        axs[1][i+2].plot(network["S"].t[t_mask], network["S"].w[(isin(network["synapses"].j, network["motor_ids_D"]) & (isin(network["synapses"].i, network["stimulation_ids"]["D"][0])))][:, t_mask].T, alpha=0.3, linewidth=1.5, color='darkgreen')
        axs[1][i+2].plot(network["S"].t[t_mask], network["S"].w[(isin(network["synapses"].j, network["motor_ids_U"]) & (isin(network["synapses"].i, network["stimulation_ids"]["D"][0])))][:, t_mask].T, alpha=0.3, linewidth=1.5, color='darkred')

        axs[0][i+2].set_ylabel('w')
        axs[1][i+2].set_ylabel('w')


    for ax in axs.flatten():
        ax.set_xlabel('time [s]')
        
    for ax,title in zip(axs[0], ['first min', 'last min', 'first min', 'last min', 'all']):
        ax.set_title(title)
        
    fig.tight_layout()
    return fig

# --- stimulation functions ---
def gameplay_stimulation(network, random_dir):
    idx = network["stimulation_ids"][random_dir]
    # todo: FIXME
    if idx.size == -1:
        network["neurons"][idx:idx+1].r_ext = 20 * network["args"]["eff"]
    else:
        for x in idx:
            network["neurons"][x:x+1].r_ext = 20 * network["args"]["eff"]
    
    network["net"].run(0.2*second)
    
    # todo: FIXME
    if idx.size == -1:
        network["neurons"][idx:idx+1].r_ext = 0
    else:
        for x in idx:
            network["neurons"][x:x+1].r_ext = 0
    
    r_U = mean([network["neurons"].r[x] for x in network["motor_ids_U"]])
    r_D = mean([network["neurons"].r[x] for x in network["motor_ids_D"]])
    
    network["net"].run(2*second)
    
    return r_U, r_D

def random_stimulation(network):
    for idx in network["stimulation_ids"].values():
        # todo: FIXME
        if idx.size == -1:
            network["neurons"][idx:idx+1].r_ext = 5
        else:
            for x in idx:
                network["neurons"][x:x+1].r_ext = 5
            
    network["net"].run(4*second)
    
    for idx in  network["stimulation_ids"].values():
        # todo: FIXME
        if idx.size == -1:
            network["neurons"][idx:idx+1].r_ext = 0
        else:
            for x in idx:
                network["neurons"][x:x+1].r_ext = 0
            
    network["net"].run(4*second)
    network["net"].run(2*second)
    
def sync_stimulation(network):
    for idx in  network["stimulation_ids"].values():
        # todo: FIXME
        if idx.size == -1:
            network["neurons"][idx:idx+1].r_ext = 100 * network["args"]["eff"]
        else:
            for x in idx:
                network["neurons"][x:x+1].r_ext = 100 * network["args"]["eff"]

    network["net"].run(0.1*second)
    
    for idx in  network["stimulation_ids"].values():
        # todo: FIXME
        if idx.size == -1:
            network["neurons"][idx:idx+1].r_ext = 0
        else:
            for x in idx:
                network["neurons"][x:x+1].r_ext = 0
    
    network["net"].run(2*second)

# --- simulation functions ---
def create_network(args):
    seed(args["random_seed"])
    
    neurons = NeuronGroup(n_neurons, eqs_neurons, method='euler')
    synapses = Synapses(neurons, neurons, eqs_syn, method='euler')
    
    neurons.X = 'rand() * WIDTH'
    neurons.Y = 'rand() * HEIGHT'
    
    synapses.connect(condition='i!=j', p=DEGREE/n_neurons)
    synapses.w = f'0.05*rand()'
    for j in range(n_neurons):
        ids = isin(synapses.j, j)
        synapses.w[:][ids] = synapses.w[:][ids] / sum(synapses.w[:][ids]) * 0.5
    
    
    # set neuron parameters
    neurons.U_r0 = args["U_r0"]
    neurons.u_r = args["U_r0"]
    neurons.Theta_ur = pow(args["Sigma_ur"],2) / (2 * args["Var_ur"]) *Hz
    neurons.Sigma_ur = args["Sigma_ur"] *Hz**0.5
    neurons.Tau_slow = args["Tau_slow"]*ms
    neurons.Tau = args["Tau"]*ms
    neurons.u = args["R_0"]
    
    # set synapse paramters
    synapses.Lr = args["Lr"]
    synapses.W_sum_max = args["W_sum_max"]
    synapses.C = args["C"]
    synapses.A_ltd = args["A_ltd"]
    synapses.R_0 = args["R_0"]*Hz
    synapses.theta = args["R_0"]*Hz
    
    # set motor and sensory ids
    motor_ids = []
    for x in MOTOR_POSITIONS_X:
        mask = ((neurons.X >= x) &
                (neurons.X <  x + MOTOR_LENGTH) &
                (neurons.Y >= MOTOR_POSITIONS_Y) &
                (neurons.Y <  MOTOR_POSITIONS_Y + MOTOR_LENGTH)) 
        ids = where(mask)[0]       
        motor_ids.append(ids)
    motor_ids_U = np.hstack((motor_ids[0],  motor_ids[2]))
    motor_ids_D = np.hstack((motor_ids[1],  motor_ids[3]))
        
    sensory_ids = []
    for x,y in zip(SENSORY_POSITIONS_X, SENSORY_POSITIONS_Y):
        ids = argmin(sqrt(pow(neurons.X - x, 2) + pow(neurons.Y - y, 2)))
        if ids not in sensory_ids:
            sensory_ids.append(ids)
    sensory_ids = array(sensory_ids)
    
    # find stimulation sites (2 neurons with the most connections to motor area neurons)
    n_motor_connections = zeros_like(sensory_ids)
    for i,idx in enumerate(sensory_ids):
        # outgoing synapses for this neuron
        mask = synapses.i[:] == idx
        targets = synapses.j[:][mask]
        n_motor_connections[i] = sum(isin(targets, motor_ids_U)) + sum(isin(targets, motor_ids_D))
    stimulation_ids = {"U": sensory_ids[argsort(n_motor_connections)[::-1][[0]]],
                       "D": sensory_ids[argsort(n_motor_connections)[::-1][[1]]]}
    
    dt_record = 50*ms
    M = StateMonitor(neurons, ['r'], record=True, dt=dt_record)
    S = StateMonitor(synapses, ['w'], record=True, dt=dt_record)
    
    net = Network(neurons, synapses, M, S)
    
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
    
    results = np.zeros(N_RUNS)
    for i in range(N_RUNS):
        random_dir = "U" if i % 2 == 0 else "D"
        r_U, r_D = gameplay_stimulation(network, random_dir)
        if random_dir == "U":
            if rand() < 1 / (1+exp(- (r_U - r_D) / network["args"]["readout_acc"])):
                sync_stimulation(network)
                results[i] = 1
            else:
                random_stimulation(network)
                results[i] = 0
        if random_dir == "D":
            if rand() < 1 / (1+exp(- (r_D - r_U) / network["args"]["readout_acc"])):
                sync_stimulation(network)
                results[i] = 1
            else:
                random_stimulation(network,)
                results[i] = 0
                
    # save run
    fig = plot_run(network)
    fname = os.path.join(args["outdir"], f"network_{args["random_seed"]}.png")
    fig.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    fig = plot_network(network)
    fname = os.path.join(args["outdir"], f"network_{args["random_seed"]}_layout.png")
    fig.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close(fig)
    del fig
        
    return mean(results[-results.size//10:])

def run_one_paramter_set(trial):
    args_list = []
    
    p_Var = trial.suggest_float("Var", 1e-4, 2e-1)
    p_Sigma = trial.suggest_float("Sigma", 0, 2e-1)
    p_tau_slow = trial.suggest_float("tau_slow", 200, 400)
    p_eff = trial.suggest_float("eff", 0.3, 0.8)
    p_readout_acc = trial.suggest_float("readout_acc", 1e-3, 2e-1)
    p_W_sum = trial.suggest_float("W_sum", 0.2, 0.95)
    p_A_ltd = trial.suggest_float("A_ltd", 0.7, 4)
    p_C = trial.suggest_float("C", 0, 20)
    
    outdir = f"results/{datetime.datetime.now().strftime("%m_%d_%H_%M")}_trial_{trial.number}"
    os.makedirs(outdir, exist_ok=True)
    
    
    for random_seed in range(N_NETWORKS_PER_PARAM_SET):
        args_list.append(
            {
                "random_seed": random_seed,
                "outdir": outdir,
                # neuron parameters
                "U_r0": 1,
                "Var_ur": p_Var,
                "Sigma_ur": p_Sigma,
                "Tau_slow": p_tau_slow,
                "Tau": 100,
                # synapse params
                "Lr": 1e-3, 
                "W_sum_max": p_W_sum, 
                "A_ltd": p_A_ltd, 
                "R_0": 2,
                "C": p_C,
                # set-up params
                "eff": p_eff,
                "readout_acc": p_readout_acc,
            })
        
    results = []
    for step, args in enumerate(args_list):
        results.append(run_single_network(args))
        
        intermediate_result = mean(array(results))
        trial.report(intermediate_result, step)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
    return mean(results)
  
def run_optimization(_):
    study = optuna.create_study(
        study_name="journal_storage_multiprocess",
        storage=JournalStorage(JournalFileBackend(file_path="./journal.log")),
        load_if_exists=True, # Useful for multi-process or multi-node optimization.
        direction='maximize',
        pruner=optuna.pruners.PercentilePruner(
            25.0, 
            n_startup_trials=N_CPU_CORES, 
            n_warmup_steps=4, 
            interval_steps=2
        )
    )
    study.optimize(run_one_paramter_set, n_trials=N_PARAM_SETS // N_CPU_CORES)
    
def main():
    print(f"Starting at {datetime.datetime.now().strftime("%H:%M:%S")}")
    start_time = time.time()
    
    with Pool(processes=N_CPU_CORES) as pool:
        pool.map(run_optimization, range(N_CPU_CORES))
    
    print(f"Ended at {datetime.datetime.now().strftime("%H:%M:%S")}. Took {(time.time()-start_time):.0f}s to run")
    
if __name__ == '__main__':
    main()