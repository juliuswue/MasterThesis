import numpy as np
import argparse
import sys
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
    
def animate(simulation_state_data):    
    # Set up the plot
    fig, axs = plt.subplot_mosaic([["Pong"]],figsize=(5,5))
    
    # --- Setup ---
    # pong
    axs["Pong"].set_xlim([0, 1])
    axs["Pong"].set_ylim([0, 1])
    axs["Pong"].set_xticks([])
    axs["Pong"].set_yticks([])
    paddle, = axs["Pong"].plot([], [], 'black', lw=5)
    ball, = axs["Pong"].plot([], [], 'ro', markersize=5)
    paddle_height = 0.25
    paddle_y = simulation_state_data["paddle_y"]
    ball_x = simulation_state_data["ball_x"]
    ball_y = simulation_state_data["ball_y"]
    state_text =  axs["Pong"].text(0.4, 0.9, '', transform= axs["Pong"].transAxes)
    
    fig.tight_layout()
    
    def update(frame):
        update_pong(frame)
    
    def update_pong(frame):
        paddle.set_data([0.07, 0.07], [paddle_y[frame] - paddle_height / 2, paddle_y[frame] + paddle_height / 2])
        ball.set_data([ball_x[frame]], [ball_y[frame]])
        state_text.set_text(f'{simulation_state_data["trial"][frame]} - {simulation_state_data["pong_state"][frame]}')
        
    ani = FuncAnimation(fig=fig, func=update, frames=len(simulation_state_data["pong_state"]), interval=50, repeat=False)
    plt.show()

def main():     
    parser = argparse.ArgumentParser(description='Visualize saved simulation data')
    parser.add_argument('data_dir', help='Directory containing simulation data files')
    
    args = parser.parse_args()
    
    # if not os.path.exists(args.data_dir):
    #     print(f"Error: Directory {args.data_dir} does not exist")
    #     sys.exit(1)
    
    # # Find files
    # game_state_files = glob.glob(os.path.join(args.data_dir, "*simulation_states.csv"))
    
    # if game_state_files:
    #     game_state_path = game_state_files[2]
    #     print(f"showing: {game_state_path}")
    #     simulation_state_data = pd.read_csv(game_state_path)
    # else:
    #     raise FileNotFoundError("No game_states_*.csv file found in the specified folder.")
    simulation_state_data = pd.read_csv(args.data_dir)
    animate(simulation_state_data)


if __name__ == "__main__":
    main()