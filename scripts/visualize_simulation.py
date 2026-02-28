import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import matplotlib.patches as patches

def animate(simulation_state_data):
    # Set up the plot
    fig, axs = plt.subplot_mosaic([["Pong"]], figsize=(5, 5))

    # --- Setup ---
    ax = axs["Pong"]
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1])
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Paddle (rectangle in data units) ---
    paddle_height = 0.4
    paddle_width = 0.05
    paddle_x = 0.0

    paddle = patches.Rectangle(
        (paddle_x, 0.5 - paddle_height / 2),
        paddle_width,
        paddle_height,
        facecolor="black"
    )
    ax.add_patch(paddle)

    ball_radius = 0.025
    ball = patches.Circle(
        (0.5, 0.5), 
        radius=ball_radius,
        facecolor="lightgray"
    )
    ax.add_patch(ball)

    paddle_y = simulation_state_data["paddle_y"].to_numpy()
    ball_x = simulation_state_data["ball_x"].to_numpy()
    ball_y = simulation_state_data["ball_y"].to_numpy()

    state_text = ax.text(0.4, 0.9, '', transform=ax.transAxes)

    fig.tight_layout()

    def update(frame):
        # update positions
        y0 = paddle_y[frame] - paddle_height / 2
        paddle.set_xy((paddle_x, y0))
        ball.center = (ball_x[frame], ball_y[frame])

        state = simulation_state_data["pong_state"][frame]
        if state == "hit":
            state_text.set_text(f'{simulation_state_data["trial"][frame]} - {state}')
            state_text.set_color("green")
        elif state == "miss":
            state_text.set_text(f'{simulation_state_data["trial"][frame]} - {state}')
            state_text.set_color("red")

        return paddle, ball, state_text

    ani = FuncAnimation(
        fig=fig,
        func=update,
        frames=len(simulation_state_data["pong_state"]),
        interval=50,
        repeat=False
    )

    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize saved simulation data')
    parser.add_argument('data_dir', help='Path to CSV file containing simulation data')
    args = parser.parse_args()

    simulation_state_data = pd.read_csv(args.data_dir)
    animate(simulation_state_data)

if __name__ == "__main__":
    main()
