import numpy as np

class PongSimulator:
    def __init__(self, dt_sim, random_seed, width=105, height=100):
        self.rng = np.random.default_rng(random_seed)
        
        self.width = width
        self.height = height
        self.dt_sim = dt_sim

        # paddle properties
        self.paddle_width = 5
        self.paddle_height = 25
        self.max_paddle_speed = self.height * dt_sim * 0.75
        self.paddle_y = self.height // 2  # center of the paddle
        self.paddle_vel = 0
        self.paddle_acc = self.height * dt_sim * 0.125#1.5  # acceleration per step

        self.ball_radius = 2.5
        
        self.reset()
        
        # constants to calculate relative positions
        min_paddle_y = self.paddle_height // 2
        max_paddle_y = self.height - self.paddle_height // 2
        min_ball_y = self.ball_radius
        max_ball_y = self.height - self.ball_radius
        self.max_neg_dist = min_ball_y - max_paddle_y
        self.max_pos_dist = max_ball_y - min_paddle_y

    def reset(self):
        self.ball_x = self.width - self.ball_radius - 0.5
        self.ball_y = self.height / 2
        self.ball_speed_x = - self.height * self.dt_sim * 0.5
        self.ball_speed_y = self.rng.uniform(low=-np.abs(self.ball_speed_x), high=np.abs(self.ball_speed_x))
        
        self.paddle_y = self.height / 2
        self.paddle_vel = 0

    def simulate(self, action):
        game_state = 'running'
        
        # update paddle
        if action == 'up':
            self.paddle_vel += self.paddle_acc
        elif action == 'down':
            self.paddle_vel -= self.paddle_acc
        self.paddle_vel = np.clip(self.paddle_vel, -self.max_paddle_speed, self.max_paddle_speed)
        self.paddle_y += self.paddle_vel
        # clamp paddle position and reset velocity if hitting boundary
        half_h = self.paddle_height // 2
        if self.paddle_y < half_h:
            self.paddle_y = half_h
            self.paddle_vel = 0
        elif self.paddle_y > self.height - half_h:
            self.paddle_y = self.height - half_h
            self.paddle_vel = 0

        # update ball
        self.ball_x += self.ball_speed_x
        self.ball_y += self.ball_speed_y

        # collision top / bottom
        if self.ball_y - self.ball_radius < 0:
            self.ball_y = self.ball_radius
            self.ball_speed_y *= -1
        if self.ball_y + self.ball_radius > self.height:
            self.ball_y = self.height - self.ball_radius
            self.ball_speed_y *= -1

        # collision right
        if self.ball_x + self.ball_radius >= self.width:
            self.ball_speed_x *= -1
            self.ball_speed_y = self.rng.uniform(low=-np.abs(self.ball_speed_x), high=np.abs(self.ball_speed_x))

        # collision paddle / left
        if self.ball_x - self.ball_radius <= self.paddle_width:
            if (self.paddle_y - half_h) <= self.ball_y <= (self.paddle_y + half_h):
                self.ball_speed_x *= -1
                self.ball_x = self.paddle_width + self.ball_radius  # small offset to avoid sticking
                game_state = 'hit'
            else:
                self.ball_speed_x *= -1
                self.ball_x = self.paddle_width + self.ball_radius
                game_state = 'miss'

        return game_state

    def get_simulation(self):
        rel_y_pos = (self.ball_y - self.paddle_y - self.max_neg_dist) / (self.max_pos_dist - self.max_neg_dist)
        stim_id = min(7, max(0, int(rel_y_pos * 8)))
        rel_x_pos = (self.ball_x - (self.paddle_width + self.ball_radius)) / (self.width - (self.paddle_width + self.ball_radius))
        
        return {
            'paddle_y': self.paddle_y / self.height,
            'ball_x': self.ball_x / self.width,
            'ball_y': self.ball_y / self.height,
            'rel_ball_x': rel_x_pos,
            'rel_ball_y': rel_y_pos,
            'stim_id': stim_id,
        }