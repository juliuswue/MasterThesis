import numpy as np

class PongSimulator:
    def __init__(self, dt_sim, random_seed, width=100, height=100):
        np.random.seed(random_seed)
        self.width = width
        self.height = height
        
        self.dt_sim = dt_sim

        # Paddle properties
        self.paddle_width = 5
        self.paddle_height = 25
        self.paddle_speed = self.height * self.dt_sim
        self.paddle_y = self.height // 2  # center of the paddle

        # Ball properties
        self.ball_radius = 2.5
        self.reset()
        
        # Get constants to calculate the relative position
        min_paddle_y = self.paddle_height // 2
        max_paddle_y = self.height - self.paddle_height // 2
        min_ball_y = self.ball_radius
        max_ball_y = self.height - self.ball_radius
        
        self.max_neg_dist = min_ball_y - max_paddle_y
        self.max_pos_dist = max_ball_y - min_paddle_y
        

    def reset(self):
        self.ball_x = self.width - self.ball_radius * 2
        self.ball_y = np.random.uniform(low= 2*self.ball_radius, high= self.height - 2 * 2 * self.ball_radius)
        self.ball_speed_x = - 0.5 * self.width * self.dt_sim
        self.ball_speed_y = np.random.uniform(low=self.ball_speed_x / 3., high=self.ball_speed_x) * np.random.choice([-1, 1])

        self.paddle_y = self.height / 2  # reset paddle to middle (center-based)

    def simulate(self, action):
        game_state = 'running'

        # Move paddle
        if action == 'up':
            self.paddle_y += self.paddle_speed
        elif action == 'down':
            self.paddle_y -= self.paddle_speed
        else:
            self.paddle_y = self.paddle_y

        # Clamp paddle position (center-based)
        half_h = self.paddle_height // 2
        self.paddle_y = max(half_h, min(self.height - half_h, self.paddle_y))

        # Move ball
        self.ball_x += self.ball_speed_x
        self.ball_y += self.ball_speed_y

        # Ball collision with top or bottom
        if self.ball_y - self.ball_radius < 0 or self.ball_y + self.ball_radius > self.height:
            self.ball_speed_y *= -1

        # Ball collision with right wall
        if self.ball_x + self.ball_radius >= self.width:
            self.ball_speed_x *= -1
            self.ball_speed_y = np.random.uniform(low=self.ball_speed_x / 3., high=self.ball_speed_x) * np.random.choice([-1, 1])

        # Ball collision with paddle (left side)
        if self.ball_x - self.ball_radius <= self.paddle_width:
            if (self.paddle_y - half_h) <= self.ball_y <= (self.paddle_y + half_h):
                self.ball_speed_x *= -1
                self.ball_x = self.paddle_width + self.ball_radius  # avoid sticking
                self.ball_speed_y = np.random.uniform(low=self.ball_speed_x / 3., high=self.ball_speed_x) * np.random.choice([-1, 1])
                game_state = 'hit'
            else:
                game_state = 'miss'

        return game_state

    def get_simulation(self):
        rel_y_pos = self.ball_y - self.paddle_y
        stim_id = min(7, max(0, int((rel_y_pos - self.max_neg_dist) / (self.max_pos_dist - self.max_neg_dist) * 8)))
        
        return {
            'paddle_y': self.paddle_y / self.height,
            'ball_x': self.ball_x / self.width,
            'ball_y': self.ball_y / self.height,
            'stim_id': stim_id
        }
