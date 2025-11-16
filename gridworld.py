import os 
import pygame 
import numpy as np 
from matplotlib import colormaps
import datetime as dt 
from PIL import Image
import tkinter as tk
from tkinter import filedialog


colormap = colormaps["Blues"]  

SCREEN_CAPTURE = True   
SCREEN_SIZE = 800
SIDE_MARGIN = 200
ACTUAL_SCREEN_WIDTH = SCREEN_SIZE + SIDE_MARGIN
BACKGROUND_COLOR = (50, 50, 50)
MOTION_NOISE = 0.1  # Probability of action failure
SENSOR_NOISE = 0.0  # Probability of sensor failure (per feature)
SENSOR_LENGTH = 1  # "radius" of sensor (in cells)
U, D, L, R = "U", "D", "L", "R"

# Visibility flags
belief_visible = True  

# Initialize PyGame
pygame.init()
screen = pygame.display.set_mode((ACTUAL_SCREEN_WIDTH, SCREEN_SIZE))
clock = pygame.time.Clock()
pygame.display.set_caption("Gridworld")


class Environment:

    # Initialization ---------------------------------------------------------------------------------------------
    
    def __init__(self, dim=15, n_obstacles=20):
        # Configuration
        self.dim = dim  # Grid size
        self.cell_size = SCREEN_SIZE // dim
        self.n_obstacles = n_obstacles
        self.action_space = [U, D, L, R]
        # Initialization 
        self.timestep = 0
        self.initialize_grid()
        self.initialize_robot() 
        self.initialize_belief() 
        self.observe()
        if SCREEN_CAPTURE:
            self.frames = []
    
    def initialize_grid(self):
        self.grid = np.zeros((self.dim, self.dim), dtype=int)
        self.grid[0,:] = 1
        self.grid[-1,:] = 1
        self.grid[:,0] = 1
        self.grid[:,-1] = 1
        for _ in range(self.n_obstacles):
            y, x = np.random.randint(1, self.dim-3, size=2)
            h, w = np.random.randint(1, 3, size=2)
            self.grid[y:y+h,x:x+w] = 1
    
    def initialize_robot(self, yx=None):
        if yx is None:
            possible_robot_coords = list(zip(*np.where(self.grid==0)))
            idx = np.random.choice(range(len(possible_robot_coords)))
            y, x = possible_robot_coords[idx]
            self.robot_y, self.robot_x = y.item(), x.item()
        else:
            y, x = yx
            self.robot_y, self.robot_x = y, x
    
    def initialize_belief(self):
        self.belief = np.ones((self.dim, self.dim), dtype=float)
        self.belief[self.grid == 1] = 0
        self.belief /= self.belief.sum()  
    
    # Step: transition to the next state -------------------------------------------------------------------------
    
    def step(self, action=None):
        self.timestep += 1
        self.observe()
        # Belief updates according to Bayes filter: 
        self.motion_belief_update(action)
        self.sensor_belief_update()
        if SCREEN_CAPTURE:
            self.save_frame()

    # Motion -----------------------------------------------------------------------------------------------------
    
    def move(self, action):
        print(f"Action: {action}")
        new_y, new_x = self._compute_motion(action)
        if np.random.uniform(0, 1) > MOTION_NOISE:
            self.robot_y, self.robot_x = new_y, new_x 
        self.step(action)
    
    def _compute_motion(self, action, yx=None):
        """
        Compute new state from [current state, action] 
        to see what would happen in a perfect (non-noisy) environment 
        """
        assert action in self.action_space, "Invalid action."
        if yx is None:
            y, x = self.robot_y, self.robot_x
        else:
            y, x = yx
        match action.upper():
            case "U":
                new_y, new_x = y-1, x
            case "D": 
                new_y, new_x = y+1, x
            case "L":
                new_y, new_x = y, x-1 
            case "R":
                new_y, new_x = y, x+1 
        if (0 <= new_y < self.dim) and (0 <= new_x < self.dim) and (not self.grid[new_y, new_x]):
            return new_y, new_x
        return y, x
    
    # Observation ------------------------------------------------------------------------------------------------
    
    def observe(self):
        """ 
        Get observation from the grid for current position.
        Observation is defined as 8 neighboring cells to the robot 
        which are either 0 (free) or 1 (blocked)
        Each bit in this observation has a chance of SENSOR_NOISE of being flipped 
        """
        observation = self._compute_observation()
        # self.true_observation = observation.copy()
        for i, j in np.ndindex(observation.shape):
            if np.random.uniform(0, 1) < SENSOR_NOISE:
                observation[i, j] = 1-observation[i, j]
        self.observation = observation
    
    def _compute_observation(self, yx=None):
        """
        Compute observation for a particular state 
        to see what would happen in a perfect (non-noisy) environment
        """
        if yx is None:
            y, x = self.robot_y, self.robot_x
        else:
            y, x = yx
        r = SENSOR_LENGTH  # "radius"
        # Pad the grid: add r cells of padding on each side
        padded = np.pad(self.grid, pad_width=r, mode="constant", constant_values=-1)
        # Shift (y, x) to match padded coordinates
        yp, xp = y+r, x+r
        observation = padded[yp-r:yp+r+1, xp-r:xp+r+1]
        # observation = self.grid[y-SENSOR_LENGTH:y+SENSOR_LENGTH+1, x-SENSOR_LENGTH:x+SENSOR_LENGTH+1]
        return observation
    
    # Bayes filter ------------------------------------------------------------------------------------------------
    # Motion update -----------------------------------------------------------------------------------------------
    
    def motion_belief_update(self, action):
        """
        Update the robot's belief about its position after taking an action, 
        using the motion model and previous belief.
        action: str -> action taken by the robot ('U', 'D', 'L', 'R')
        """
        if action is None:
            return 
        # Create a new belief matrix to store the updated belief after action
        new_belief = np.zeros(self.belief.shape)
        # Iterate over all possible states in the grid (all (y, x) positions)
        for y, x in np.ndindex(self.grid.shape):
            # Compute the sum of probabilities over all previous states (prev_y, prev_x)
            # For each possible previous state, use the motion model to get transition probabilities
            for prev_y, prev_x in np.ndindex(self.grid.shape):
                # Transition probability from previous state (prev_y, prev_x) to current state (y, x)
                transition_prob = self.motion_model((prev_y, prev_x), action)
                # Multiply the transition probability by the belief at the previous state
                new_belief[y, x] += transition_prob[y, x] * self.belief[prev_y, prev_x]
        # Set belief in blocked cells to 0 (since the robot can't be there)
        new_belief[self.grid == 1] = 0
        # Normalize the belief so that it sums to 1.0 (important for valid probability distribution)
        total_belief = new_belief.sum()
        if total_belief > 0:
            new_belief /= total_belief
        else:
            # If belief sum is zero (e.g., all possible locations are blocked), reset to uniform belief
            new_belief = np.ones(new_belief.shape)/new_belief.size
        # Update the robot's belief with the newly computed belief
        self.belief = new_belief
    
    def motion_model(self, prev_state, action):
        """
        Given the robot's previous state and action, calculate the transition probabilities 
        to all possible new states, accounting for motion noise.
        prev_state: (y, x) -> the previous robot location.
        action: str -> one of 'U', 'D', 'L', 'R' for up, down, left, right.
        Returns: A 2D numpy array of probabilities P(x_t | u_t, x_(t-1)) for all possible states x_t.
        """
        # Get the intended next state given the action
        y, x = prev_state
        y_intended, x_intended = self._compute_motion(action, prev_state)
        # Initialize transition probability matrix with zeros
        p = np.zeros(self.grid.shape)
        # If the intended cell is not blocked
        if 0 <= y_intended < self.dim and 0 <= x_intended < self.dim and not self.grid[y_intended, x_intended]:
            # Probability of reaching the intended state (successful action)
            p[y_intended, x_intended] = 1 - MOTION_NOISE
        # If the intended cell is blocked or out of bounds
        else:
            # If the intended cell is blocked or out of bounds, robot stays in the same state
            p[y, x] = 1 - MOTION_NOISE
        # Handle the case where the robot stays in the same location (due to motion noise)
        p[y, x] += MOTION_NOISE
        return p

    # Sensor update -----------------------------------------------------------------------------------------------
    
    def sensor_belief_update(self):
        """
        aka observation update, measurement update, correction
        """
        belief = np.zeros(self.belief.shape)
        for y, x in np.ndindex(self.grid.shape):
            # state: (y, x)
            likelihood = self.sensor_model((y, x), self.observation)
            belief[y, x] = self.belief[y, x] * likelihood
        # Force the belief in blocked cells to 0:
        belief[self.grid == 1] = 0.0
        eta = belief.sum()
        if eta > 0:
            belief /= eta
            self.belief = belief
    
    def sensor_model(self, state, true_observation):
        """
        """
        sensor_accuracy = 1-SENSOR_NOISE
        y, x = state
        # What the robot *should* see in this state:
        expected_observation = self._compute_observation(state).reshape(-1)  
        prob = 1.0
        for true_feature, expected_feature in zip(true_observation.reshape(-1), expected_observation):
            if true_feature == expected_feature:
                # If true observation == expected observation, this state is likely! 
                prob *= sensor_accuracy
            else:
                # if true observation != expected observation, this state is unlikely! 
                # if SENSOR_ACCURACY==1, then the prob of this state is simply 0 
                prob *= 1-sensor_accuracy
        return prob


    # Drawing ----------------------------------------------------------------------------------------------------
    
    def draw_grid(self):
        d = self.cell_size
        for y, x in np.ndindex(self.grid.shape):
            c = 255*(1-self.grid[y, x]) 
            color = (c, c, c) 
            pygame.draw.rect(screen, color, pygame.Rect(x*d, y*d, d, d))
    
    def draw_gridlines(self, xloc=0, dim=None):
        if dim is None:
            dim = self.dim
        gridline_color = (30, 30, 30) 
        lw = 2
        d = self.cell_size
        for x in range(dim+1):
            pygame.draw.line(screen, gridline_color, (xloc+x*d, 0), (xloc+x*d, dim*d), lw)
        for y in range(dim+1): 
            pygame.draw.line(screen, gridline_color, (xloc+0, y*d), (xloc+dim*d, y*d), lw)
    
    def draw_robot(self):
        d = self.cell_size
        x, y = self.robot_x, self.robot_y
        robot_color = (255, 0, 0)
        pygame.draw.rect(screen, robot_color, pygame.Rect(x*d, y*d, d, d))
    
    def draw_belief(self):
        d = self.cell_size
        surface = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE), pygame.SRCALPHA)
        for y, x in np.ndindex(self.belief.shape):
            rgba = colormap(self.belief[y, x]**0.2)  # **0.2 scales the color intensity!
            r, g, b = tuple(int(255*c) for c in rgba[:3])  # drop alpha, scale to 0–255
            a = 255*int(self.grid[y, x] == 0)  # belief is only obaque over free cells! 
            color = (r, g, b, a) #
            pygame.draw.rect(surface, color, pygame.Rect(x*d, y*d, d, d))
        screen.blit(surface, (0, 0))
    
    def draw_observation(self):
        # Draw observation in upper right corner of screen 
        obs = self.observation
        # obs = self.true_observation
        if obs is None:
            return
        d = self.cell_size
        r = SENSOR_LENGTH
        xloc = ACTUAL_SCREEN_WIDTH-(len(obs)*d)  # x-location of observation grid
        # Draw cells 
        for i, j in np.ndindex(obs.shape):
            c = 255*(1-obs[i, j]) 
            color = (c, c, c) 
            # center point is red:
            if i == r and j == r:
                color = (255, 0, 0)
            pygame.draw.rect(screen, color, pygame.Rect(xloc+(j*d), i*d, d, d))
        # Draw gridlines for observation
        self.draw_gridlines(xloc, len(obs))
    
    def draw_info(self):
        # Write timestep in bottom right corner
        info = f"t = {self.timestep}"
        font = pygame.font.SysFont("Arial", 15)
        text = font.render(info, True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.topleft = (SCREEN_SIZE+20, SCREEN_SIZE-30)
        screen.blit(text, text_rect)
    
    def save_frame(self): 
        # frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame_surface = pygame.display.get_surface().copy()
        frame_array = pygame.image.tostring(frame_surface, "RGB")
        frame = Image.frombytes("RGB", frame_surface.get_size(), frame_array)
        self.frames.append(frame) 
    
    def gif(self):
        if not SCREEN_CAPTURE or self.timestep == 0: 
            return
        print("Saving GIF...") 
        timestamp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if not os.path.exists("gifs"):
            os.mkdir("gifs")
        self.frames[0].save(
            f"gifs/gridworld {timestamp}.gif", 
            save_all=True, 
            append_images=self.frames[1:], 
            # loop=1, 
            duration=100
        )
    
    def render(self):
        screen.fill(BACKGROUND_COLOR)
        self.draw_info()
        self.draw_grid()
        if belief_visible:
            self.draw_belief()
        else:
            self.draw_robot()
        self.draw_gridlines()
        self.draw_observation()
        pygame.display.flip()


if __name__ == "__main__":
    env = Environment(dim=30, n_obstacles=30)
    # Main loop
    running = True
    while running:
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            mods = pygame.key.get_mods()
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                match event.key:
                    case pygame.K_ESCAPE:
                        running = False
                    case pygame.K_SPACE:
                        belief_visible = not belief_visible
                    case pygame.K_UP:
                        env.move(U)
                    case pygame.K_DOWN:
                        env.move(D)
                    case pygame.K_LEFT:
                        env.move(L)
                    case pygame.K_RIGHT:
                        env.move(R)
                    case pygame.K_s if (mods & pygame.KMOD_META):
                        env.gif()
        env.render()
        clock.tick(60)  
    pygame.quit()
