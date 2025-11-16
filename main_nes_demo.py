from main_nes import * 
from PIL import Image 
import imageio.v2 as imageio 
import datetime as dt 


# True map
NUM_BEACONS = 200  # 100
# Screen 
SCREEN_CAPTURE = True 
# Predicted map 
# Colors
BACKGROUND_COLOR = (30, 30, 30)
ROBOT_COLOR = (200, 50, 50)
SENSOR_COLOR = (50, 200, 50)
OBSTACLE_COLOR = (100, 100, 100)
SENSOR_HIT_COLOR = (230, 0, 0)
PASSIVE_BEACON_COLOR = (130, 130, 130) 
ACTIVE_BEACON_COLOR = (0, 255, 0) 
# Robot
AUTONOMOUS = True  
# Robot sensors 
OMNI_RANGE = SENSOR_RANGE  # range around robot where it can detect beacons 
OCCUPANCY_ERROR = 0.001  # 0.01, 0.001
OCCUPANCY_LOG_ODDS = np.log((1-OCCUPANCY_ERROR)/OCCUPANCY_ERROR)  # log-odds of occupancy
# Policy 
EXPLORATION_RATE = 0.1  # currently only used in baseline policy 
# Target 
TARGET_IMAGE_SIZE = 35
# Visibility 
flag_changed = False 
obstacles_visible = True  
robot_info_visible = True 
trajectory_visible = True 
beacon_visible = False 
grid_visible = False  
occupancy_values_visible = False
# Kalman filter 
V = np.diag([1e-6, 1e-6, 1e-6])  # initial state-estimate covariance 
Q = np.diag([1e-6, 1e-6, 1e-6])  # sensor-noise covariance


class Demo(Evolution):

    # Initialization ---------------------------------------------------------------------------------------------
    
    def reset(self, reset_map=True, reset_screen_capture=True):
        """
        Reset the map and robot, but not the evolution or 
        This method can be called for each new offspring to generate a new map and robot,
        without resetting the ID or evolutionary history. 
        """
        self.timestep = 0 
        if reset_map:
            self.initialize_map()
            self.initialize_beacons()
            self.initialize_target()
        self.initialize_robot()
        self.initialize_belief()
        self.initialize_grid()
        if SCREEN_CAPTURE and reset_screen_capture:
            self.prev_frame = None 
            self.frames = []
    
    def initialize_beacons(self):
        """
        Use obstacle corners as beacons.
        Only the squares are technically obstacles, and when two squares touch, 
        I do not count this as a corner and it is not a beacon.
        I simply take all squares' corners and then remove all points that 
        appear more than once.
        """
        self.beacons = []
        self.beacon_distances = []
        self.beacon_bearings = []
        
        '''
        # Just for some testing, generate beacons completely randomly 
        #  and independently of obstacles!! (i.e. not as corners of obstacles!)
        if NUM_BEACONS > 0:
            MIN_BEACON_DISTANCE = 0.5 * OMNI_RANGE
            while True:
                x, y = np.random.uniform(5, SCREEN_SIZE-5, size=2)
                # Only actually add to self.beacons if the proposed beacons is sufficiently 
                #  far away from all existing beacons! 
                #  This minimum distance (MIN_BEACON_DISTANCE) should not be too large, because then
                #  the robot will never be in range of 3 beacons for trilateration!! 
                valid = True 
                for other_beacon in self.beacons:
                    if np.linalg.norm(np.array(other_beacon)-np.array([x, y])) < MIN_BEACON_DISTANCE:
                        valid = False 
                        break 
                if valid:
                    self.beacons.append((x, y))
                    self.beacon_distances.append(inf) 
                    self.beacon_bearings.append(0)
                # Once we have NUM_BEACONS beacons, leave the while loop
                if len(self.beacons) == NUM_BEACONS:
                    break 
        '''

        if (NUM_BEACONS > 0) or (NUM_BEACONS is None):
            corners = set()
            remove = set()
            for obstacle in self.obstacles:
                for dx in [0, CELL_SIZE]:
                    for dy in [0, CELL_SIZE]:
                        corner = (obstacle.x+dx, obstacle.y+dy)
                        # TODO: (only a small detail) rectangles that touch the edges of the screen should
                        #  also be removed 
                        if corner not in corners:
                            corners.add(corner)
                        else:
                            remove.add(corner)
            # Remove those corners that have appeared more than once 
            corners = list(corners - remove)
            # Go through these corners and turn then into pygame.Rect objects, 
            #  and then put them into self.beacons and self.beacon_distances
            #  The latter is simply the list of detected distances, which are initialized to infinity 
            # Only actually keep NUM_BEACONS random beacons: 
            if NUM_BEACONS is not None:
                np.random.shuffle(corners)
                corners = corners[:NUM_BEACONS]
            for i, (x, y) in enumerate(corners):
                self.beacons.append((x, y))
                self.beacon_distances.append(inf) 
                self.beacon_bearings.append(0)

    def initialize_belief(self):
        """
        Initialize the mean and covariance of the belief according to the Kalman Filter.
        Belief is the robot's belief of its own pose in absolute terms! 
        Initialize belief at spawn point (local localization!) and set a small variance.
        For global localization, use random initial point and large variance.
        """
        self.belief_mean = np.array([self.spawn_x, self.spawn_y, self.robot_angle])
        self.belief_cov = V 
        # Initialize belief trajectory 
        self.belief_trajectory = [self.belief_mean[:2]]

    # Step: transition to the next state -------------------------------------------------------------------------
    
    def step(self, action):
        """
        Transition from current environment state to the next 
        """
        global flag_changed
        self.compute_sensors()
        self.compute_omni()
        self.compute_occupancy()
        self.update_belief(action) 
        if SCREEN_CAPTURE:
            self.save_frame()
        self.render()
        clock.tick(60) 
        self.timestep += 1
        flag_changed = False 
    
    # Sensors ----------------------------------------------------------------------------------------------------

    def compute_omni(self):
        """
        Checks which beacons are in range of the robot's omnidirectional sensor 
        Computes the observation vector based on omni-sensor readings
        First applies trilateration to omni-sensor readings (detected beacon positions, distances, bearings) 
        and then outputs the estimated robot pose according to this trilateration.
        This estimated pose wil be used as the robot's "observation".
        """
        # Check which beacons are in range (detected)
        detected_beacon_positions = [] 
        detected_beacon_distances = []
        detected_beacon_bearings = []
        self.beacon_distances = [inf for _ in self.beacons]
        robot_pos = np.array([self.spawn_x+self.robot_x, self.spawn_y+self.robot_y])
        for j, (beacon_x, beacon_y) in enumerate(self.beacons):
            # Use ACTUAL beacon location and ACTUAL robot location to get distance measure 
            #  and only later add noise: 
            beacon_pos = np.array([beacon_x, beacon_y])
            distance = np.linalg.norm(beacon_pos-robot_pos)
            if distance < OMNI_RANGE:
                detected_beacon_positions.append((beacon_x, beacon_y))
                # Distance
                self.beacon_distances[j] = distance 
                detected_beacon_distances.append(distance)
                # Bearing 
                #  Robot doesn't actually know true self.robot_angle, but it knows the 
                #  relative angle between its orientation and a beacon that is perceives -> "bearing" 
                bearing = np.atan2(beacon_y-(self.spawn_y+self.robot_y), beacon_x-(self.spawn_x+self.robot_x))-self.robot_angle 
                # Wrap into interval [0, 2π]:
                bearing %= 2*pi 
                self.beacon_bearings[j] = bearing 
                detected_beacon_bearings.append(bearing)
        # Compute "observation" (trilaterated robot position in absolute coordinates) 
        self.observation = None 
        # Trilateration
        trilateration = trilaterate(detected_beacon_positions, detected_beacon_distances, detected_beacon_bearings) 
        if trilateration is not None:
            # Add sensor noise (using matrix Q) 
            noise = np.random.multivariate_normal(np.zeros(3), Q, size=1)[0]
            trilateration += noise 
            # Save this as the observation 
            self.observation = trilateration 

    # Kalman filter ----------------------------------------------------------------------------------------------

    def update_belief(self, action): 
        """
        Updates the belief using the Kalman filter 
        Parameter 'action' must be a tuple containing (v_l, v_r) 
        """
        mean = self.belief_mean 
        cov = self.belief_cov 

        # Motion update ("Prediction")
        #  Simple model of how model moves given the action without 
        #  any consideration of obstacles
        v_l, v_r = action 
        v_linear = MOVE_SPEED * (v_l + v_r) / 2  # Average speed of both motors
        v_angular = ROTATE_SPEED * (v_r - v_l) / ROBOT_RADIUS  # Differential rotation
        # Believed orientation:
        angle = mean[2]
        # u vector: action (we're using linear velocity, angular velocity here)
        u = np.array([v_linear, v_angular]) 
        # A matrix (effect of environment on next state)
        #  Assume environment has no effect -> use identity matrix 
        #  So we can just leave it 
        # B matrix (effect of taking action u on next state)
        B = np.array([
            [np.cos(angle), 0],
            [np.sin(angle), 0],
            [0,             1]
        ])
        # Update mean 
        mean = mean + B.dot(u) 
        # Make sure angle stays in range [0, 2π]:
        mean[2] %= 2*pi 
        # Update covariance (belief_var)
        #  Actually just the diagonal of the covariance matrix, as we assume independence, 
        #  i.e. just the variances 
        cov = cov + R 

        # Sensor update ("Correction")
        # We can only trilaterate if there are at least 3 points (2 points with angle), 
        #  so if we can't trilaterate, we can't do the correction step? 
        if self.observation is not None:
            # Kalman gain
            K = cov @ np.linalg.inv(cov + Q)
            # Updated state estimate
            mean = mean + K @ (self.observation - mean)
            # Updated covariance estimate
            I = np.eye(3)
            cov = (I - K) @ cov

        # Set belief to the result of the Kalman filter 
        self.belief_mean = mean
        self.belief_cov = cov

        # Append believed pose to belief trajectory
        self.belief_trajectory.append(self.belief_mean[:2])

    # Navigation -------------------------------------------------------------------------------------------------

    def baseline_policy(self):
        """
        Baseline policy (handcrafted, as opposed to learned or evolved): 
        Pick direction with greatest distance to any obstacle according to sensor readings 
        If all sensor readings are the same, just keep going 
        With some probability, pick a random direction (random offset added to current direction)
        """
        # Determine best angle based on sensor data, then move in that direction
        #  But certain percentage of the time, pick a random direction (random offset added to current direction) 
        if sum(hit for _, _, hit, _ in self.sensors) == 0: 
            # If no sensor detects a hit, just keep going in current direction 
            best_angle = self.robot_angle
        elif np.random.uniform() < EXPLORATION_RATE:
            # Pick random direction close to the last best direction
            best_angle = (self.robot_angle + np.random.uniform(-pi, pi)) % NUM_SENSORS 
        else:
            # Choose the best direction based on sensor data
            best_direction = np.argmax([distance for _, _, _, distance in self.sensors]).item() 
            best_angle = (self.robot_angle + best_direction * SENSOR_ANGLE_STEP) % (2*pi)
        
        # Compute difference between current angle and desired angle 
        angle_diff = (best_angle - self.robot_angle) % (2*pi)
        angle_diff = np.clip(angle_diff, -pi/2, pi/2)  # limit the robot to turn at most pi/2 radians (90°) per step
        
        return angle_diff

    # Drawing ----------------------------------------------------------------------------------------------------
    
    def draw_obstacles(self):
        """
        Render obstacle blocks.
        """
        if not obstacles_visible:
            return 
        for obs in self.obstacles: 
            pygame.draw.rect(screen, OBSTACLE_COLOR, obs) 
    
    def draw_beacons(self):
        """
        Render beacons and circles around detected beacons with radius being the distance
        to the robot.
        """
        if not beacon_visible:
            return 
        for i, ((beacon_x, beacon_y), distance) in enumerate(zip(self.beacons, self.beacon_distances)):
            color = (0, 0, 0)
            if distance < inf:
                color = ACTIVE_BEACON_COLOR
                # Draw circle around beacon with radius = distance to robot 
                #  Since we know this is the distance between the robot and the beacon, the robot 
                #  must be somewhere on that circle (or close to it, due to noise)
                pygame.draw.circle(screen, color, (beacon_x, beacon_y), distance, 1)
            else:
                color = PASSIVE_BEACON_COLOR
            pygame.draw.circle(screen, color, (beacon_x, beacon_y), 3)
            '''
            # Draw beacon id 
            beacon_text = font.render(str(i), True, color)
            beacon_text_rect = beacon_text.get_rect(center=(beacon_x, beacon_y))
            screen.blit(beacon_text, beacon_text_rect)
            '''
    
    def draw_grid(self):
        """
        Render occupancy grid map using the occupancy log-odds to determine the color intensity. 
        Specially highlight those grid cells that are in the robot's sensor range.
        """
        if not grid_visible:
            return 
        # Draw occupancy values 
        # Instead of plotting the probability, I will plot the normalized log-odds, because then
        #  differences are more visible
        # p = 1-1/(1+np.exp(self.occupancy))
        # q = (self.occupancy-min(self.occupancy))/(max(self.occupancy)-min(self.occupancy))
        # Use constant min (-500) and max (500) values for q:
        q = (np.clip(self.occupancy, -500, 500)-(-500))/(500-(-500))
        color_steepness = 2 
        for i, cell_x in enumerate(range(0, SCREEN_SIZE, GRID_CELL_SIZE)):
            for j, cell_y in enumerate(range(0, SCREEN_SIZE, GRID_CELL_SIZE)):
                # Using inverse sigmoid function to map log-odds to nice colors values
                #  This exaggerates the differences between occupancy values close to 0 and 1, 
                #  because most of the time, occupancy values are actually very close to 0 or 1
                color = int(255 * np.clip(1/2-1/color_steepness*np.log(1/max(q[i,j], 1e-6)-1), 0, 1))
                pygame.draw.rect(screen, (0, color//4, color), (cell_x, cell_y, GRID_CELL_SIZE, GRID_CELL_SIZE))
                # Occupancy value as probability
                if occupancy_values_visible:
                    cell_text = font_small.render(str(int(self.occupancy[i,j])), True, (255, 255, 255))
                    cell_text_rect = cell_text.get_rect(center=(cell_x+GRID_CELL_SIZE//2, cell_y+GRID_CELL_SIZE//2))
                    screen.blit(cell_text, cell_text_rect)
        # Draw grid lines
        for i in range(0, SCREEN_SIZE, GRID_CELL_SIZE):
            pygame.draw.line(screen, (50, 50, 50), (i, 0), (i, SCREEN_SIZE))
            pygame.draw.line(screen, (50, 50, 50), (0, i), (SCREEN_SIZE, i))
        # Highlight grid cells that are in range
        for i, cell_x in enumerate(range(0, SCREEN_SIZE, GRID_CELL_SIZE)):
            for j, cell_y in enumerate(range(0, SCREEN_SIZE, GRID_CELL_SIZE)):
                distance = self.grid_distances[i,j]
                if distance < inf:
                    # Draw line from robot to grid cell center
                    if self.detected_occupancy[i,j]:
                        # In range and occupancy is detected
                        pygame.draw.rect(screen, (255, 255, 0), (cell_x, cell_y, GRID_CELL_SIZE, GRID_CELL_SIZE))
                    else:
                        # In range and NO occupancy is detected
                        pygame.draw.rect(screen, (255, 255, 0), (cell_x, cell_y, GRID_CELL_SIZE, GRID_CELL_SIZE), 1)

    def draw_robot(self):
        """
        Render the robot's body itself, trajectory, robot-related information (text),
        and robot-related Kalman estimates (estimated pose)
        """
        # Draw trajectory (past positions)
        if trajectory_visible:
            points = [(self.spawn_x+x, self.spawn_y+y) for x, y in self.trajectory]
            # pygame.draw.lines(screen, (90, 30, 30), False, points, 3)
            pygame.draw.lines(screen, (200, 150, 150), False, points, 3)
        # Get robot's current (absolute) position
        x, y = self.spawn_x + self.robot_x, self.spawn_y + self.robot_y 
        if robot_info_visible:
            # Draw sensors
            for sensor_x, sensor_y, hit, _ in self.sensors:
                color = SENSOR_HIT_COLOR if hit else SENSOR_COLOR
                pygame.draw.line(screen, color, (x, y), (self.spawn_x+sensor_x, self.spawn_y+sensor_y), 2)
                # pygame.draw.circle(screen, color, (self.spawn_x+sensor_x, self.spawn_y+sensor_y), 3)
            # Draw omnidirectional sensor (circle around robot indicating OMNI_RANGE)
            '''
            pygame.draw.circle(screen, ROBOT_COLOR, (x, y), OMNI_RANGE, 2)
            '''
            # Draw lines from robot to detected beacons 
            if beacon_visible:
                for (beacon_x, beacon_y), distance, bearing in zip(self.beacons, self.beacon_distances, self.beacon_bearings):
                    if distance < inf:
                        pygame.draw.line(screen, ACTIVE_BEACON_COLOR, (x, y), (beacon_x, beacon_y), 2)
                        '''
                        # Draw beacon distances
                        #  I'm dividing by SENSOR_STEP_SIZE so we get the same units as the sensor distances 
                        distance_text = font.render(str(int(distance//SENSOR_STEP_SIZE)), True, (255, 255, 255))
                        distance_text_rect = distance_text.get_rect(center=(beacon_x+10, beacon_y+10))
                        screen.blit(distance_text, distance_text_rect)
                        '''
                        '''
                        # Draw beacon bearings
                        bearing_text = font.render(str(int(np.rad2deg(2*pi-bearing)))+"°", True, (255, 255, 255))
                        bearing_text_rect = bearing_text.get_rect(center=(beacon_x+10, beacon_y+10))
                        screen.blit(bearing_text, bearing_text_rect)
                        '''
        # Draw the robot body (disk)
        pygame.draw.circle(screen, ROBOT_COLOR, (x, y), ROBOT_RADIUS)
        # Draw the direction line (heading)
        heading_x = x + ROBOT_RADIUS * math.cos(self.robot_angle)
        heading_y = y + ROBOT_RADIUS * math.sin(self.robot_angle)
        pygame.draw.line(screen, (0, 0, 0), (x, y), (heading_x, heading_y), 3)
        # Draw estimated position according to trilateration 
        if beacon_visible and self.observation is not None:
            est_x, est_y, _ = self.observation 
            pygame.draw.circle(screen, (255, 255, 255), (est_x, est_y), 3)
        # Draw motor speed text inside the robot body
        if robot_info_visible:
            # vel_text = font.render(f"[{self.v_l}, {self.v_r}]", True, (255, 255, 255))
            # screen.blit(vel_text, (x-11, y-6))  
            for i, (_, _, hit, distance) in enumerate(self.sensors):
                # Draw sensor distance values around the robot (text)
                sensor_angle = self.robot_angle + i * SENSOR_ANGLE_STEP
                text_x = x + (ROBOT_RADIUS + 20) * math.cos(sensor_angle)  # Position outside the robot
                text_y = y + (ROBOT_RADIUS + 20) * math.sin(sensor_angle)  
                # Only draw if finite distance 
                if distance < inf:
                    proximity = 1-(distance/(SENSOR_RANGE//SENSOR_STEP_SIZE)) 
                    # distance_text = font.render(str(int(distance)), True, (255, 255, 255) if hit else (100, 100, 100))
                    distance_text = font.render(str(round(proximity, 2)), True, (255, 255, 255) if hit else (100, 100, 100))
                    text_rect = distance_text.get_rect(center=(text_x, text_y))
                    screen.blit(distance_text, text_rect)

    def draw_belief(self):
        """
        Render the estimated robot's body and trajectory according to the Kalman filter.
        """
        # Draw trajectory (i.e. past believed positions)
        if trajectory_visible:
            pygame.draw.lines(screen, (30, 30, 90), False, self.belief_trajectory, 3)
        # Get current believed position (mean and variance)
        mean_x, mean_y, mean_angle = self.belief_mean 
        cov = self.belief_cov 
        
        # Draw covariance 
        # (assuming covariance is always a diagonal matrix -> eigenvectors are [1, 0], [0, 1]
        #  and eigenvalues are diagonal entries)
        n_std = 2  # (n_std=1, conf=68%), (n_std=2, conf=95%)
        eigvals = cov.diagonal()[:2]
        ellipse_width, ellipse_height = 2 * n_std * np.sqrt(eigvals)
        # print((ellipse_width.round(2).item(), ellipse_height.round(2).item()))
        # Draw ellipse (position uncertainty)
        ellipse = pygame.Rect(mean_x-ellipse_width/2, mean_y-ellipse_height/2, ellipse_width, ellipse_height)
        # pygame.draw.rect(screen, (30, 30, 200), ellipse, 2) 
        pygame.draw.ellipse(screen, (30, 30, 200), ellipse, 2)
        # Draw orientation uncertainty
        # angle = np.arctan2(*vecs[:,0][::-1])  # angle of major axis (in radians)

        # Draw the robot body (disk) at (mean) believed location 
        pygame.draw.circle(screen, (50, 50, 255), (mean_x, mean_y), ROBOT_RADIUS)
        # Draw the direction line (heading) of (mean) believed orientation 
        heading_x = mean_x + ROBOT_RADIUS * math.cos(mean_angle)
        heading_y = mean_y + ROBOT_RADIUS * math.sin(mean_angle)
        pygame.draw.line(screen, (0, 0, 0), (mean_x, mean_y), (heading_x, heading_y), 3)

    def draw_target(self):
        """
        Render the target by drawing the star image in the target cell
        (see initialize_target method).
        """
        screen.blit(TARGET_IMAGE, (self.target_x-TARGET_IMAGE_SIZE/2, self.target_y-TARGET_IMAGE_SIZE/2))

    def save_frame(self): 
        """
        Append current frame to list of frames, so that this list can later be turned into a video. 
        """
        frame_surface = pygame.display.get_surface().copy()
        frame_array3d = pygame.surfarray.array3d(frame_surface) 
        # Only add if frame is not empty and different from last frame 
        if (frame_array3d.sum() > 0) and (self.prev_frame is None or (frame_array3d != self.prev_frame).any()):
            frame_array = pygame.image.tostring(frame_surface, "RGB")
            frame = Image.frombytes("RGB", frame_surface.get_size(), frame_array)
            self.frames.append(frame) 
            self.prev_frame = frame_array3d 
    
    def to_video(self):
        """
        Turn saved list of frames into a mp4 file and save it. 
        This method can be called using the keyboard shortcut CMD + S (or CTRL + S). 
        """
        if len(self.frames) == 0: 
            return
        if not os.path.exists("screencapture"):
            os.mkdir("screencapture")
        timestamp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"screencapture/main {timestamp}.mp4"
        print(f"Saving file as {filename}") 
        # Save first frame as image
        # with imageio.get_writer(f"screencapture/main {timestamp}.png") as writer: 
        #     writer.append_data(np.array(frames[1]))
        with imageio.get_writer(filename, fps=60, codec="libx264") as writer: 
            for frame in self.frames[1:]:
                writer.append_data(np.array(frame))
    
    def render(self):
        """
        Call all draw methods. 
        This method is called each timestep to render the current state of the map and robot. 
        """
        screen.fill(BACKGROUND_COLOR)
        self.draw_grid()
        self.draw_obstacles()
        self.draw_beacons()
        self.draw_belief() 
        self.draw_target()
        self.draw_robot()
        pygame.display.flip()


if __name__ == "__main__":
    # PyGame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Mobile Robot")
    clock = pygame.time.Clock()
    pygame.font.init()
    font = pygame.font.SysFont("Calibri", 12)
    font_small = pygame.font.SysFont("Calibri", 8)
    TARGET_IMAGE = pygame.image.load("resources/target.png").convert_alpha() 
    TARGET_IMAGE = pygame.transform.scale(TARGET_IMAGE, (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE)) 
    
    # Environment 
    env = Demo() 
    
    if AUTONOMOUS:
        env.load_weights()  # load particular or most recent weights 

    # Main loop
    running = True 
    while running:
        # System controls
        mods = pygame.key.get_mods()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False 
            elif event.type == pygame.KEYDOWN:
                match event.key: 
                    case pygame.K_ESCAPE: 
                        running = False 
                    case pygame.K_b:
                        # "B" key toggles beacon visibility 
                        beacon_visible = not beacon_visible
                        flag_changed = True
                    case pygame.K_g:
                        # "G" key toggles grid visibility
                        grid_visible = not grid_visible
                        flag_changed = True
                    case pygame.K_i:
                        # "I" key toggles robot info visibility 
                        robot_info_visible = not robot_info_visible
                        flag_changed = True
                    case pygame.K_o:
                        # "O" key toggles obstacle visibility
                        obstacles_visible = not obstacles_visible
                        flag_changed = True
                    case pygame.K_r:
                        # "R" resets the environment 
                        env.reset()
                    case pygame.K_s if (mods & (pygame.KMOD_META | pygame.KMOD_CTRL)):
                        # CMD + "S" key saves run so far as mp4 file 
                        env.to_video()
                    case pygame.K_t:
                        # "T" key toggles trajectory visibility
                        trajectory_visible = not trajectory_visible
                        flag_changed = True
                    case pygame.K_v:
                        # "V" key toggles occupancy values
                        occupancy_values_visible = not occupancy_values_visible
                        flag_changed = True

        # Robot controls 
        if AUTONOMOUS:
            # Demonstrate evolved policy:
            env.move_autonomously()
        else:
            v_l, v_r = 0, 0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:  
                # "W" key controls the left motor
                v_l = 1
            if keys[pygame.K_UP]:  
                # UP arrow key controls the right motor
                v_r = 1
            # Move the robot according to (v_l, v_r)
            env.move((v_l, v_r))
    
    pygame.quit() 
