# Autonomous-Robotic-System
Autonomous robotic system simulating exploration, localization, mapping, and navigation using differential drive, Kalman Filter, EKF-SLAM, and a neural controller evolved via Natural Evolution Strategies (NES). Developed in Python (Pygame, NumPy, SciPy, Matplotlib) for the Autonomous Robotic Systems course.

This project implements a full **autonomous robot pipeline** — from simulation and localization to mapping and learned navigation — for a **differential-drive robot** in a 2D environment with obstacles.  
It was developed for the **Autonomous Robotic Systems (ARS)** course at **Maastricht University**.

---

## Project Overview

The robot autonomously explores an unknown environment, avoids obstacles, constructs an occupancy grid map, and finds a target (“rescue mission”) without human input.

The project was completed in **four milestones**:
1. **Simulation:** Differential-drive robot simulation with obstacle avoidance  
2. **Localization:** Pose estimation using a **Linear Kalman Filter**  
3. **Mapping:** Occupancy grid + **EKF-SLAM** for simultaneous localization and mapping  
4. **Navigation:** Learned neural controller trained via **Natural Evolution Strategies (NES)** for autonomous exploration

---

## System Components

| Module | Key Algorithm | Purpose |
|--------|----------------|----------|
| **Simulation** | Differential-drive kinematics | Robot motion & sensor data |
| **Localization** | Kalman Filter | Estimate robot pose under noise |
| **Mapping** | Occupancy grid, EKF-SLAM | Build map & reduce pose uncertainty |
| **Navigation** | RNN + NES optimization | Learn control policy for exploration |

---
## Example Outputs
Simulation of a mobile robot using EKF-SLAM, occupancy grid mapping and natural evolution strategies (NES), built for the course "Autonomous Robotic Systems" (2025) in M.Sc. Artificial Intelligence at Maastricht University.
### Fitness and diversity of the evolved agent over time

<p align="center">
    <img src="figures/plot.png" alt="" width=1000/>
</p>

1D example of an optimization process using natural evolution strategies. The solution is $x=9$ (value with maximum fitness) and the initialization is at $x=0$. Over 20 generations, the parameters of the search distribution are updated such that its mean is approximately at the solution value. 

<p align="center">
    <img src="figures/nes2.gif?v=1" alt="" style="width: 970px; height: auto;"/>
</p>

---

### Demonstration of the simulated robot's occupancy grid mapping and behavior

<p align="center">
    <img src="figures/occupancy2.gif?v=1" alt="" style="width: 500px; height: auto;"/>
</p>

<p align="center">
    <img src="figures/behavior1.gif?v=1" alt="" style="width: 500px; height: auto;"/>
</p>

Earlier version of the evolved policy under a different fitness function, which rewarded displacement from the spawn point.

<p align="center">
    <img src="figures/behavior2.gif?v=1" alt="" style="width: 500px; height: auto;"/>
</p>

---

### Demonstration of trilateration

Actual state in red and belief state in blue. The actual state may deviate due to motion noise, but the belief state then gets corrected by applying the Bayes filter using the trilateration data. 

<p align="center">
    <img src="figures/trilateration.gif?v=1" alt="" style="width: 500px; height: auto;"/>
</p>

---
### Demonstration of a discrete Bayes filter (Histogram filter) for localization 

The 8 cells adjacent to the agent (also shown in the upper right corner of the image) combined with the action is the (noisy) evidence that is used to update the belief state. 

<p align="center">
    <img src="figures/localization-discrete-bayes-filter.gif?v=1" alt="" style="width: 500px; height: auto;"/>
</p>



## References
- Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics* – MIT Press  
- Wierstra et al. (2011). *Natural Evolution Strategies*  
- Tipaldi & Burgard (2015). *Occupancy Mapping*  
- Pygame & SciPy documentation  
