# IsaacLab UAV Swarm Navigation
<p align="center">
  <img src="assets/isaac-sim-env.png" width="80%" alt="UAV Swarm Environment in Isaac Sim">
</p>


## Overview

This project explores **Multi-Agent Reinforcement Learning (MARL)** algorithms for controlling a **quadrotor swarm** using **Crazyflie** micro-UAVs within the **IsaacLab + IsaacSim** simulation stack.  
The objective is to train scalable decentralized swarm policies with full GPU-accelerated physics.

The final system will enable:
- Multi-agent UAV navigation  
- Obstacle avoidance  
- Inter-agent collision avoidance  
- Formation learning  
- Curriculum-based progressive training  

---



## Methodology

The training process is divided into curriculum-based stages to ensure stable policy learning and reduce environment complexity:

### **Stage 1 — Hovering**
Basic stabilization in place using thrust + attitude control.
<p align="center">
  <img src="assets/swarm_progress.gif" width="80%" alt="UAV Swarm Hover Training Demo">
</p>


### **Stage 2 — Point-to-Point Navigation (no obstacles)**
Agents learn to reach designated targets without environment disturbances.

### **Stage 3 — Navigation with Static Obstacles**
Adds randomized cubic obstacles to develop obstacle-avoidance behavior.
<p align="center">
  <img src="assets/isaac-sim-env.png" width="80%" alt="Stage 3 Obstacle Environment">
</p>

### **Stage 4 — Multi-Agent Formation Navigation**
Multiple Crazyflies maintaining formation while navigating toward shared goals.

### **Stage 5 — Swarm Obstacle Navigation**
Combined formation control + obstacle avoidance for cooperative swarm behavior.

---

## Demo

The animations above show the simulation environment with obstacles and the training progression.  
For more videos and logs, check the `/assets` folder in this repository.

---

## Installation & Usage (coming soon)

Instructions on how to:
- build the IsaacLab extension  
- register MARL tasks  
- train using **SKRL IPPO/MAPPO**  
- run the swarm simulation  

will be added shortly.

