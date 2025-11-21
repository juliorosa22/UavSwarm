# IsaacLab UAV Swarm Navigation

<p align="center">
  <img src="assets/swarm_training.gif" width="80%" alt="UAV Swarm Hover Training Demo">
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

### **Stage 0 — Hovering**
Basic stabilization in place using thrust + attitude control.

### **Stage 1 — Point-to-Point Navigation (no obstacles)**
Agents learn to reach designated targets without environment disturbances.

### **Stage 2 — Navigation with Static Obstacles**
Adds randomized cubic obstacles to develop obstacle-avoidance behavior.

### **Stage 3 — Multi-Agent Navigation**
Multiple Crazyflies operating simultaneously in the same environment.

### **Stage 4 — Inter-Agent Collision Avoidance**
Penalty and termination for drone-drone collisions, promoting safe spacing.

### **Stage 5 — Formation Learning (no obstacles)**
Agents learn to maintain spatial structure (e.g., V-formation) during navigation.

---

## Demo

The animation above shows multiple training phases (hovering → navigation → multi-agent behavior).  
For more videos and logs, check the `/assets` folder in this repository.

---

## Installation & Usage (coming soon)

Instructions on how to:
- build the IsaacLab extension  
- register MARL tasks  
- train using **SKRL IPPO/MAPPO**  
- run the swarm simulation  

will be added shortly.

