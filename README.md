# IsaacLab UAV Swarm navigation 

## Overview

This project aims to explore MARL algorithms for controlling a Quadcopter Swarm using the Crazyflies quadcopter as the main robot template using the IsaacLab and Isaacsim MARL stack for fast policy training.

**Methodology:**
The methodology adopted for a stable learning process consists to use curriculum learning. by breaking down the final task of Swarm navigation within a obstacle enviroment with inter-agent collision capabilities in the following subtasks:
<h2>Stage 0: Hovering</h2>
<h2>Stage 1: point-to-point navigation (no obstacles)<h2>
<h2>Stage 2: navigation with static obstacles<h2>
<h2>Stage 3: Multi-agent<h2>
<h2>Stage 4: inter-agent collision avoidance<h2>
<h2>Stage 5: Learn formation without obstacles<h2>