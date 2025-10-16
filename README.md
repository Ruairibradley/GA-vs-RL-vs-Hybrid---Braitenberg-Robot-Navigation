# GA-vs-RL-vs-Hybrid---Braitenberg-Robot-Navigation
The comparative study of Genetic Algorithms, Reinforcement Learning, and Hybrid models for Braitenberg robots learning in learning autonomous light seeking behaviour.

This project explores and compares the performance of Genetic Algorithms (GA), Reinforcement Learning (Q-Learning), and a Hybrid GA+QL approach for training *Braitenberg vehicles* to autonomously seek light.  
It was developed as part of the *Acquired Intelligence and Adaptive Behaviour* module at the University of Sussex (2025).

---

## Overview

Braitenberg vehicles are simple sensor-based robots that exhibit intelligent behaviour from basic rules, they have two light sensors connected to two motors powering the wheels.  
This project investigates how different learning paradigms - evolutionary vs reinforcement-based - influence the robot’s ability to adapt to dynamic environments.
All models were developed offline in simulation to find weightings for the motors and then transfered onto the real robot for analysis on performance, and also to highlight sim to real limitations. 

Three models were implemented:
- Genetic Algorithm (GA): Evolves controller weights over generations based on performance (fitness) in the environment.  
- Q-Learning (RL): Learns optimal actions via trial-and-error and reward feedback.  
- Hybrid GA+QL: Combines GA evolution for initialisation with Q-Learning fine-tuning to balance exploration and exploitation.

---

## Objectives
- Compare GA, RL, and hybrid learning models in light-seeking tasks.
- Evaluate performance across noise levels and generalisation scenarios.
- Analyse convergence rates, adaptability, and stability of each model.

---

## Implementation Details

Language: Python  
Libraries: `numpy`, `matplotlib`, `random`, `math`  

Core Components:
- Environment simulation with adjustable noise and light source position.
- Braitenberg robot with two sensors and motor control parameters.
- Fitness evaluation and reward feedback system.
- Comparative visualisation of performance metrics and learning curves.

---

## How to Run
Clone this repository:

git clone https://github.com/Ruairibradley/AIAB_Code_RuairiBradley-279123.git
cd AIAB_Code_RuairiBradley-279123



Open the project file:
Launch your preferred Python IDE or text editor (e.g. VS Code, PyCharm, Jupyter Notebook, or IDLE).
Open the main file:

AIAB_Code_RuairiBradley-279123.py


Follow the in-code commented instructions for specific experiment set up:
The script includes detailed commented sections explaining how to:
Configure and run each model (Genetic Algorithm, Q-Learning, and Hybrid GA+QL).
Create and adjust simulation environments (e.g. noise level, light source position, iteration count).
Run experiments for each approach individually or compare them side by side.
Simply follow the comments in the file to uncomment or modify parameters as needed.

Run the simulation:
Execute the file directly from your IDE, or run it via the command line:

python AIAB_Code_RuairiBradley-279123.py


View results:
Once executed, the program will generate performance graphs comparing model convergence, adaptability, and success rates.
Output plots may display automatically or be saved within the working directory depending on your chosen configuration.


