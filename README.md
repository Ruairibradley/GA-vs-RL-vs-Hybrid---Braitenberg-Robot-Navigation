# GA-vs-RL-vs-Hybrid---Braitenberg-Robot-Navigation
The comparative study of Genetic Algorithms, Reinforcement Learning, and Hybrid models for Braitenberg robots learning in learning autonomous light seeking behaviour.

This project explores and compares the performance of Genetic Algorithms (GA), Reinforcement Learning (Q-Learning), and a Hybrid GA+QL approach for training *Braitenberg vehicles* to autonomously seek light.  
It was developed as part of the *Acquired Intelligence and Adaptive Behaviour* module at the University of Sussex (2025).

---

## Full Report

A detailed technical report discussing the experimental design, analysis, and findings is available here:

[Download Full Report (PDF)](./AIAB_Report_RuairiBradley%20-%20279123.pdf)


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

## Visual Results

The following figures illustrate the performance, behaviour, and results from the comparative study of **Genetic Algorithms (GA)**, **Reinforcement Learning (RL)**, and a **hybrid GA+RL** model in Braitenberg light-seeking vehicles.

---

<p align="center">
  <img src="./images/AIAB%20braitenberg%20image.png" width="600"/>
</p>
<p align="center"><em>Figure 1 – Example of the Braitenberg vehicle setup used for testing autonomous light-following behaviour.</em></p>

---

<p align="center">
  <img src="./images/aiab%20performance%20over%20time%20example.png" width="600"/>
</p>
<p align="center"><em>Figure 2 – Comparative performance of Genetic Algorithm, Reinforcement Learning, and hybrid models over multiple training epochs.</em></p>

---

<p align="center">
  <img src="./images/aiab%20robot%20tests%20example.png" width="600"/>
</p>
<p align="center"><em>Figure 3 – Visualisation of robot test trajectories under each learning paradigm, showing path efficiency and light-seeking accuracy.</em></p>

---

<p align="center">
  <img src="./images/aiab%20success%20rates%20image.png" width="600"/>
</p>
<p align="center"><em>Figure 4 – Success rates for each approach averaged over several test environments. The hybrid GA+RL approach consistently achieved the highest success rate and stability.</em></p>


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


