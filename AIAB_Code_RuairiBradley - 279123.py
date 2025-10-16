# AIAB Coursework 
# Author: [Ruairi Bradley]
# Student ID: [279123]
# May 2025
# Description:
# This script contains the code for the 2 light trained models used in the experiments described in the report, with comments on how I tweaked for 5 and 10 light variations.
# including implementations for:
#   - Genetic Algorithm (GA)
#   - Q-Learning (QL)
#   - Hybrid GA + QL model
#   - Comparative analysis and result plotting
#
# Instructions:
# To execute a specific model in this one file set up:
# Comment out all other models but the one you want to run / tweak commented areas to align for more training lights. 
# Please bear in mind the plotting at the end wont run unless all the models have saved their results accoridngly. 
#
# Submission Format:
# This file has been formatted as a complete, standalone source script (.py) for submission,
# but In practive I actually had each model in separate files and saved results from each run for comparison plotting at the end once all models were run.







# ***** GA Model: *****

# Genetic Algorithm for Braitenberg Light-Seeking Robot
# simulates a robot evolving to find light more effectively using a Genetic Algorithm and randomised training light positions to promote generalisable behaviour
# By Ruairi Bradley
import numpy as np
from copy import deepcopy
import random
import matplotlib.pyplot as plt
import json

# vehicle set up given from spec
class Braitenberg:
    def __init__(self, initial_pos, initial_bearing, geno):
        self.geno = geno  
        self.initial_bearing = initial_bearing
        self.pos = initial_pos

    def get_geno(self):
        return self.geno

# sim environment 
class environment:
    def __init__(self, sig=0.1):
        self.dt = 0.05
        self.R = 0.05
        self.b = 45
        self.sig = sig # noise 

    def run(self, T, agent, motor_gain=0.5, show=True, S=(0, 0)):
        w_ll, w_lr, w_rl, w_rr, bl, br = agent.get_geno()

        sl_pos = np.zeros((2, 1))
        sr_pos = np.zeros((2, 1))
        S = np.array(S)
        sensor_gain = 0.5
        vl = vr = 0

        initial_bearing = agent.initial_bearing / 360 * 2 * np.pi
        b = self.b / 360 * 2 * np.pi

        steps = int(T / self.dt)
        pos = np.zeros((2, steps))
        bearing = np.zeros((1, steps))
        lightIntensity = np.zeros((2, steps))

        pos[:, 0] = agent.pos
        bearing[:, 0] = initial_bearing

        for i in range(1, steps):
            vc = (vl + vr) / 2
            va = (vr - vl) / (2 * self.R)

            pos[0, i] = pos[0, i - 1] + self.dt * vc * np.cos(bearing[0, i - 1])
            pos[1, i] = pos[1, i - 1] + self.dt * vc * np.sin(bearing[0, i - 1])
            bearing[0, i] = np.mod(bearing[0, i - 1] + self.dt * va, 2 * np.pi)

            sl_pos[0] = pos[0, i] + self.R * np.cos(bearing[0, i] + b)
            sl_pos[1] = pos[1, i] + self.R * np.sin(bearing[0, i] + b)
            sr_pos[0] = pos[0, i] + self.R * np.cos(bearing[0, i] - b)
            sr_pos[1] = pos[1, i] + self.R * np.sin(bearing[0, i] - b)

            dl = np.sqrt((sl_pos[0] - S[0])**2 + (sl_pos[1] - S[1])**2)
            dr = np.sqrt((sr_pos[0] - S[0])**2 + (sr_pos[1] - S[1])**2)

            il = sensor_gain / dl
            ir = sensor_gain / dr
            lightIntensity[0, i] = il
            lightIntensity[1, i] = ir

            lm = il * w_ll + ir * w_rl + bl + np.random.normal(0, self.sig)
            rm = il * w_lr + ir * w_rr + br + np.random.normal(0, self.sig)

            vl = motor_gain * lm
            vr = motor_gain * rm

        return pos, lightIntensity

def get_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

# Fitness func - reward being close largely as this the behaviour we want
# def evaluate_fitness(positions, intensities, lightsource=(0, 0)):
#     start = get_distance(positions[:, 0], lightsource)
#     end = get_distance(positions[:, -1], lightsource)
#     if end < 0.3:
#         return 1.0
#     return np.clip((start - end) / start, 0, 1)

# total dist moved closer
def fitness(positions, intensities, lightsource=(0, 0)):
    distances = [get_distance(positions[:, i], lightsource) for i in range(positions.shape[1])]
    start = distances[0]
    end = distances[-1]
    avg_distance = np.mean(distances)
    progress = np.maximum(start - avg_distance, 0)
    # big reward for getting close
    bonus = 1.0 if end < 0.5 else 0.0
    fitness = progress / start + bonus
    return np.clip(fitness, 0, 1.0)

def mutate(genotype, mean=0, standard_d=0.4, min_value=0, max_value=5):
    new_genotype = deepcopy(genotype)
    new_genotype += np.random.normal(mean, standard_d, size=genotype.shape)
    return np.clip(new_genotype, min_value, max_value)

# plot was way too jumpy to read trends 
def smooth_fitness(data, window_size=10):
    box = np.ones(window_size) / window_size
    return np.convolve(data, box, mode='same')

def random_light():
    # to generate random light position - predefined was leading to memorisation rather than generalising 
    return (random.uniform(-3, 3), random.uniform(-3, 3))


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    env = environment(sig=0.05)
    test_lights = [(-1.5, 1.5), (2.5, -1.5), (-2.5, 1), (1, 2.5), (3.5, 0)]  # 5 unseen lights
    pop_size = 50
    generations = 200 # 300 for 5 and 500 for 10 
    runtime = 20 # play w this see resutls / runtime tradeoff 


    # train le pop
    pop = [np.random.uniform(0, 1, 6) for _ in range(pop_size)]
    best_fit = []
    best_genome = None
    num_training_lights = 2 # change this to 5 and 10 for other tests

    for gen in range(generations):
        training_lights = [random_light() for _ in range(num_training_lights)]
        scores = []
        for individual in pop:
            total = 0
            for light in training_lights:
                # start_pos = (3,3)
                # start_bearing = 10
                # Attempting to randomise start position and bearing to improve generalisation - currently struggling need to sort this
                start_pos = (3 + random.uniform(-0.5, 0.5), 3 + random.uniform(-0.5, 0.5))
                ANGLES = [0, 45, 90, 135, 180, 225, 270, 315] # real robot tests mimic from these angle sets for start positions also** 
                start_bearing = random.choice(ANGLES)
                agent = Braitenberg(start_pos, start_bearing, individual)
                traj, intensities = env.run(runtime, agent, show=False, S=light)
                total += fitness(traj, intensities, light)
            scores.append(total / len(training_lights))
        sorted_idx = np.argsort(scores)[::-1]
        pop = [pop[i] for i in sorted_idx]
        scores = [scores[i] for i in sorted_idx]
        if best_genome is None or scores[0] > max(best_fit, default=0):
            best_genome = pop[0]
        best_fit.append(scores[0])

        new_pop = pop[:5]
        while len(new_pop) < pop_size:
            parent = random.choice(pop[:15])
            child = mutate(np.array(parent))
            new_pop.append(child)
        pop = new_pop 

        if gen % 10 == 0 or gen == generations - 1:
            print(f"Gen {gen} | Best: {scores[0]:.4f}")

    # save best genome for robot transfer code
    np.save("GAgeno2lights.npy", np.array(best_genome)) # need to run one more time wasnt saving as npy array YAY obvs change name for 5 and 10 lights
    # combine all my plots into 1 plot for report 
    np.save("fitness_2lights.npy", np.array(best_fit))  # or fitness_5lights.npy, fitness_10lights.npy

    # plot for this models (num of lights) individual performance
    # NOTE: low fitness != bad - can just be the environment is tough 
    plt.plot(best_fit, label='Raw', alpha=0.4)
    plt.plot(smooth_fitness(best_fit), label='Smoothed')
    plt.title("GA - 2 Light Trained Performance Over Time") # change title for other tests
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # unseen light tests
    total_success = 0
    total_runs = 0
    light_success_rates = []
    print("\nGeneralization Test on 5 Unseen Light Positions")

    for i, test_light in enumerate(test_lights):
        light_success = 0
        light_distances = []
        for trial in range(10):
            agent = Braitenberg((3, 3), 10, best_genome)
            _, final_dist, success = env.run(runtime, agent, show=False, S=test_light), get_distance((3, 3), test_light), False
            pos, _ = env.run(runtime, agent, show=False, S=test_light)
            final = get_distance(pos[:, -1], test_light)
            success = final < 0.5
            light_distances.append(final)
            if success:
                light_success += 1
            total_runs += 1
        total_success += light_success
        success_rate = light_success / 10
        light_success_rates.append(success_rate)

        print(f"\nTest Light {i+1}: {test_light}")
        print(f"  Success Rate: {light_success}/10 = {success_rate * 100:.1f}%")
        print(f"  Avg Final Distance: {np.mean(light_distances):.3f}")
        print(f"  Std Dev: {np.std(light_distances):.3f}")

    # success rate in terminal results 
    overall_success_rate = total_success / total_runs * 100
    print(f"\nOverall Success Rate Across All Unseen Lights") 
    print(f"Total Success: {total_success}/{total_runs}")
    print(f"Overall Success Rate: {overall_success_rate:.1f}%")
    # bar chart of success rate for this training depth
    labels = [str(light) for light in test_lights]
    plt.bar(labels, [r * 100 for r in light_success_rates], color='cornflowerblue')
    plt.ylim(0, 100)
    plt.ylabel("Success Rate (%)")
    plt.xlabel("Unseen Light Positions")
    plt.title("GA - 2 Light Trained Success Rate on Unseen Light Positions") # change title for other tests again
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    np.save("ga_success_2lights.npy", np.array(light_success_rates))  # mashing bar charts together for report write up copy this across each model and rename accorindly

# ***** End of GA *****













# ***** QL Model: *****

import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

def random_lights(n=2): # randomising light positions as with predefined the set ups were just memorisinsg light pos rather than learning behaviour
    return [(random.uniform(-3, 3), random.uniform(-3, 3)) for _ in range(n)]

num_training_lights = 2  # change to 5 or 10 for other tests
test_lights = [(-1.5, 1.5), (2.5, -1.5), (-2.5, 1), (1, 2.5), (3.5, 0)]  # Five unseen test positions
# experim settings
spawn_range = (2.5, 3.5)  # Where bots start relative to lights
angle_options = [0, 45, 90, 135, 180, 225, 270, 315]  # Possible starting angles

# same vehicle and envioronment set up
class Braitenberg:
    def __init__(self, pos, angle, weights):
        self.weights = weights  
        self.pos = pos  
        self.angle = np.radians(angle)  

class Environment:
    def __init__(self, noise=0.05):
        self.dt = 0.05  
        self.wheel_size = 0.05  
        self.sensor_angle = 45 
        self.noise = noise 
    
    def run_sim(self, time, bot, speed=0.5, light_pos=(0, 0)):
        w_ll, w_lr, w_rl, w_rr, bl, br = bot.weights
        
        left_sensor = np.zeros(2)
        right_sensor = np.zeros(2)
        light = np.array(light_pos)
        
        left_speed = 0
        right_speed = 0
        sensor_angle = np.radians(self.sensor_angle)
        
        steps = int(time / self.dt)
        positions = np.zeros((2, steps))
        angles = np.zeros(steps)
        sensor_readings = np.zeros((2, steps))
        
        positions[:, 0] = bot.pos
        angles[0] = bot.angle
        
        for i in range(1, steps):
            forward_speed = (left_speed + right_speed) / 2
            turn_speed = (right_speed - left_speed) / (2 * self.wheel_size)
            
            positions[0, i] = positions[0, i-1] + self.dt * forward_speed * np.cos(angles[i-1])
            positions[1, i] = positions[1, i-1] + self.dt * forward_speed * np.sin(angles[i-1])
            angles[i] = (angles[i-1] + self.dt * turn_speed) % (2 * np.pi)
            
            left_sensor[0] = positions[0, i] + self.wheel_size * np.cos(angles[i] + sensor_angle)
            left_sensor[1] = positions[1, i] + self.wheel_size * np.sin(angles[i] + sensor_angle)
            right_sensor[0] = positions[0, i] + self.wheel_size * np.cos(angles[i] - sensor_angle)
            right_sensor[1] = positions[1, i] + self.wheel_size * np.sin(angles[i] - sensor_angle)
            
            left_dist = np.sqrt((left_sensor[0]-light[0])**2 + (left_sensor[1]-light[1])**2)
            right_dist = np.sqrt((right_sensor[0]-light[0])**2 + (right_sensor[1]-light[1])**2)
            
            left_light = 0.5 / (left_dist + 0.001)
            right_light = 0.5 / (right_dist + 0.001)
            
            sensor_readings[0, i] = left_light
            sensor_readings[1, i] = right_light
            
            left_motor = left_light*w_ll + right_light*w_rl + bl + np.random.normal(0, self.noise)
            right_motor = left_light*w_lr + right_light*w_rr + br + np.random.normal(0, self.noise)
            
            left_speed = speed * left_motor
            right_speed = speed * right_motor
        
        bot.pos = positions[:, -1]
        bot.angle = angles[-1]
        
        return positions, sensor_readings

# my stab 
class QLearner:
    def __init__(self):
        # tweaked these weights after watching some runs get stuck circling lights
        # Adding soft left/right preference seemed to help stability early in training
        self.designs = [
            [0.8, 0.8, 0.2, 0.2, 0.5, 0.5],  # balanced
            [1.0, 0.2, 0.8, 0.0, 0.3, 0.7],   # soft left
            [0.2, 1.0, 0.0, 0.8, 0.7, 0.3],   # soft right
            [1.0, 0.0, 1.0, 0.0, 0.1, 0.9],   # hard left
            [0.0, 1.0, 0.0, 1.0, 0.9, 0.1]    # RHard right
        ]
        
        # initialize Q-table (10x10 grid for states)
        # init Q-table with noise to encourage diversity in early decisions
        # tried zeros first but it led to local optima – randomness worked better
        self.q_table = np.random.uniform(-1, 1, (10, 10, len(self.designs)))
        
        # learning settings
        self.learn_rate = 0.3
        self.discount = 0.95
        self.explore = 1.0       # start exploring a lot
        self.explore_decay = 0.998
        self.min_explore = 0.05  # ut always explore a little
    
    def get_state(self, left, right):
        # simplify sensor readings into a grid state
        diff = right - left
        total = right + left + 0.001  # Avoid divide by zero error thrown
        # normalize and bin the difference
        norm_diff = np.clip(diff / total, -1, 1)
        diff_bin = min(int((norm_diff + 1) * 5), 9)
        # bin the total intensity
        total_bin = min(int(total), 9)
        return diff_bin, total_bin
    
    def choose_design(self, state):
        # initially had explore decay too fast – agent got stuck 
        # slowed decay to give more variety early on
        # sometimes explore randomly
        if random.random() < self.explore:
            return random.randint(0, len(self.designs)-1)
        # otherwise pick the best known design
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, new_state):
        # standard Q-learning update
        best_future = np.max(self.q_table[new_state])
        self.q_table[state][action] += self.learn_rate * (
            reward + self.discount * best_future - self.q_table[state][action]
        )
    
    def update_settings(self):
        # gradually reduce exploration
        self.explore = max(self.explore * self.explore_decay, self.min_explore)
        # and learning rate too
        self.learn_rate = max(self.learn_rate * 0.999, 0.1)
    
    def get_weights(self, index):
        return self.designs[index]

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def run_experiment():
    # setup everything
    world = Environment()
    learner = QLearner()
    sim_time = 15  # seconds per trial - trials seem this is long enough for good convergence
    rewards = []
    best_q = None
    best_avg = -999999
    recent = deque(maxlen=100)

    print("Training the vehicles...")
    for episode in range(10000): # 20000 for 5 and 10 (more complex enviro)
        total = 0
        train_lights = random_lights(n=num_training_lights)
        
        # train on each light position
        for light in train_lights:
            # random start position and angle
            start = (random.uniform(*spawn_range), random.uniform(*spawn_range))
            angle = random.choice(angle_options)
            bot = Braitenberg(start, angle, learner.get_weights(0))
            # first observation
            pos, senses = world.run_sim(sim_time/15, bot, light_pos=light)
            state = learner.get_state(*senses[:, -1])
            success = False
            # run for 30 control steps max
            for _ in range(30):
                prev_pos = bot.pos.copy()
                # choose and apply a design
                action = learner.choose_design(state)
                bot.weights = learner.get_weights(action)
                # run the simulation
                pos, senses = world.run_sim(sim_time/15, bot, light_pos=light)
                new_state = learner.get_state(*senses[:, -1])
                # reward is movement toward light, bonus if reached
                # reward shaping: tested linear vs. squared reward vs. step bonus
                # this combo gave fastest convergence
                dist_change = distance(prev_pos, light) - distance(bot.pos, light)
                reward = dist_change * 2
                if distance(bot.pos, light) < 0.5:
                    reward += 5
                    # reward += 10 - caused agents to rush and spin mostly so reduced
                    success = True
                # Learn from this experience
                learner.learn(state, action, reward, new_state)
                state = new_state
                total += reward
                if success:
                    break
            recent.append(1 if success else 0)
        # track performance
        rewards.append(total / len(train_lights))
        learner.update_settings()
        # save the best version
        if len(recent) == 100: # 100-episode rolling window after seeing too much noise with 50
            avg = np.mean(rewards[-100:])
            if avg > best_avg:
                best_avg = avg
                best_q = np.copy(learner.q_table)
        
        # Progress
        if episode % 100 == 0:
            print(f"Episode {episode} | Avg reward: {np.mean(rewards[-100:]):.2f} | "
                  f"Exploring: {learner.explore:.2f}")

    print("\nTesting on new light positions...")
    if best_q is not None:
        learner.q_table = best_q

    results = []
    labels = []
    for i, light in enumerate(test_lights):
        count = 0
        labels.append(f"Test {i+1}\n{light}")
        # try 10 times for each test light
        # keeping same testing structure as training but using best-known Q-table
        # no exploration during test 
        for _ in range(10):
            start = (random.uniform(*spawn_range), random.uniform(*spawn_range))
            angle = random.choice(angle_options)
            bot = Braitenberg(start, angle, learner.get_weights(0))
            for _ in range(30):
                pos, senses = world.run_sim(sim_time/15, bot, light_pos=light)
                s = learner.get_state(*senses[:, -1])
                a = np.argmax(learner.q_table[s])
                bot.weights = learner.get_weights(a)
                if distance(bot.pos, light) < 0.5:
                    count += 1
                    break
        rate = count / 10 * 100
        results.append(rate)
        print(f"Light {i+1} @ {light} → Success: {rate:.1f}%")

    print(f"\nOverall success rate: {np.mean(results):.1f}%")
    np.save("Qtable2Lights.npy", learner.q_table) # change the save for 5 and 10
    return rewards, results, labels

def smooth(y, window=50):
    return np.convolve(y, np.ones(window)/window, mode='valid')

# specific model plots - change titles for different training conditions
#reward plot 
def plot_results(rewards, success_rates, labels):
    plt.figure(figsize=(10, 4))
    plt.plot(rewards, alpha=0.3, label='Raw')
    plt.plot(smooth(rewards), label='Smoothed')
    plt.title("QL - 2 Training Light Performance Over Time") # change graph titles for 5 and 10
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
# success % plot
    plt.figure(figsize=(8, 4))
    bars = plt.bar(range(len(success_rates)), success_rates)
    plt.xticks(range(len(success_rates)), labels, rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.ylabel("Success Rate (%)")
    plt.title("QL - 2 Light Trained Success Rate on Unseen Light Positions") # change again for 5 and 10
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}%', 
                ha='center', va='bottom')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    reward_data, test_results, test_labels = run_experiment()
    # save to combine the plots later on 
    np.save("reward_2lights.npy", np.array(reward_data)) # alter for 5 and 10
    # combine bar charts save
    np.save("ql_success_2lights.npy", np.array(test_results)) # same for this

    plot_results(reward_data, test_results, test_labels)

# ***** End of QL ***** 












# ***** Hybrid Model: *****

# Hybrid GA Q-Learning agent to try max the light seeking behaviour
import numpy as np 
import random
import matplotlib.pyplot as plt
# TODO: 

class Braitenberg:
    def __init__(self, pos, angle, weights):
        self.weights = weights 
        self.pos = pos  
        self.angle = angle * np.pi / 180 
        
class Environment:
    def __init__(self, noise=0.05): 
        self.time_step = 0.05  
        self.robot_size = 0.05  
        self.sensor_spread = 45  
        self.noise_level = noise  
        
    def run(self, duration, robot, light_pos):
        weights = robot.weights 
        rad_angle = self.sensor_spread * np.pi / 180  
        steps = int(duration / self.time_step)  
        
        positions = np.zeros((2, steps))
        headings = np.zeros(steps)
        sensor_readings = np.zeros((2, steps))
        
        positions[:, 0] = robot.pos
        headings[0] = robot.angle
        left_v = right_v = 0  
        

        for t in range(1, steps): 
            avg_speed = (left_v + right_v) / 2
            turn_rate = (right_v - left_v) / (2 * self.robot_size) 
            
            positions[0, t] = positions[0, t-1] + self.time_step * avg_speed * np.cos(headings[t-1])
            positions[1, t] = positions[1, t-1] + self.time_step * avg_speed * np.sin(headings[t-1])
            headings[t] = (headings[t-1] + self.time_step * turn_rate) % (2 * np.pi)
            
            left_sensor = [
                positions[0, t] + self.robot_size * np.cos(headings[t] + rad_angle),
                positions[1, t] + self.robot_size * np.sin(headings[t] + rad_angle)
            ]
            right_sensor = [
                positions[0, t] + self.robot_size * np.cos(headings[t] - rad_angle),
                positions[1, t] + self.robot_size * np.sin(headings[t] - rad_angle)
            ]
            
            left_dist = np.sqrt((left_sensor[0]-light_pos[0])**2 + (left_sensor[1]-light_pos[1])**2)
            right_dist = np.sqrt((right_sensor[0]-light_pos[0])**2 + (right_sensor[1]-light_pos[1])**2)
            
            left_reading = 0.5 / (left_dist + 0.001) 
            right_reading = 0.5 / (right_dist + 0.001)
            sensor_readings[:, t] = [left_reading, right_reading]
            
            left_motor = (left_reading * weights[0] + right_reading * weights[2] + weights[4] + 
                         random.gauss(0, self.noise_level))
            right_motor = (left_reading * weights[1] + right_reading * weights[3] + weights[5] + 
                          random.gauss(0, self.noise_level))

            left_v = 0.5 * left_motor  
            right_v = 0.5 * right_motor
        
        robot.pos = positions[:, -1]
        robot.angle = headings[-1]
        return positions, sensor_readings
    





# Q-learning controller with GA elements
class QLearner:
    def __init__(self):
        # hand-tuned presets based on trial and error from QL model
        self.configs = [
            [0.8, 0.8, 0.2, 0.2, 0.5, 0.5],  # default balanced
            [1.0, 0.2, 0.8, 0.0, 0.3, 0.7],  # soft left
            [0.2, 1.0, 0.0, 0.8, 0.7, 0.3],  # soft right
            [1.0, 0.0, 1.0, 0.0, 0.1, 0.9],  # hard left
            [0.0, 1.0, 0.0, 1.0, 0.9, 0.1],  # hard right
        ]
        # Q-table - 10x10 states x 5 actions
        self.q_values = np.zeros((10, 10, len(self.configs)))
        # TODO: Maybe make state space configurable
    
    # state discretization - could be improved
    def _get_state(self, left, right):
        total = left + right
        diff = right - left
        if total > 0:
            norm_diff = diff / total  # normalized difference
        else:
            norm_diff = 0  # when no light detected
        # bin into 10 states each
        diff_idx = min(int((norm_diff + 1) * 5), 9)  
        total_idx = min(int(total), 9) 
        return diff_idx, total_idx
    
    # simple epsilon-greedy
    def pick_action(self, state):
        return np.argmax(self.q_values[state])

    
    # apply action to robot
    def do_action(self, robot, state):
        action = self.pick_action(state)
        robot.weights = self.configs[action]  # set new weights
        return action
    
    # standard Q-learning update
    def update(self, state, action, reward, next_state, 
              learning_rate=0.3, discount=0.95):  # standard params
        best_next = np.max(self.q_values[next_state])
        td_error = reward + discount * best_next - self.q_values[state][action]
        self.q_values[state][action] += learning_rate * td_error
    
    # for GA - creates mutated copy
    def make_copy(self):
        new_controller = QLearner()
        # noise - maybe should tune this
        new_controller.q_values = self.q_values.copy() + np.random.normal(0, 0.5, self.q_values.shape)
        return new_controller



# main training function - believe its working
def train_robots(pop_size=20, generations=200, num_lights=2):
    world = Environment()
    angles = [0, 45, 90, 135, 180, 225, 270, 315]  #same discrete set across all tests sim and real for start angle
    start_area = (2.5, 3.5)  # start in top-right area range around 3,3

    # initialize population
    controllers = [QLearner() for _ in range(pop_size)]
    performance = []

    # training loop
    for gen in range(generations):
        scores = []
        for controller in controllers:
            total_score = 0
            # random light positions - same as other experiments
            lights = [(random.uniform(-3, 3), random.uniform(-3, 3)) 
                     for _ in range(num_lights)]
            for light in lights:
                # random start config
                start = (random.uniform(*start_area), random.uniform(*start_area))
                angle = random.choice(angles)
                robot = Braitenberg(start, angle, controller.configs[0])  # start with default
                # initial step
                pos, readings = world.run(1.0, robot, light)
                state = controller._get_state(*readings[:, -1])
                # learning steps
                for step in range(30):  # 30 steps per light
                    action = controller.do_action(robot, state)
                    old_pos = robot.pos.copy()  # for reward calc
                    pos, readings = world.run(1.0, robot, light)
                    new_state = controller._get_state(*readings[:, -1])
                    # reward calculation - simple distance improvement
                    prev_dist = np.sqrt((old_pos[0]-light[0])**2 + (old_pos[1]-light[1])**2)
                    new_dist = np.sqrt((robot.pos[0]-light[0])**2 + (robot.pos[1]-light[1])**2)
                    reward = (prev_dist - new_dist) * 2  # scale factor arbitrary
                    # bonus for reaching light
                    if new_dist < 0.5:
                        reward += 5  # big reward
                    # update Q-values
                    controller.update(state, action, reward, new_state)
                    state = new_state
                    total_score += reward
            scores.append(total_score / num_lights)  # average per light
        performance.append(max(scores))

        # evolutionary selection
        ranked = np.argsort(scores)[::-1]  # sort descending
        elite = [controllers[i] for i in ranked[:5]]  # keep top 5
        next_gen = elite.copy()

        # create offspring - simple mutation
        while len(next_gen) < pop_size:
            parent = random.choice(elite)
            child = parent.make_copy()
            next_gen.append(child)
        controllers = next_gen
        print(f"Gen {gen+1} best: {scores[ranked[0]]:.2f}")  # progress

    return controllers[ranked[0]], performance  # return best


# testing func
def run_tests(best_controller):
    world = Environment()
    test_lights = [(-1.5, 1.5), (2.5, -1.5), (-2.5, 1), (1, 2.5), (3.5, 0)]  # fixed pos unseens
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    start_area = (2.5, 3.5)
    results = []
    
    for light in test_lights:
        successes = 0
        for trial in range(10):  # 10 trials per light
            start = (random.uniform(*start_area), random.uniform(*start_area))
            angle = random.choice(angles)
            robot = Braitenberg(start, angle, best_controller.configs[0])
            for step in range(30):  # max 30 steps
                pos, readings = world.run(1.0, robot, light)
                state = best_controller._get_state(*readings[:, -1])
                action = best_controller.pick_action(state)
                robot.weights = best_controller.configs[action]
                # check if reached light
                if np.sqrt((robot.pos[0]-light[0])**2 + (robot.pos[1]-light[1])**2) < 0.5:
                    successes += 1
                    break
        results.append(successes / 10)  # success rate
    
    return results

# for the plot aesthetics 
def smooth_plot(data, window=50):
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')

# main Executio
if __name__ == "__main__":
    # fixed seeds - same as other tests
    random.seed(42)
    np.random.seed(42)

    print("Training hybrid GA-Q controller...")
    best_ctrl, rewards = train_robots(num_lights=2, generations=200) # change this to 5 or 10 for other tests and 200 gens for 5 and 150 for 10

    np.save("hybrid_reward_2lights.npy", np.array(rewards)) # also change this for 5 and 10

    print("\nTesting on unseen positions...")
    success_rates = run_tests(best_ctrl)

    np.save("hybrid_success_2lights.npy", np.array(success_rates)) # change name for 5 and 10 

    # reward plot
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Raw")
    plt.plot(smooth_plot(rewards), label="Smoothed")
    plt.title("Hybrid GA+QL - 2 Lights Performance Over Time") # change title for 5 and 10 lights
    plt.xlabel("Generation")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # success % plot
    plt.figure(figsize=(8, 5))
    labels = [str(pos) for pos in [(-1.5, 1.5), (2.5, -1.5), (-2.5, 1), (1, 2.5), (3.5, 0)]]
    plt.bar(labels, np.array(success_rates) * 100)
    plt.ylim(0, 100)
    plt.ylabel("Success Rate (%)")
    plt.title("Hybrid 2 Light - Success on Unseen Light Positions") # alter for differing depths 5 and 10
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    print(f"\nOverall Success Rate: {np.mean(success_rates)*100:.1f}%")
    # TODO: Add more analysis later standard deviation? etc
    # IDEA: Maybe visualize some trajectories - forcompare to the real?

# ***** End of hybrid *****










# ***** Combined plotting for report analysis: *****
# Must have all models trained and saved .npy files before this section will run 

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

def smooth(data, window=50):
    box = np.ones(window) / window
    return np.convolve(data, box, mode='valid')

# Load fitness/reward data for GA, QL, and Hybrid
fitness_2 = np.load("fitness_2lights.npy")
fitness_5 = np.load("fitness_5lights.npy")
fitness_10 = np.load("fitness_10lights.npy")
reward_2 = np.load("reward_2lights.npy")
reward_5 = np.load("reward_5lights.npy")
reward_10 = np.load("reward_10lights.npy")
hyb_2 = np.load("hybrid_reward_2lights.npy")
hyb_5 = np.load("hybrid_reward_5lights.npy")
hyb_10 = np.load("hybrid_reward_10lights.npy")

fitness_2_sm = smooth(fitness_2)
fitness_5_sm = smooth(fitness_5)
fitness_10_sm = smooth(fitness_10)
reward_2_sm = smooth(reward_2, window=100)
reward_5_sm = smooth(reward_5, window=100)
reward_10_sm = smooth(reward_10, window=100)
hyb_2_sm = smooth(hyb_2, window=10)
hyb_5_sm = smooth(hyb_5, window=10)
hyb_10_sm = smooth(hyb_10, window=10)

# This produces Figure 3: GA performance over generations 
plt.figure(figsize=(10, 5))
plt.plot(fitness_2_sm, label="GA - 2 Lights")
plt.plot(fitness_5_sm, label="GA - 5 Lights")
plt.plot(fitness_10_sm, label="GA - 10 Lights")
plt.title("Genetic Algorithm Agent Performance Over Time")
plt.xlabel("Generation")
plt.ylabel("Fitness (Smoothed)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# This plots Figure 4: Q-Learning agent reward progression over episodes
plt.figure(figsize=(10, 5))
plt.plot(reward_2_sm, label="QL - 2 Lights")
plt.plot(reward_5_sm, label="QL - 5 Lights")
plt.plot(reward_10_sm, label="QL - 10 Lights")
plt.title("Q-Learning Agent Reward Over Time")
plt.xlabel("Episode")
plt.ylabel("Reward (Smoothed)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# This plot produces: Figure 2: Hybrid GA+QL reward over generations
plt.figure(figsize=(10, 5))
plt.plot(hyb_2_sm, label="Hybrid - 2 Lights")
plt.plot(hyb_5_sm, label="Hybrid - 5 Lights")
plt.plot(hyb_10_sm, label="Hybrid - 10 Lights")
plt.title("Hybrid GA+QL Agent Reward Over Time")
plt.xlabel("Generation")
plt.ylabel("Reward (Smoothed)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Load success data and convert to percentage for Figure 5 Bar Chart plot
ga_2_all = np.load("ga_success_2lights.npy") * 100
ga_5_all = np.load("ga_success_5lights.npy") * 100
ga_10_all = np.load("ga_success_10lights.npy") * 100

ql_2_all = np.load("ql_success_2lights.npy") 
ql_5_all = np.load("ql_success_5lights.npy") 
ql_10_all = np.load("ql_success_10lights.npy") 

hyb_2_all = np.load("hybrid_success_2lights.npy") * 100
hyb_5_all = np.load("hybrid_success_5lights.npy") * 100
hyb_10_all = np.load("hybrid_success_10lights.npy") * 100

# The produces the information for Table 5: Paired T-tests (GA vs QL, GA vs Hybrid, QL vs Hybrid)
print("\nPaired t-tests on per-light success rates:")
print("GA vs QL:")
print(f"  2 Lights: t = {ttest_rel(ql_2_all, ga_2_all).statistic:.3f}, p = {ttest_rel(ql_2_all, ga_2_all).pvalue:.4f}")
print(f"  5 Lights: t = {ttest_rel(ql_5_all, ga_5_all).statistic:.3f}, p = {ttest_rel(ql_5_all, ga_5_all).pvalue:.4f}")
print(f" 10 Lights: t = {ttest_rel(ql_10_all, ga_10_all).statistic:.3f}, p = {ttest_rel(ql_10_all, ga_10_all).pvalue:.4f}")
print("GA vs Hybrid:")
print(f"  2 Lights: t = {ttest_rel(hyb_2_all, ga_2_all).statistic:.3f}, p = {ttest_rel(hyb_2_all, ga_2_all).pvalue:.4f}")
print(f"  5 Lights: t = {ttest_rel(hyb_5_all, ga_5_all).statistic:.3f}, p = {ttest_rel(hyb_5_all, ga_5_all).pvalue:.4f}")
print(f" 10 Lights: t = {ttest_rel(hyb_10_all, ga_10_all).statistic:.3f}, p = {ttest_rel(hyb_10_all, ga_10_all).pvalue:.4f}")
print("QL vs Hybrid:")
print(f"  2 Lights: t = {ttest_rel(hyb_2_all, ql_2_all).statistic:.3f}, p = {ttest_rel(hyb_2_all, ql_2_all).pvalue:.4f}")
print(f"  5 Lights: t = {ttest_rel(hyb_5_all, ql_5_all).statistic:.3f}, p = {ttest_rel(hyb_5_all, ql_5_all).pvalue:.4f}")
print(f" 10 Lights: t = {ttest_rel(hyb_10_all, ql_10_all).statistic:.3f}, p = {ttest_rel(hyb_10_all, ql_10_all).pvalue:.4f}")

# Bar Chart of Overall Success averages for Figure 5
ga_means = [np.mean(ga_2_all), np.mean(ga_5_all), np.mean(ga_10_all)]
ql_means = [np.mean(ql_2_all), np.mean(ql_5_all), np.mean(ql_10_all)]
hyb_means = [np.mean(hyb_2_all), np.mean(hyb_5_all), np.mean(hyb_10_all)]

labels = ['2 Lights', '5 Lights', '10 Lights']
x = np.arange(len(labels))
width = 0.25

# produces the plot for Figure 5: Success rate comparison across agent types and training setups
plt.figure(figsize=(10, 5))
plt.bar(x - width, ga_means, width=width, label="GA", color='dodgerblue')
plt.bar(x, ql_means, width=width, label="QL", color='tomato')
plt.bar(x + width, hyb_means, width=width, label="Hybrid", color='mediumseagreen')
plt.xticks(x, labels)
plt.ylim(0, 100)
plt.ylabel("Success Rate (%)")
plt.title("Success Rate Across Agent Types (Unseen Test Lights)")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()
