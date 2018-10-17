'''
Author: Jack Geissinger
Date: October 12, 2018

Reference: A. G. Barto, R. S. Sutton, and C. W. Anderson, “Neuronlike adaptive elements that
can solve difficult learning control problems,”
IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-13, pp. 834–846,
Sept./Oct. 1983.
'''

import gym
import numpy as np

'''
Initialize constants in the simulation using the parameters specified in Barto 1983.
a - alpha     : learning rate
d - delta     : trace decay rate in the eligibility trace
g - gamma     : discount factor (extinction of observations if no external reinforcement)
l - lambda    : trace decay rate in the trace of x[i]
b - beta      : rate of change of v[i]
T - timesteps : number of timesteps the simulation should run for
'''
a = 1000
d = 0.9
g, b = 0.95, 0.5
l = 0.8
T = 500

'''
Initialize parameters for noise signal
'''
mu, sigma = 0, 0.01 # mean and standard deviation

'''
Initialize the matrices in the simulation.
x         : a 162x1 representation of our decoded observations at each timestep
v         : a 162x1 matrix to contain our updating rule in the ACE
w         : a 162x1 matrix containing weights in the ASE
e         : a 162x1 matrix representing the eligibility trace of each channel in the decoder
xbar      : a 162x1 matrix that represents the trace of x
p         : a prediction of eventual reward at each timestep
timesteps : a vector for storing the number of timesteps for each episode
'''
x = np.zeros((162,))
v = np.zeros((162,))
w = np.zeros((162,))
xbar = np.zeros((162,))
p = np.zeros((T+1,))
timesteps = np.zeros((100,))

'''
decode function will take in an 1x4 observation and transform it into an index to modify x.

We are breaking up position, velocity, angle, and angular velocity into regions
based on Barto 1983, and using the regions that each value is in to toggle a single
value in the 162x1 vector representing x above. Position, velocity, angle and angular
velocity have 3, 3, 6, and 3 regions, respectively. There is 3x3x6x3 = 162 options total.

See Barto 1983 for more information.
'''
def decode(observation):
    pos = observation[0]
    vel = observation[1]
    theta = observation[2]
    omega = observation[3]

    regions = [0]*4

    if -2.4 <= pos < -0.8:
        regions[0] = 0
    elif -0.8 <= pos < 0.8:
        regions[0] = 1
    else:
        regions[0] = 2

    if vel < -0.5:
        regions[1] = 0
    elif -0.5 <= vel < 0.5:
        regions[1] = 1
    else:
        regions[1] = 2

    if -0.20943951 <= theta < -0.10472:
        regions[2] = 0
    elif -0.10472 <= theta < -0.017:
        regions[2] = 1
    elif -0.017 <= theta < 0:
        regions[2] = 2
    elif 0 <= theta < 0.017:
        regions[2] = 3
    elif 0.017 <= theta < 0.10472:
        regions[2] = 4
    else:
        regions[2] = 5

    if omega < -0.872:
        regions[3] = 0
    elif -0.872 <= omega < 0.872:
        regions[3] = 1
    else:
        regions[3] = 2

    idx = (162//3)*regions[0] + (162//3//3)*regions[1] + (162//3//3//6)*regions[2] + (162//3//3//6//3)*regions[3]
    return idx

'''
Simulation with OpenAI's gym CartPole-v1 environment.

Notes: 1. OpenAI uses +1 as a reward per successful timestep, Barto 1983 used 0
for each successful timestep and -1 for failure. 2. The actions available are 1 and 0 in OpenAI,
and 1 and -1 in Barto 1983.
'''

env = gym.make('CartPole-v1')
for i_episode in range(100):
    observation = env.reset()
    e = np.zeros((162,))
    for t in range(T):
        env.render()

        # Decode the observation and update x
        x = np.zeros((162,))
        idx = decode(observation)
        x[idx] = 1

        # Find p(t)
        p[t+1] = np.dot(v, x)

        # Update reward
        reward_hat = g*p[t+1] - p[t]

        # Modify the updating rule, v, and update the trace of x
        v[:] = v[:] + b*reward_hat*xbar[:]
        xbar[:] = l*xbar[:] + (1-l)*x[:]

        # Update the weights
        w[:] = w[:] + a*reward_hat*e[:]

        # Determine the action to take
        noise = np.random.normal(mu, sigma, 1)
        action = 1 if np.dot(w, x) + noise >= 0 else 0

        # Take the action
        observation, reward, done, info = env.step(action)

        # Find the eligibility trace
        m = 1 if action == 1 else -1 # Barto 1983 uses a different action scheme than OpenAI, so this was necessary
        e[:] = d*e[:] + (1 - d)*m*x[:]

        # Check if it failed
        if done:

            # Update reward for failing
            reward_hat = -1 - p[t] # Want to take away rewards if it fails

            v[:] = v[:] + b*reward_hat*xbar[:]
            xbar[:] = l*xbar[:] + (1-l)*x[:]

            # Update weights since it failed
            w[:] = w[:] + a*reward_hat*e[:]

            print("Episode completed at {} timestep".format(t+1))
            break

# TODO: Make plots for timestep vs episode to visualize improvement
