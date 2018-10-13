import gym
import numpy as np



'''
Initialize constants in the simulation.
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
p         :
timesteps : a vector for storing the number of timesteps for each episode
'''
x = np.zeros((162,))
v = np.zeros((162,))
w = np.zeros((162,))
xbar = np.zeros((162,))
p = np.zeros((T+1,))
timesteps = np.zeros((100,))

'''
decode function will take in an observation and transform it into an index to update
x
'''
def decode(observation):
    x = observation[0]
    xdot = observation[1]
    theta = observation[2]
    thetadot = observation[3]

    regions = [0]*4

    if -2.4 <= x < -0.8:
        regions[0] = 0
    elif -0.8 <= x < 0.8:
        regions[0] = 1
    else:
        regions[0] = 2

    if xdot < -0.5:
        regions[1] = 0
    elif -0.5 <= xdot < 0.5:
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

    if thetadot < -0.872:
        regions[3] = 0
    elif -0.872 <= thetadot < 0.872:
        regions[3] = 1
    else:
        regions[3] = 2

    idx = (162//3)*regions[0] + (162//3//3)*regions[1] + (162//3//3//6)*regions[2] + (162//3//3//6//3)*regions[3]
    return idx

env = gym.make('CartPole-v1')
for i_episode in range(100):
    observation = env.reset()
    reward = 1
    old_idx = 0
    e = np.zeros((162,))
    for t in range(T):
        env.render()

        x = np.zeros((162,))

        # Decode the observation and update x
        idx = decode(observation)
        x[idx] = 1

        # Find the trace of x, p(t), and v(t+1)
        p[t+1] = np.dot(v, x)

        # Update reward
        reward_hat = g*p[t+1] - p[t]

        v[:] = v[:] + b*reward_hat*xbar[:]
        xbar[:] = l*xbar[:] + (1-l)*x[:]

        # Update weights
        w[:] = w[:] + a*reward_hat*e[:]

        # Determine the action to take
        noise = np.random.normal(mu, sigma, 1)
        action = 1 if np.dot(w, x) + noise >= 0 else 0

        # Take the action
        observation, reward, done, info = env.step(action)

        # Find the eligibility trace
        m = 1 if action == 1 else -1
        e[:] = d*e[:] + (1 - d)*m*x[:] # Barto 1983 uses a different action scheme than OpenAI

        # Check if it failed
        if done:

            # Update reward
            reward_hat = -1 - p[t] # Want to take away rewards if it fails

            v[:] = v[:] + b*reward_hat*xbar[:]
            xbar[:] = l*xbar[:] + (1-l)*x[:]

            # Update weights
            w[:] = w[:] + a*reward_hat*e[:]

            print("Episode completed at {} timestep".format(t+1))
            break

# Make plots for timestep per episode
