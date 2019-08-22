import numpy as np
np.random.seed(0)
import gym
import matplotlib.pyplot as plt


class TileCoder:
    def __init__(self, layers, features):
        """
        Parameters
        - layers: int, the number of layers in tile coding
        - features: int, the number of features, also the shape of weights
        """
        self.layers = layers
        self.features = features
        self.codebook = {}

    def get_feature(self, codeword):
        if codeword in self.codebook:
            return self.codebook[codeword]
        count = len(self.codebook)
        if count >= self.features:  # collide when codebook is full
            return hash(codeword) % self.features
        else:
            self.codebook[codeword] = count
            return count

    def __call__(self, floats=(), ints=()):
        """
        Parameters
        - floats: tuple of floats, each of which is within [0., 1.]
        - ints: tuple of ints
        Returns
        - features : list of ints
        """
        dim = len(floats)
        scaled_floats = tuple(f * self.layers * self.layers for f in floats)
        features = []
        for layer in range(self.layers):
            codeword = (layer,) + tuple(int((f + (1 + dim * i) * layer) / self.layers) \
                                        for i, f in enumerate(scaled_floats)) + ints
            feature = self.get_feature(codeword)
            features.append(feature)
        return features


class SARSAAgent:
    def __init__(self, env, layers=8, features=2000, gamma=1.,
                 learning_rate=0.03, epsilon=0.001):
        self.action_n = env.action_space.n
        self.obs_low = env.observation_space.low
        self.obs_scale = env.observation_space.high - env.observation_space.low
        self.encoder = TileCoder(layers, features)
        self.w = np.zeros(features)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def encode(self, observation, action):
        states = tuple((observation - self.obs_low) / self.obs_scale)
        actions = (action,)
        return self.encoder(states, actions)

    def get_q(self, observation, action):
        features = self.encode(observation, action)
        return self.w[features].sum()

    def decide(self, observation):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        else:
            qs = [self.get_q(observation, action) for action in range(self.action_n)]
            return np.argmax(qs)

    def learn(self, observation, action, reward, observation_next, done, action_next=None):
        u = reward
        if not done:
            u += (self.gamma * self.get_q(observation_next, action_next))
        delta = u - self.get_q(observation, action)
        features = self.encode(observation, action)
        self.w[features] += (self.learning_rate * delta)


class SARSALambdaAgent(SARSAAgent):
    def __init__(self, env, layers=8, features=2000, gamma=1.,
                 learning_rate=0.03, epsilon=0.001, lambd=0.9):
        super().__init__(env=env, layers=layers, features=features,
                         gamma=gamma, learning_rate=learning_rate, epsilon=epsilon)
        self.lambd = lambd
        self.z = np.zeros(features)

    def learn(self, observation, action, reward, observation_next, done, action_next=None):
        u = reward
        if not done:
            u += (self.gamma * self.get_q(observation_next, action_next))
            self.z *= (self.gamma * self.lambd)
            features = self.encode(observation, action)
            self.z[features] = 1.  # replacement trace
        delta = u - self.get_q(observation, action)
        self.w += (self.learning_rate * delta * self.z)
        if done:
            self.z = np.zeros_like(self.z)

def play_sarsa(env, agent, train=False, render=False):
    episode_reward = 0
    observation = env.reset()
    action = agent.decide(observation)
    while True:
        if render:
            env.render()
        observation_next, reward, done, _ = env.step(action)
        episode_reward += reward
        if done:
            if train:
                agent.learn(observation, action, reward, observation_next, done)
            break
        action_next = agent.decide(observation_next)
        if train:
            agent.learn(observation, action, reward, observation_next, done, action_next)
        observation, action = observation_next, action_next
    return episode_reward


env = gym.make('MountainCar-v0')
env.seed(0)
env = gym.wrappers.Monitor(env, "./records", video_callable=lambda _: True)

agent = SARSALambdaAgent(env)

episodes = 75
episode_rewards = []
for episode in range(episodes):
    episode_reward = play_sarsa(env, agent, train=True)
    episode_rewards.append(episode_reward)

plt.plot(episode_rewards)


agent.epsilon = 0.
episodes = 100
episode_rewards = [play_sarsa(env, agent, train=False) for _ in range(episodes)]
print('average award = {} / {} = {}'.format(
        sum(episode_rewards), len(episode_rewards), np.mean(episode_rewards)))


poses = np.linspace(env.unwrapped.min_position, env.unwrapped.max_position, 256)
vels = np.linspace(-env.unwrapped.max_speed, env.unwrapped.max_speed, 256)
positions, velocities = np.meshgrid(poses, vels)

@np.vectorize
def decide(position, velocity):
    return agent.decide((position, velocity))

action_values = decide(positions, velocities)

fig, ax = plt.subplots()
c = ax.pcolormesh(positions, velocities, action_values)
ax.set_xlabel('position')
ax.set_ylabel('velocity')
fig.colorbar(c, ax=ax, boundaries=[-.5, .5, 1.5, 2.5], ticks=[0, 1, 2])
fig.show()

import pickle
with open('./resources/agent.pkl', 'wb') as file:
    pickle.dump(agent, file)


env.close()