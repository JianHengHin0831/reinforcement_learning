import numpy as np

# Simple environment: Random state and action spaces
state_dim = 3
action_dim = 2

# Simple random data for demonstration
states = np.random.rand(100, state_dim)
actions = np.random.rand(100, action_dim)
rewards = np.random.rand(100)
next_states = np.random.rand(100, state_dim)
dones = np.random.choice([True, False], size=100)

# Actor model
class Actor:
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.actor_parameters = np.random.rand(state_dim, action_dim)

    def predict(self, state):
        return self.action_bound * self.actor_parameters[state]

# Critic model
class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.critic_values = np.random.rand(state_dim)

# Actor-Critic
class ActorCritic:
    def __init__(self, state_dim, action_dim, action_bound):
        self.actor = Actor(state_dim, action_dim, action_bound)
        self.critic = Critic(state_dim)

    def train(self, state, action, reward, next_state, done):
        target = reward + self.critic.critic_values[next_state] if not done else reward
        td_error = target - self.critic.critic_values[state]
        self.actor.actor_parameters[state] += 0.1 * td_error  # Adjust learning rate
        self.critic.critic_values[state] += 0.1 * td_error  # Adjust learning rate

# Train the Actor-Critic agent
ac_agent = ActorCritic(state_dim, action_dim, 1)  # Action bound set to 1 for demonstration

num_episodes = 1000  # Increase number of episodes
for episode in range(num_episodes):
    state = np.random.randint(0, state_dim)
    episode_reward = 0

    while True:
        action = ac_agent.actor.predict(state)
        next_state = np.random.randint(0, state_dim)
        reward = rewards[np.random.randint(0, 100)]
        done = dones[np.random.randint(0, 100)]

        ac_agent.train(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            break

    print(f"Episode: {episode + 1}, Reward: {episode_reward}")
