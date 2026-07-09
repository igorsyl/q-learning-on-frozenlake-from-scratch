"""
Q-Learning on FrozenLake from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_q_table
import numpy as np

def init_q_table(num_states, num_actions):
    """Return a zero-initialized Q-table of shape (num_states, num_actions)."""
    # build a 2D float64 numpy array of zeros sized by states and actions.
    return np.zeros((num_states, num_actions), dtype=np.float64)

# Step 2 - max_q_value
import numpy as np

def max_q_value(q_table, state):
    """Return the maximum Q value across all actions for the given state."""
    # index the row for `state` and return its maximum value
    return q_table[state, :].max()

# Step 3 - greedy_action
import numpy as np

def greedy_action(q_table, state):
    """Return the action index with the highest Q value at the given state."""
    # return argmax over the action axis for this state's Q values
    return np.argmax(q_table[state, :]).item()

# Step 4 - sample_random_action
def sample_random_action(action_space):
    # draw a uniformly random action from the given Gymnasium action space
    return action_space.sample().item()

# Step 5 - should_explore
def should_explore(epsilon, rng):
    """Return True with probability epsilon using the provided numpy Generator."""
    # draw a uniform sample from rng and compare it to epsilon
    return rng.uniform() < epsilon

# Step 6 - epsilon_greedy_action
import numpy as np

def epsilon_greedy_action(q_table, state, epsilon, action_space, rng):
    """Return an epsilon-greedy action for the given state."""
    # with prob epsilon explore via action_space, else take greedy action
    if should_explore(epsilon, rng):
        return sample_random_action(action_space)
    else:
        return greedy_action(q_table, state)

# Step 7 - decay_epsilon
def decay_epsilon(epsilon, decay_rate, min_epsilon):
    # return max(min_epsilon, epsilon * decay_rate)
    return max(min_epsilon, epsilon * decay_rate)

# Step 8 - td_target
def td_target(reward, gamma, q_table, next_state, done):
    # compute r + gamma * max_a Q(next_state, a), zeroing the bootstrap when done.
    if done:
        return reward
    else:
        return reward + gamma * max_q_value(q_table, next_state)

# Step 9 - td_error
def td_error(target, q_table, state, action):
    # return the TD error: target minus current Q(state, action)
    return target - q_table[state][action]

# Step 10 - q_learning_update
def q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma):
    # apply Q(s,a) += alpha * (target - Q(s,a)) in place and return the new Q value
    target = td_target(reward, gamma, q_table, next_state, done)
    error = td_error(target, q_table, state, action)
    update = alpha * error
    q_table[state][action] += update
    return q_table[state][action]

# Step 11 - interaction_step
def interaction_step(env, q_table, state, epsilon, alpha, gamma, rng):
    # select epsilon-greedy action, step env, apply Q-learning update, return (next_state, reward, done)
    action = epsilon_greedy_action(q_table, state, epsilon, env.action_space, rng)
    obs, reward, terminated, truncated, info = env.step(action)
    next_state = obs
    done = terminated or truncated
    q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma)
    return (next_state, float(reward), done)

# Step 12 - run_training_episode
def run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=200):
    # reset env, then repeatedly call interaction_step until done or max_steps, returning total reward.
    obs, info = env.reset()
    state = obs
    total_reward = 0
    for i in range(max_steps):
        (state, reward, done) = interaction_step(env, q_table, state, epsilon, alpha, gamma, rng)
        total_reward += reward
        if done: break
    return total_reward

# Step 13 - train_q_learning
import numpy as np

def train_q_learning(env, num_episodes, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995, seed=0, max_steps=200):
    # train a Q-learning agent for num_episodes; return (q_table, returns)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = init_q_table(num_states, num_actions)
    episode_returns = []
    epsilon = epsilon_start
    for episode in range(num_episodes):
        episode_return = run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=max_steps)
        episode_returns.append(episode_return)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    return (q_table, episode_returns)

# Step 14 - extract_greedy_policy
def extract_greedy_policy(q_table):
    # return a 1D int64 array mapping each state to its best (argmax) action.
    return np.argmax(q_table, 1)

# Step 15 - run_greedy_episode
def run_greedy_episode(env, policy, seed=None, max_steps=200):
    """Run one greedy episode and return True if the goal was reached."""
    # reset env, follow policy[state] each step, return bool(success)
    state, info = env.reset()
    for step in range(max_steps):
        action = policy[state]
        state, reward, terminated, truncated, info = env.step(action)
        success = reward == 1
        done = terminated or truncated
        if done: break

    return success

# Step 16 - evaluate_success_rate (not yet solved)
# TODO: implement

