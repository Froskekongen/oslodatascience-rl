import gym
import numpy as np
import glob
import pickle
from random import randint

def downsample(image):
    # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
    return image[::2, ::2, :]

def remove_color(image):
    """Convert all color (RGB is the third dimension in the image)"""
    return image[:, :, 0]

def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = input_observation
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1 # everything else (paddles, ball) just set to 1
    # Convert from 80 x 80 matrix to 6400 x 1 matrix

    processed_observation = processed_observation[25:195,]

    processed_observation = processed_observation.astype(np.float).ravel()

    # subtract the previous frame from the current one so we are only processing on changes in the game
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    # store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation
    return input_observation, prev_processed_observations


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2

def relu(vector):
    vector[vector < 0] = 0
    return vector

def apply_neural_nets(observation_matrix, weights):
    """ Based on the observation_matrix and weights, compute the new hidden layer values and the new output layer values"""
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = np.tanh(hidden_layer_values)
    hidden_layer_values = np.dot(np.array(hidden_layer_values)[np.newaxis], weights['2'])
    hidden_layer_values = np.tanh(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['3'])
    output_layer_values = softmax(output_layer_values)
    return hidden_layer_values, output_layer_values

def choose_action(random_action, probability):
    random_value = randint(4, 5) #np.random.uniform()

    if (random_action):
        return random_value
    else:
        if np.argmax(probability) == 1:
            return 4
        else:
            return 5


    # if random_value < probability:
    #     # signifies up in openai gym
    #     return 2
    # else:
    #     # signifies down in openai gym
    #     return 3

def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
    """ See here: http://neuralnetworksanddeeplearning.com/chap2.html"""
    delta_L = gradient_log_p
    dC_dw3 = np.dot(hidden_layer_values.T, delta_L)
    delta_l2 = np.dot(dC_dw3.T, weights['2'].T)
    delta_l2 = np.tanh(delta_l2)
    dC_dw2 = np.dot(delta_l2.T, observation_values)
    delta_l3 = np.dot(dC_dw2.T, weights['2'].T)
    delta_l3 = np.tanh(delta_l3)
    dC_dw1 = np.dot(delta_l3.T, observation_values)
    return {
        '1': dC_dw1,
        '2': dC_dw2,
        '3': dC_dw3
    }

def update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
    """ See here: http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop"""
    epsilon = 1e-5
    for layer_name in weights.keys():
        g = g_dict[layer_name]
        expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g**2
        weights[layer_name] += (learning_rate * g)/(np.sqrt(expectation_g_squared[layer_name] + epsilon))
        g_dict[layer_name] = np.zeros_like(weights[layer_name]) # reset batch gradient buffer

def discount_rewards(rewards, gamma):
    """ Actions you took 20 steps before the end result are less important to the overall result than an action you took a step ago.
    This implements that logic by discounting the reward on previous actions based on how long ago they were taken"""
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

def discount_with_rewards(gradient_log_p, episode_rewards, gamma):
    """ discount the gradient with the normalized rewards """
    discounted_episode_rewards = discount_rewards(episode_rewards, gamma)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return gradient_log_p * discounted_episode_rewards


def main():
    env = gym.make("SpaceInvaders-v0")
    observation = env.reset() # This gets us the image

    # hyperparameters
    batch_size = 10
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99
    num_hidden_layer_neurons = 512
    second_num_hidden_layer_neurons = 256
    input_dimensions = 170 * 160
    learning_rate = 1e-2

    episode_number = 0
    reward_sum = 0
    running_reward = None
    prev_processed_observations = None

    max_random_actions = 90
    max_actions = 100
    current_action_count = 0
    random_action_counter = 0
    random_action = True
    decrease_random_action_after_episode = 20
    episode_number_for_random_action = 0
    max_score = 0

    files_present = glob.glob('weights.pkl')

    weights = {
        '1': np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
        '2': np.random.randn(num_hidden_layer_neurons, second_num_hidden_layer_neurons) / np.sqrt(second_num_hidden_layer_neurons),
        '3': np.random.randn(second_num_hidden_layer_neurons, 6) / np.sqrt(second_num_hidden_layer_neurons)
    }

    if files_present:
        print( 'WARNING: This file already exists!')
        weights = pickle.load(open("weights.pkl", "rb"))
    else:
        print( 'WARNING: NO FILE!')



        #df = pd.read_csv(file_path, header=None, usecols=[3,6])

   # genfromtxt('weights1.csv', delimiter=',')
    # To be used with rmsprop algorithm (http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop)
    expectation_g_squared = {}
    g_dict = {}
    for layer_name in weights.keys():
        expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
        g_dict[layer_name] = np.zeros_like(weights[layer_name])

    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []


    while True:
        env.render()
        processed_observations, prev_processed_observations = preprocess_observations(observation, prev_processed_observations, input_dimensions)
        hidden_layer_values, all_moves_probability = apply_neural_nets(processed_observations, weights)

        episode_observations.append(processed_observations)
        episode_hidden_layer_values.append(hidden_layer_values)

        random_value = randint(0, 1)

        if (random_value == 1 and
                random_action_counter <= max_random_actions and
                current_action_count <= max_actions):
            random_action_counter += 1
            random_action = True
        else:
            if current_action_count >= max_actions:
                current_action_count = 0
                random_action_counter = 0
            random_action = False

        # every 20 episodes will decrease number of random actions
        if (episode_number_for_random_action == decrease_random_action_after_episode and
                max_random_actions > 0):
            episode_number_for_random_action = 0
            max_random_actions -= 1

        # print('episode: %f --> %r random action : rv=  %f , random_action_counter = %f , current_action_count = %f ' %
        #       (episode_number, random_action, random_value, random_action_counter, current_action_count))

        action = choose_action(random_action, all_moves_probability)

        current_action_count += 1

        # carry out the chosen action
        observation, reward, done, info = env.step(action)

        reward_sum += reward
        episode_rewards.append(reward)

        # see here: http://cs231n.github.io/neural-networks-2/#losses
        #fake_label = 1 if action == 2 else 0
        fake_label = np.zeros(6)
        fake_label[action] = 1
        # if action == 4:
        #     fake_label[0] = 1
        # else:
        #     fake_label[1] = 1
        loss_function_gradient = fake_label - all_moves_probability
        episode_gradient_log_ps.append(loss_function_gradient)


        if done: # an episode finished
            episode_number += 1
            episode_number_for_random_action +=1

            # Combine the following values for the episode
            episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
            episode_observations = np.vstack(episode_observations)
            episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
            episode_rewards = np.vstack(episode_rewards)

            # Tweak the gradient of the log_ps based on the discounted rewards
            episode_gradient_log_ps_discounted = discount_with_rewards(episode_gradient_log_ps, episode_rewards, gamma)

            gradient = compute_gradient(
                episode_gradient_log_ps_discounted,
                episode_hidden_layer_values,
                episode_observations,
                weights
            )

            # Sum the gradient for use when we hit the batch size
            for layer_name in gradient:
                g_dict[layer_name] += gradient[layer_name]

            if episode_number % batch_size == 0:
                update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)

            episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], [] # reset values
            observation = env.reset() # reset env
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            if reward_sum > max_score:
                max_score = reward_sum
            print('episode# %f resetting env. episode reward total was %f. running mean: %f MAX: %f' % (episode_number,
                                                                                                        reward_sum,
                                                                                                        running_reward,
                                                                                                        max_score))
            print('episode: %f --> %r random action : MAXRAND = %f , rv=  %f , random_action_counter = %f , current_action_count = %f ' %
                         (episode_number, random_action,max_random_actions, random_value, random_action_counter, current_action_count))

            pickle.dump(weights, open("weights.pkl", "wb"))

            #for layer_name in weights.keys():
            #    df = pd.DataFrame(weights[layer_name])
            #    df.to_csv("weights" +layer_name+ ".csv", header=None, index=None)

            reward_sum = 0
            prev_processed_observations = None

main()
