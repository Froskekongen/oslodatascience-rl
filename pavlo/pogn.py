import gym
import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import RMSprop


class Game(object):
    '''Class for playing an atari game.'''

    def __init__(self, gameName, agent, render=False):
        self.gameName = gameName
        self.agent = agent
        self.render = render
        self.env = gym.make(self.gameName)
        self.episode_number = 0  # becomes 1 when we start

    def play(self):
        '''Play the game.'''
        observation = self._resetEpisode()
        running_reward = None
        while True:
            if self.render:
                self.env.render()

            action = self.agent.drawAction(observation)

            # step the environment and get new measurements
            observation, reward, done, info = self.env.step(action)
            self.reward_sum += reward
            self.agent.update(reward, done, info)

            if done:  # an episode has finished
                running_reward = self.reward_sum if running_reward is None else running_reward * 0.99 + self.reward_sum * 0.01
                print 'resetting env. episode reward total was %f. running mean: %f' % (self.reward_sum, running_reward)
                # print('ep %d: reward total was %f.' % (self.episode_number, self.reward_sum))
                observation = self._resetEpisode()

    def _resetEpisode(self):
        self.reward_sum = 0
        self.episode_number += 1
        observation = self.env.reset()
        return observation


class Agent(object):
    '''Abstract class for an agent.
    An Agent should implement:
        - model: typically a keras model object.
        - update: update agent after every response (handle response form env. and call updataModel method).
        - preprocess: preprocess observation from environment. Called by drawAction.
        - policy: give an action based on predictions.
        - updateModel: update the model object.
    '''

    model = NotImplemented  # object for holding the model.

    def __init__(self):
        self.resetMemory()

    def update(self, reward, done, info):
        '''Is called to receive the feedback from the environment.
        It has three tasks:
            - store relevant feedback
            - update model if appropriate
            - handle end of game (e.g. reset some states)
        '''
        raise NotImplementedError

    def preprocess_observations(self, input_observation):
        '''Preprocess observation, and typically store in states list'''
        raise NotImplementedError

    def choose_action(self, probability):
        '''Returns an action based on given predictions.'''
        raise NotImplementedError

    def updateModel(self):
        '''Should do all work with updating weights.'''
        raise NotImplementedError

    def resetMemory(self):
        '''Resets actions, states, and rewards.'''
        self.actions = []
        self.episode_observations = []
        self.episode_rewards = []

    def currentState(self):
        '''Returns the latest state.'''
        return self.episode_observations[-1]

    def drawAction(self, input_observation):
        '''Draw an action based on the new observation.'''
        self.preprocess_observations(input_observation)
        probability = self.predict(self.currentState())
        action = self.choose_action(probability)
        self.actions.append(action)
        return action

    def predict(self, states):
        '''Returns predictions based on give states.'''
        return self.model.predict(states)


class KarpathyPolicyPong(Agent):
    '''Karpathy dense policy network.'''
    num_hidden_layer_neurons = 200  # number of hidden layer neurons
    batch_size = 10  # every how many episodes to do a param update?
    learning_rate = 1e-3
    gamma = 0.99  # discount factor for reward
    decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
    input_dimensions = 80 * 80  # input dimensionality: 80x80 grid

    def __init__(self, modelFileName, resume=False):
        super(KarpathyPolicyPong, self).__init__()
        self.modelFileName = modelFileName
        self.resume = resume
        self.prev_processed_observation = None
        self.episode_number = 0
        self.setupModel()

    @staticmethod
    def choose_action(probability):
        random_value = np.random.uniform()
        if random_value < probability:
            # signifies up in openai gym
            return 2
        else:
            # signifies down in openai gym
            return 3

    def update(self, reward, done, info):
        '''See update func in Agent class'''
        self.episode_rewards.append(reward)
        if done:
            self.episode_number += 1
            self.prev_processed_observation = None
            if self.episode_number % self.batch_size == 0:
                self.updateModel()

    def updateModel(self):
        '''Should do all work with updating weights.'''
        print('Updating weights...')
        # stack together all inputs, actions, and rewards for this episode
        fakeLabels = [1 if action == 2 else 0 for action in self.actions]

        episode_observations = np.vstack(self.episode_observations)
        episode_gradient_log_ps = np.vstack(fakeLabels)
        episode_rewards = np.vstack(self.episode_rewards)
        self.resetMemory()

        # compute the discounted reward backwards through time
        episode_gradient_log_ps_discounted = self.discount_with_rewards(episode_rewards)

        # update our model weights (all in one batch)
        self.model.train_on_batch(episode_observations,
                                  episode_gradient_log_ps,
                                  sample_weight=episode_gradient_log_ps_discounted.reshape((-1,)))

    def discount_with_rewards(self, episode_rewards):
        """ discount the gradient with the normalized rewards """
        discounted_episode_rewards = self.discount_rewards(episode_rewards)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards

    def discount_rewards(self, rewards):
        """ take 1D float array of rewards and compute discounted reward """
        """ Actions you took 20 steps before the end result are less important to the overall result than an action you took a step ago.
               This implements that logic by discounting the reward on previous actions based on how long ago they were taken"""
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(xrange(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def setupModel(self):
            inp = Input(shape=(self.input_dimensions,))
            h = Dense(self.num_hidden_layer_neurons, activation='relu')(inp)
            out = Dense(1, activation='sigmoid')(h)
            self.model = Model(inp, out)
            optim = RMSprop(self.learning_rate, self.decay_rate)
            self.model.compile(optim, 'binary_crossentropy')

    def preprocess_observations(self, input_observation):
        """ convert the 210x160x3 uint8 frame into a 6400 float vector """
        processed_observation = input_observation[35:195]  # crop
        processed_observation = self.downsample(processed_observation)
        processed_observation = self.remove_color(processed_observation)
        processed_observation = self.remove_background(processed_observation)
        processed_observation[processed_observation != 0] = 1  # everything else (paddles, ball) just set to 1
        # Convert from 80 x 80 matrix to 1600 x 1 matrix
        processed_observation = processed_observation.astype(np.float).ravel()

        # subtract the previous frame from the current one so we are only processing on changes in the game
        if self.prev_processed_observation is not None:
            input_observation = processed_observation - self.prev_processed_observation
        else:
            input_observation = np.zeros(self.input_dimensions)
        # store the previous frame so we can subtract from it next time
        self.prev_processed_observation = processed_observation
        input_observation = input_observation.reshape((1, -1))
        self.episode_observations.append(input_observation)

    @staticmethod
    def downsample(image):
        # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
        return image[::2, ::2, :]

    @staticmethod
    def remove_color(image):
        """Convert all color (RGB is the third dimension in the image)"""
        return image[:, :, 0]

    @staticmethod
    def remove_background(image):
        image[image == 144] = 0
        image[image == 109] = 0
        return image


def main():
    render = True
    filename = 'test.h5'
    resume = False
    # filename = 'pong_gym_keras_mlp_full_batch.h5'
    # resume = True
    # render = True

    agent = KarpathyPolicyPong(filename, resume=resume)
    game = Game('Pong-v0', agent, render=render)
    game.play()

main()


