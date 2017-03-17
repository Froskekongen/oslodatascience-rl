import numpy as np
import gym
from keras.layers import Dense, Input, Flatten
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from common import LogPong

class Game(object):
    '''Class for playing an atari game.'''
    def __init__(self, gameName, agent, render=False, logfile=None):
        self.gameName = gameName
        self.agent = agent
        self.render = render
        self.logfile = logfile
        self.logger = LogPong(self.logfile) if self.logfile is not None else None
        self.env = gym.make(self.gameName)
        self.episode = -1 # becomes 0 when we start

    def _resetEpisode(self):
        self.rewardSum = 0
        self.episode += 1
        observation = self.env.reset()
        return observation

    def play(self):
        '''Play the game.'''
        observation = self._resetEpisode()
        while True:
            if self.render: self.env.render()

            action = self.agent.drawAction(observation)

            # step the environment and get new measurements
            observation, reward, done, info = self.env.step(action)
            self.rewardSum += reward
            self.agent.update(reward, done, info) 
            
            if done: # an episode has finished
                print('ep %d: reward total was %f.' % (self.episode, self.rewardSum))
                if self.logger is not None:
                    self.logger.log(self.episode, self.rewardSum) # log progress
                observation = self._resetEpisode()



class PlayReplicates(object):
    '''Play multiple replicates of the same game.'''
    def __init__(self, gameName, nbReplicates):
        self.gameName = gameName
        self.nbReplicates = nbReplicates


class Agent(object):
    '''Abstract class for an agent.'''
    def __init__(self):
        self.resetMemory()
        

    def resetMemory(self):
        self.actions = [] 
        self.states= [] 
        self.rewards = []

    def update(self, reward, done, info):
        '''Is called to receive the feedback from the environment.
        It has three tasks:
            - store relevant feedback
            - update model if appropriate
            - handle end of game (e.g. reset some states)
        '''
        raise NotImplementedError

    def drawAction(self, observation):
        self.preprocess(observation)
        pred = self.predict(self.currentState())
        action = self.policy(pred)
        self.actions.append(action)
        return action

    def predict(self, states):
        '''Returns predictions based on give states.'''
        raise NotImplementedError

    def preprocess(self, observation):
        raise NotImplementedError

    def currentState(self):
        '''Returns the latest state.'''
        return self.states[-1]

    def policy(self, pred):
        '''Returns an action based on given predictions.'''
        raise NotImplementedError

    def updateModel(self):
        '''Should do all work with updating weights.'''
        raise NotImplementedError


class KarpathyPolicyPong(Agent):
    '''Karpathy dense policy network.'''
    H = 200 # number of hidden layer neurons
    batch_size = 10 # every how many episodes to do a param update?
    learning_rate = 1e-3
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
    D = 80 * 80 # input dimensionality: 80x80 grid
    def __init__(self, modelFileName, resume=False):
        super().__init__()
        self.modelFileName = modelFileName
        self.resume = resume
        self.prev_x = None
        self.episode = 0
        self._getModel()

    def predict(self, states):
        '''Returns predictions based on give states.'''
        return self.model.predict(states)

    def policy(self, pred):
        '''Returns an action based on given predictions.'''
        action = 2 if np.random.uniform() < pred else 3 # roll the dice!
        return action

    def update(self, reward, done, info):
        self.rewards.append(reward)
        if done:
            self.episode += 1
            self.prev_x = None
            if self.episode % self.batch_size == 0:
                self.updateModel()

    def updateModel(self):
        print('Updating weights...')
        # stack together all inputs, actions, and rewards for this episode
        epx = np.vstack(self.states)
        fakeLabels = [1 if action == 2 else 0 for action in self.actions]
        epy = np.vstack(fakeLabels)
        epr = np.vstack(self.rewards)
        self.resetMemory()
    
        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
    
        # update our model weights (all in one batch)
        self.model.train_on_batch(epx, epy, sample_weight=discounted_epr.reshape((-1,)))

        if self.episode % (batch_size * 3) == 0: 
            self.model.save(self.modelFileName)

    def _getModel(self):
        """Make keras model"""
        if self.resume:
            self.model = load_model(self.modelFileName)
        else:
            inp = Input(shape=(self.D,))
            h = Dense(self.H, activation='relu')(inp)
            out = Dense(1, activation='sigmoid')(h)
            self.model = Model(inp, out)
            optim = RMSprop(self.learning_rate, self.decay_rate)
            self.model.compile(optim, 'binary_crossentropy')

    @staticmethod
    def _preprocess_image(I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I.astype(np.float).ravel()

    def preprocess(self, observation):
        cur_x = self._preprocess_image(observation)
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.D)
        self.prev_x = cur_x
        x = x.reshape((1, -1))
        self.states.append(x)

def test():
    render = False
    filename = 'test.h5'
    resume = False
    # filename = 'pong_gym_keras_mlp_full_batch.h5'
    # resume = True
    # render = True

    gym.undo_logger_setup() # Stop gym logging
    agent = KarpathyPolicyPong(filename, resume=resume)
    game = Game('Pong-v0', agent, render=render, logfile='test.log')
    game.play()


if __name__ == '__main__': 
    test()

