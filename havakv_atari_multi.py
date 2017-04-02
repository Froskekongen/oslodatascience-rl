import numpy as np
import gym
from keras.layers import Conv2D, Dense, Input, Flatten
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
import keras.backend as K
from common import LogPong
from skimage.color import rgb2gray
from skimage.transform import resize

class GameAbstract(object):
    '''Abstract game class
    Holds functions that are used for single and multiple games.
    '''
    def __init__(self, gameName, agent, logfile=None):
        self.gameName = gameName
        self.agent = agent
        self.logfile = logfile
        self.logger = LogPong(self.logfile) if self.logfile is not None else None

    def setup(self):
        '''Set up environments.'''
        raise NotImplementedError


class Game(GameAbstract):
    '''Regular game class for playing with an agent.'''
    def __init__(self, gameName, agent, render=False, logfile=None):
        super().__init__(gameName, agent, logfile)
        self.render = render


class GameForContainer(GameAbstract):
    def __init__(self, gameName, container, logfile=None):
        self.gameName = gameName
        self.container = container
        self.logfile = logfile
        self.logger = LogPong(self.logfile) if self.logfile is not None else None

    def resetEpisode(self):
        self.rewardSum = 0
        self.episode += 1
        observation = self.env.reset()
        return observation

    def setup(self):
        self.env = gym.make(self.gameName)
        self.episode = 0 
        self.observation = self._resetEpisode()

    def step(self, action):
        '''Step one frame in game.
        Need to run setup() before we can step.
        '''
        # step the environment and get new measurements
        observation, reward, done, info = self.env.step(action)
        self.rewardSum += reward
        self.container.appendStateActionReward(state, action, reward)
        self.done = done
        
        if done: # an episode has finished
            if self.logger is not None:
                self.logger.log(self.episode, self.rewardSum) # log progress
            self.observation = self.resetEpisode()


class MultiGame(GameAbstract):
    '''Used for playing multiple games simultaneously, on one core.'''
    def __init__(self, gameName, agent, logfile=None):
        super().__init__(gameName, agent, logfile)
        self.nbReplicates = self.agent.nbReplicates

    def setup(self):
        self.games = []
        for container in self.agent.containers:
            self.games.append(GameForContainer(self.gameName, container, None))
        for game in self.games:
            game.setup()

    def step(self, actions):
        for game, action in zip(self.games, actions):
            game.step(action)

    def play(self):
        while True:
            actions = self.agent.drawActions()
            self.step(actions)
            if self.agent.fullBatch: 
                self.agent.update()




class Container(object):
    '''Container for holding game info.
        Typically states, actions and rewards.
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        '''Reset information in container.'''
        self.actions = [] 
        self.states= [] 
        self.rewards = []

    def currentState(self):
        '''Returns last state.'''
        return self.states[-1]

    def appendStateActionReward(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)



class Agent(object):
    '''Abstract class for an agent.
    An Agent should implement:
        - model: typically a keras model object.
        - update: update agent after every response (handle response form env. and call updataModel method).
        - preprocess: preprocess observation from environment. Called by drawAction.
        - policy: give an action based on predictions.
        - updateModel: update the model object.
    '''

    model = NotImplemented # object for holding the model.

    def __init__(self, containerClass, nbContainers=1, **kwargsContainer):
        self.containerClass = containerClass
        self.nbContainers = nbContainers
        self.kwargsContainer = kwargsContainer
        self.containers = [self.containerClass(**self.kwargsContainer) for _ in self.nbContainers]


    def update(self, reward, done, info):
        '''Is called to receive the feedback from the environment.
        It has three tasks:
            - store relevant feedback
            - update model if appropriate
            - handle end of game (e.g. reset some states)
        '''
        raise NotImplementedError

    def preprocess(self, observation):
        '''Preprocess observation, and typically store in states list'''
        raise NotImplementedError

    def policy(self, pred):
        '''Returns an action based on given predictions.'''
        raise NotImplementedError

    def updateModel(self):
        '''Should do all work with updating weights.'''
        raise NotImplementedError

    def setupModel(self):
        '''Function for setting up the self.model object'''
        raise NotImplementedError
    
    def resetExperiences(self):
        '''Resetting containers after updating the model.'''
        for container in self.containers:
            container.reset()

    def currentStates(self):
        '''Returns all last states.'''
        return [container.currentState() for container in self.containers]

    # def _appendActions(self, actions):
        # '''Store actions in containers.'''
        # for container, action in zip(self.containers, actions):
            # container.actions.append(action)

    def drawActions(self):
        '''Draw an action for each container.'''
        # self.preprocess(observation)
        preds = self.predict(self.currentStates())
        actions = self.policy(preds)
        # self._appendActions(actions)
        return actions

    def predict(self, states):
        '''Returns predictions based on give states.'''
        states = np.vstack(states)
        return self.model.predict(states)



