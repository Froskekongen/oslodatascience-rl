import numpy as np
import gym
from keras.layers import Conv2D, Dense, Input, Flatten
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from common import LogPong
from skimage.color import rgb2gray
from skimage.transform import resize

class Game(object):
    '''Class for playing an atari game.'''
    def __init__(self, gameName, agent, render=False, logfile=None):
        self.gameName = gameName
        self.agent = agent
        self.render = render
        self.logfile = logfile
        self.logger = LogPong(self.logfile) if self.logfile is not None else None
        self.env = gym.make(self.gameName)
        self.episode = 0 # becomes 1 when we start

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
    '''Abstract class for an agent.
    An Agent should implement:
        - model: typically a keras model object.
        - update: update agent after every response (handle response form env. and call updataModel method).
        - preprocess: preprocess observation from environment. Called by drawAction.
        - policy: give an action based on predictions.
        - updateModel: update the model object.
    '''

    self.model = NotImplemented # object for holding the model.

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

    def preprocess(self, observation):
        '''Preprocess observation, and typically store in states list'''
        raise NotImplementedError

    def policy(self, pred):
        '''Returns an action based on given predictions.'''
        raise NotImplementedError

    def updateModel(self):
        '''Should do all work with updating weights.'''
        raise NotImplementedError

    def resetMemory(self):
        '''Resets actions, states, and rewards.'''
        self.actions = [] 
        self.states= [] 
        self.rewards = []

    def currentState(self):
        '''Returns the latest state.'''
        return self.states[-1]

    def drawAction(self, observation):
        '''Draw an action based on the new observation.'''
        self.preprocess(observation)
        pred = self.predict(self.currentState())
        action = self.policy(pred)
        self.actions.append(action)
        return action

    def predict(self, states):
        '''Returns predictions based on give states.'''
        return self.model.predict(states)






class StandardAtari(Agent):
    '''Abstract class for the standard atari models
    Includes:
        - preprocessing of atari images.
        - keras model.
    '''
    D = 84 # Scaled images are 84x84.
    nbImgInState = 4 # We pass the last 4 images as a state.

    def preprocess(self, observation):
        '''Preprocess observation, and typically store in states list'''
        observation = self.preprocessImage(observation)
        newState = np.zeros((1, D, D, self.nbImgInState))
        if len(states) != 0:
            newState[..., :-1] = self.currentState()[..., 1:]
        newState[..., -1] = observation
        self.states.append(newState)

    @staticmethod
    def preprocessImage(img):
        '''Compute luminance (grayscale in range [0, 1]) and resize to (D, D).'''
        img = rgb2gray(img) # compute luminance 210x160
        img = resize(img, (self.D, self.D)) # resize image
        return img

    def setupModel(self):
        '''Not Implemented (Just a suggestion for structure): 
        Set up the standard DeepMind convnet in Keras.
            modelInputShape = (self.D, self.D, self.nbImgInState)
            self.model = self.deepMindAtariNet(self.nbClasses, modelInputShape, True)
            model.compile(...)
        '''
        raise NotImplementedError

    @staticmethod
    def deepMindAtariNet(nbClasses, inputShape, includeTop=True):
        '''Set up the 3 conv layer keras model.
        classes: Number of outputs.
        inputShape: The input shape without the batch size.
        includeTop: If you only want the whole net, or just the convolutions.
        '''
        inp = Input(shape=inputShape)
        x = Conv2D(32, 8, 8, subsample=(4, 4), activation='relu', border_mode='same', name='conv1')(inp)
        x = Conv2D(64, 4, 4, subsample=(2, 2), activation='relu', border_mode='same', name='conv2')(x)
        x = Conv2D(64, 3, 3, activation='relu', border_mode='same', name='conv3')(x)
        if includeTop:
            x = Flatten(name='flatten')(x)
            x = Dense(512, activation='relu', name='dense1')(x)
            out = Dense(nbClasses, activation='softmax', name='output')(x)
        else:
            out = x
        model = Model(inp, out)
        return model


class A2C_OneGame(StandardAtari):
    '''Almost like the A3C agent, but without the with only one game played.
    nbClasses: Number of action classes.
    nbSteps: Number of steps before updating the agent.
    actionDict: Map an action {0, .. nbClasses} to the actions (passed to atari).
    '''

    def __init__(self, nbClasses, nbSteps, actionDict):
        super().__init__()
        self.nbClasses = nbClasses
        self.nbSteps = nbSteps
        self.setupModel()
        self.actionDict = actionDict

    def setupModel(self):
        '''Setup models:
        self.model is the action predictions.
        self.valueModel is the prediction of the value function.
        '''
        inputShape = (self.D, self.D, self.nbImgInState)
        model = self.deepMindAtariNet(self.nbClasses, inputShape, includeTop=False)
        inp = Input(shape=inputShape)
        x = model(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu', name='dense1')(x)

        action = Dense(self.nbClasses, activation='softmax', name='action')(x)
        self.model = Model(inp, action)

        value = Dense(1, activation='linear', name='value')(x)
        self.valueModel = Model(inp, value)

    def policy(self, pred):
        sampleClass = np.random.choice(range(self.nbClasses), 1, p=pred)
        action = self.actionDict[sampleClass]
        return action
















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
        self.setupModel()

    def policy(self, pred):
        '''Returns an action based on given predictions.'''
        action = 2 if np.random.uniform() < pred else 3 # roll the dice!
        return action

    def update(self, reward, done, info):
        '''See update func in Agent class'''
        self.rewards.append(reward)
        if done:
            self.episode += 1
            self.prev_x = None
            if self.episode % self.batch_size == 0:
                self.updateModel()

    def updateModel(self):
        '''Should do all work with updating weights.'''
        print('Updating weights...')
        # stack together all inputs, actions, and rewards for this episode
        epx = np.vstack(self.states)
        fakeLabels = [1 if action == 2 else 0 for action in self.actions]
        epy = np.vstack(fakeLabels)
        epr = np.vstack(self.rewards)
        self.resetMemory()
    
        # compute the discounted reward backwards through time
        discounted_epr = self._discountRewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
    
        # update our model weights (all in one batch)
        self.model.train_on_batch(epx, epy, sample_weight=discounted_epr.reshape((-1,)))

        if self.episode % (self.batch_size * 3) == 0: 
            self.model.save(self.modelFileName)

    def _discountRewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def setupModel(self):
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
        '''Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector'''
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I.astype(np.float).ravel()

    def preprocess(self, observation):
        '''Proprocess observation. And store in states list'''
        cur_x = self._preprocess_image(observation)
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.D)
        self.prev_x = cur_x
        x = x.reshape((1, -1))
        self.states.append(x)




#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------
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
