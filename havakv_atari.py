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

class Game(object):
    '''Class for playing an atari game.'''
    def __init__(self, gameName, agent, render=False, logfile=None):
        self.gameName = gameName
        self.agent = agent
        self.render = render
        self.logfile = logfile
        self.logger = LogPong(self.logfile) if self.logfile is not None else None

    def _resetEpisode(self):
        self.rewardSum = 0
        self.episode += 1
        observation = self.env.reset()
        return observation

    def play(self):
        '''Play the game'''
        self.setupGame()
        while True:
            self.step()

    def setupGame(self):
        self.env = gym.make(self.gameName)
        self.episode = 0 # becomes 1 when we start
        self.observation = self._resetEpisode()

    def step(self):
        '''Step one frame in game.
        Need to run setupGame before we can step.
        '''
        if self.render: self.env.render()

        action = self.agent.drawAction(self.observation)

        # step the environment and get new measurements
        self.observation, reward, done, info = self.env.step(action)
        self.rewardSum += reward
        self.agent.update(reward, done, info) 
        
        if done: # an episode has finished
            print('ep %d: reward total was %f.' % (self.episode, self.rewardSum))
            if self.logger is not None:
                self.logger.log(self.episode, self.rewardSum) # log progress
            self.observation = self._resetEpisode()

# class _GameSingleForMultiple(Game):
    # '''This class is similar to the Game class, but it used for playing multiple games.
    # It is created to be used with MultiGames.
    # '''
    # def step(self):
        # self.observation, reward, done, info = self.env.step(action)
        # self.rewardSum += reward


    # def step(self):
        # '''Step one frame in game.
        # Need to run setupGame before we can step.
        # '''
        # raise NotImplementedError

        # action = self.agent.drawAction(self.observation)

        # # step the environment and get new measurements
        # self.observation, reward, done, info = self.env.step(action)
        # self.rewardSum += reward
        # self.agent.update(reward, done, info) 
        
        # if done: # an episode has finished
            # print('ep %d: reward total was %f.' % (self.episode, self.rewardSum))
            # if self.logger is not None:
                # self.logger.log(self.episode, self.rewardSum) # log progress
            # self.observation = self._resetEpisode()


# class MultiGames(Game):
    # '''Play multiple games with a single agent.'''
    # def __init__(self, gameName, nbReplicates, agent, render=False, logfile=None):
        # super().__init__(gameName, agent, render, logfile)
        # self.nbReplicates = nbReplicates

    # def setuptGame(self):
        # raise NotImplementedError('This function is not used for multiple games')

    # def setupGames(self):
        # self.envs = [gym.make(self.gameName) for _ in range(nbReplicates)]


# class GameReplicates(object):
    # '''Play multiple replicates of the same game but NOT parallelized.
    # nbReplicates: Number of replicates.
    # gameName: The name of the game.
    # agents: A MultipleAgents object holding all agents.
    # logfile: 
    # '''
    
    # def __init__(self, nbReplicates, gameName, agents, logfile=None):
        # self.nbReplicates = nbReplicates
        # self.gameName = gameName
        # self.agents = agents
        # self.logfile = logfile
        # self.logger = LogPong(self.logfile) if self.logfile is not None else None

    # def setupGames(self):
        # # Only one game are used for logging
        # self.games = [Game(self.gameName, self.agents.mainAgent, False, self.logfile)]
        # for agent in self.agents.workerAgents:
            # self.games.append(Game(self.gameName, agent, False, None))

        # for game in self.games:
            # game.setupGame()

    # def step(self):
        # '''Step through all games.'''
        # for game in self.games:
            # if not game.agent.done: game.step()

    # def play(self):
        # '''Play all games.'''
        # self.setupGames()
        # while True:
            # self.step()
            # # if all games are done
            # if False not in [game.agent.done for game in self.games]: 
                # self.agents.updateAgents()
        


# class MultipleAgents(object):
    # ''''Does nothing, but can possibly be used for distributed agents...
    # The first agent will be used for updating the model, and the model will be sent
    # to the others.
    # '''
    # def __init__(self, nbReplicates, agentClass,  **kwargsAgent):
        # self.nbReplicates = nbReplicates
        # self.agentClass = agentClass
        # self.kwargsAgent = kwargsAgent
        # raise NotImplementedError('Does nothing, but can possibly be used for distributed agents...')

    # @property
    # def workerAgents(self):
        # return self.agents[1:]

    # @property
    # def mainAgent(self):
        # return self.agents[0]

    # def setupAgents(self):
        # self.agents = [self.agentClass(**self.kwargsAgent) for _ in self.nbReplicates]
        # self.mainAgent.setupModel()
        # self.distributeModelToWorkers()

    # def updateAgents(self):
        # '''Update the model in the agents.'''
        # self.collectExperiences()
        # self.updateModelMainAgent()
        # self.distributeModelToWorkers()
        # self.resetExperiences()
        # raise NotImplementedError('Need to reset agents!!!!!!!!')
        # raise NotImplementedError('Need to take into account that we store last observatoin in observation list, making it longer.')
        # raise NotImplementedError('Set rewards if game not done')

    # def resetExperiences():
        # '''Reset experiences of the agents'''
        # raise NotImplementedError

    # def collectExperiences(self):
        # for agent in self.workerAgents:
            # self.mainAgent.appendExperiences(agent.getExperiences())

    # def updateModelMainAgent(self):
        # '''Perform the update of the model in the mainAgent.'''
        # self.mainAgent.updateModel()

    # def distributeModelToWorkers(self):
        # '''Send the main model to the worker agents.'''
        # for agent in self.workerAgents:
            # agent.model = self.mainAgent.model




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

    def setupModel(self):
        '''Function for setting up the self.model object'''
        raise NotImplementedError
    
    def resetExperiences(self):
        '''Resetting agent after updating the model.'''
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

    def getExperienes(self):
        '''Should return all experiences. 
        Useful when we have multiple worker agents.
        '''
        return [self.actions, self.states, self.rewards]

    def appendExperiences(self, experiences):
        '''Should append experiences from getExperiences().
        Useful when we have multiple worker agents.
        '''
        self.actions, self.states, self.rewards = experiences





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
        newState = np.zeros((1, self.D, self.D, self.nbImgInState))
        if len(self.states) != 0:
            newState[..., :-1] = self.currentState()[..., 1:]
        newState[..., -1] = observation
        self.states.append(newState)

    def preprocessImage(self, img):
        '''Compute luminance (grayscale in range [0, 1]) and resize to (D, D).'''
        img = rgb2gray(img) # compute luminance 210x160
        img = resize(img, (self.D, self.D), mode='constant') # resize image
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
    actionSpace: Allowed actions (passed to atari).
    '''

    gamma = 0.99 # discount factor for reward
    mseBeta = 0.5 # Weighting of value mse loss.
    entropyBeta = 0.1 # Weighting of entropy loss.
    learningRate = 1e-4
    decayRate = 0.99 # decay factor for RMSProp leaky sum of grad^2

    def __init__(self, nbClasses, nbSteps, actionSpace, modelFileName, resume=False, setupModel=True):
        super().__init__()
        self.nbClasses = nbClasses
        self.nbSteps = nbSteps
        self.actionSpace = actionSpace
        self.modelFileName = modelFileName
        self.resume = resume

        if setupModel:
            self.setupModel()
        self._makeActionClassMapping()
        self.episode = 0
        self.stepNumber = 0 # iterates every frame

    def resetMemory(self):
        '''Resets actions, states, rewards, and predicted values.'''
        super().resetMemory()
        self.valuePreds = []

    def _makeActionClassMapping(self):
        self.action2Class = {action: i for i, action in enumerate(self.actionSpace)}
        self.class2Action = {i: action for i, action in enumerate(self.actionSpace)}

    def setupModel(self):
        '''Setup models:
        self.actionModel is the action predictions.
        self.valueModel is the prediction of the value function.
        self.model is the model with both outputs
        '''
        if self.resume:
            self.model = load_model(self.modelFileName)
            # Need the other models as well...
            return
        inputShape = (self.D, self.D, self.nbImgInState)
        model = self.deepMindAtariNet(self.nbClasses, inputShape, includeTop=False)
        inp = Input(shape=inputShape)
        x = model(inp)
        x = Flatten()(x)
        x = Dense(512, activation='relu', name='dense1')(x)

        action = Dense(self.nbClasses, activation='softmax', name='action')(x)
        self.actionModel = Model(inp, action)
        # Should we compile model?

        value = Dense(1, activation='linear', name='value')(x)
        self.valueModel = Model(inp, value)
        # Should we compile model?

        self.model = Model(inp, [action, value])
        # loss = {'action': 'categorical_crossentropy', 'value': 'mse'}
        # loss = {'action': categoricalCrossentropyWithWeights, 'value': 'mse'}
        actionAndEntropyLoss = makeActionAndEntropyLossA3C(self.entropyBeta)
        loss = {'action': actionAndEntropyLoss, 'value': 'mse'}
        loss_weights = {'action': 1, 'value': self.mseBeta}
        optim = RMSprop(self.learningRate, self.decayRate)
        self.model.compile(optim, loss) # Need to make it possible to set other optimizers

    def drawAction(self, observation):
        '''Draw an action based on the new obseravtio.'''
        self.preprocess(observation)
        actionPred, valuePred = self.predict(self.currentState())
        self.valuePreds.append(valuePred)
        action = self.policy(actionPred)
        self.actions.append(action)
        return action

    def policy(self, pred):
        sampleClass = np.random.choice(range(self.nbClasses), 1, p=pred[0])[0]
        action = self.class2Action[sampleClass]
        return action

    def update(self, reward, done, info):
        self.rewards.append(reward)
        self.stepNumber += 1

        if (self.stepNumber == self.nbSteps) or done:
            if len(self.states) == 1 + len(self.actions):
                self.states = self.states[1:] # The first element is from last update
            if not done:
                self.rewards[-1] = self.valuePreds[-1]
            self.updateModel()
            self.resetExperiences()
         
            self.stepNumber = 0
            # prevState = self.currentState()
            # self.resetMemory()
            # self.states.append(prevState) # Store last state (if not done)
            if done:
                self.episode += 1
                # self.resetMemory()

            if self.episode % 10 == 0: 
                self.model.save(self.modelFileName)

    def resetExperiences(self, done=False):
        '''Resetting agent after updating the model.
        done: If game has passed done=True.
        '''
        if done: 
            self.resetMemory()
        else:
            prevState = self.currentState()
            self.resetMemory()
            self.states.append(prevState) # Store last state (if not done)
        

    def updateModel(self):
        rewards = np.vstack(self.rewards)
        discountedRewards = self._discountRewards(rewards)
        X = np.vstack(self.states)
        fakeLabels = [self.action2Class[action] for action in self.actions]
        Y = np.vstack(fakeLabels)
        valuePreds = np.vstack(self.valuePreds)
        actionValues = discountedRewards - valuePreds
        Y = responseWithSampleWeights(Y, actionValues, self.nbClasses)
        self.model.train_on_batch(X, [Y, discountedRewards])


    def _discountRewards(self, r):
        """Take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def getExperienes(self):
        '''Should return all experiences. 
        Useful when we have multiple worker agents.
        '''
        return [self.actions, self.states, self.rewards, self.valuePreds] 

    def appendExperiences(self, experiences):
        '''Should append experiences from getExperiences().
        Useful when we have multiple worker agents.
        '''
        self.actions, self.states, self.rewards, self.valuePreds = experiences

class A3C_SingleWorker(A2C_OneGame):
    '''Like the A3C, but it does not update the model.
    It only plays the game.
    '''
    self.done = False
    def update(self, reward, done, info):
        self.rewards.append(reward)
        self.stepNumber += 1
        if (self.stepNumber == self.nbSteps) or done:
            self.done = True


def responseWithSampleWeights(y, sampleWeights, nbClasses):
    '''Function for making labels ytrueWithWeights passed to 
    categoricalCrossentropyWithWeights(ytrueWithWeights, ypred).
    y: Vector with zero-indexed classes.
    sampleWeights: vector of sample weights.
    nbClasses: number of classes.
    returns: One-hot matrix with y, and last columns contain responses.
    '''
    n = len(y)
    Y = np.zeros((n, nbClasses + 1))
    Y[:, :-1] = to_categorical(y, nbClasses)
    Y[:, -1] = sampleWeights.flatten()
    return Y

def categoricalCrossentropyWithWeights(ytrueWithWeights, ypred):
    '''Like regular categorical cross entropy, but with sample weights for every row.
    ytrueWithWeights is a matrix where the first columns are one hot encoder for the
    classes, while the last column contains the sample weights.
    '''
    return K.categorical_crossentropy(ypred, ytrueWithWeights[:, :-1]) * ytrueWithWeights[:, -1]

def entropyLoss(ypred):
    '''Entropy loss.
    Loss = - sum(pred * log(pred))
    '''
    return K.categorical_crossentropy(ypred, ypred)

def makeActionAndEntropyLossA3C(beta):
    '''The part of the A3C loss function concerned with the actions, 
    i.e. action loss and entropy loss.
    Here we return the loss function than can be passed to Keras.
    beta: Weighting of entropy.
    '''
    def loss(ytrueWithWeights, ypred):
        '''Action and entropy loss for the A3C algorithm.
        ytrueWithWeights: A matrix where the first columns are one hot encoder for the
            classes, while the last column contains the sample weights.
        ypred: Predictions.
        '''
        policyLoss = categoricalCrossentropyWithWeights(ytrueWithWeights, ypred)
        entropy = entropyLoss(ypred)
        return policyLoss - beta * entropy # - because the entropy is positive with minimal values in 0 and 1
    return loss
    





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

def testA2C():
    render = False
    filename = 'testA2C.h5'
    resume = False
    # resume = True
    # render = True

    gym.undo_logger_setup() # Stop gym logging
    actionSpace = [2, 3]
    agent = A2C_OneGame(2, 1024, actionSpace, filename, resume=resume)
    game = Game('Pong-v0', agent, render=render, logfile='test.log')
    game.play()

if __name__ == '__main__': 
    # test()
    testA2C()
