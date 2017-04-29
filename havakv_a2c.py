import numpy as np
import gym

from keras.layers import Conv2D, Dense, Input, Flatten
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
import keras.backend as K

# from common import LogPong
from skimage.color import rgb2gray
from skimage.transform import resize


class GameForContainer(object):
    '''Play a game and pass data to a container object.
    Should not be run alone, but by MultiGame object.
    gameName: Name of game, e.g. Pong-v0.
    container: Object from Container class.
    render: Render the game played.
    '''
    def __init__(self, gameName, container, logfile=None, render=False):
        self.gameName = gameName
        self.container = container
        # self.logfile = logfile
        # self.logger = LogPong(self.logfile) if self.logfile is not None else None
        self.render = render

        self.env = gym.make(self.gameName)
        self.episode = 0 
        self.resetEpisode()

    def resetEpisode(self):
        '''Reset when an episode has finished (the game passes done == True).'''
        self.rewardSum = 0
        self.episode += 1
        observation = self.env.reset()
        self.container.addObservation(observation)

    def step(self, action):
        '''Step one frame in game.
        Need to run setup() before we can step.
        action: The action that should be passed to the environment.
        '''
        if self.render: self.env.render()
        observation, reward, done, info = self.env.step(action)
        self.rewardSum += reward
        self.container.addEnvStep(observation, reward, done, info)
        self.done = done
        
        if self.done: # an episode has finished
            # if self.logger is not None:
                # self.logger.log(self.episode, self.rewardSum) # log progress
            print(self.episode, self.rewardSum)
            self.resetEpisode()


class MultiGame(object):
    '''Used for playing multiple games simultaneously (all on one core).
    gameName: Name of game, e.g. Pong-v0.
    agent: An Agent object.
    render: Render the game played.
    '''
    def __init__(self, gameName, agent, logfile=None, render=False):
        self.gameName = gameName
        self.agent = agent
        self.logfile = logfile
        # self.logger = LogPong(self.logfile) if self.logfile is not None else None
        self.nbReplicates = self.agent.nbReplicates
        self.render = render

        self.games = []
        for container in self.agent.containers:
            self.games.append(GameForContainer(self.gameName, container, self.logfile, self.render))

    def _step(self, actions):
        '''Step through all games.
        actions: Iterable with one action for each game.
        '''
        for game, action in zip(self.games, actions):
            game.step(action)

    def step(self):
        '''Draw actions, step environment, and potentially update agent.'''
        actions = self.agent.drawActions()
        self._step(actions)
        if self.agent.fullBatch: 
            self.agent.update()

    def play(self):
        '''Play the game. Run step().'''
        while True:
            self.step()



class Container(object):
    '''Abstract class for holding and handling game info. Typically states, actions and rewards.
    Get observations and rewards from game, and actions from agent.
    agent: The agent that used date from the container. 
        Is used to get e.g. dimension of state.
    '''
    def __init__(self, agent):
        self.agent = agent
        self.setup()
        self.episode = 0

    def setup(self):
        '''Setup information in container.
        Also used to reset memory.
        '''
        self.actions = [] 
        self.states= [] 
        self.rewards = []
        self.dones = []
        self.infos = []

    @property
    def currentState(self):
        '''Returns last state.'''
        return self.states[-1]

    def addEnvStep(self, observation, reward, done, info):
        '''Add new data from environment.'''
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)
        if done:
            self.episode += 1
        else:
            self.addObservation(observation)

    def _preprocessObservation(self, observation):
        '''Make state from a new observation and append to states.'''
        raise NotImplementedError

    def addObservation(self, observation):
        '''Adds observation to container as a state.'''
        state = self._preprocessObservation(observation)
        self.states.append(state)

    def addAction(self, action):
        '''Add action to container.'''
        self.actions.append(action)

    @property
    def isDone(self):
        '''If the last environment step returned done.'''
        if len(self.dones) == 0:
            return False
        if self.dones[-1]:
            return True
        return False

    def reset(self):
        '''Reset container after gradient update.
        Typically calls setup function.
        '''
        raise NotImplementedError


class Agent(object):
    '''Abstract class for an agent.
    An agent consist of: 
        - containers for handling the data.
        - a model: typically a keras model object.

    nbReplicates: Number of containers the agent should hold.
    '''

    model = NotImplemented # object for holding the model.
    containerClass = NotImplemented # Class type for storing data.

    def __init__(self, nbReplicates=1, **kwargsContainer):
        self.nbReplicates = nbReplicates
        self.kwargsContainer = kwargsContainer
        self.containers = [self.containerClass(self, **self.kwargsContainer) for _ in range(self.nbReplicates)]

    def drawActions(self):
        '''Draw an action for each container.'''
        preds = self.predict(self.currentStates)
        actions = self.policy(preds)
        self._addActionsToContainers(actions)
        return actions

    def policy(self, pred):
        '''Returns an action based on given predictions.'''
        raise NotImplementedError

    @property
    def fullBatch(self):
        '''True when we have enough data to make an update.'''
        raise NotImplementedError

    def update(self):
        '''Update the model.'''
        raise NotImplementedError

    def predict(self, states):
        '''Returns predictions based on give states.'''
        states = np.vstack(states)
        return self.model.predict(states)

    @property
    def currentStates(self):
        '''Gives the last state from each container.'''
        return [container.currentState for container in self.containers]

    def _addActionsToContainers(self, actions):
        '''Store actions to all the containers.'''
        for container, action in zip(self.containers, actions):
            container.addAction(action)

    def getDiscountedRewards(self):
        '''Returns discounted rewards from containers.'''
        dr = [container.discountedRewards() for container in self.containers]
        return np.concatenate(dr)

    def getStates(self, removeStatesWithoutActions=True):
        '''Return all states.
        removeStatesWithoutActions: Removes the current state if we have not taken
            an action for it yet.
        '''
        states = []
        for container in self.containers:
            st = container.states
            if removeStatesWithoutActions and (len(container.actions) < len(st)):
                states.extend(st[:-1])
            else:
                states.extend(st)
        return np.vstack(states)

    def getActions(self):
        '''Return all actions.'''
        actions = []
        for container in self.containers:
            actions.extend(container.actions)
        return np.vstack(actions)





class A2C_Container(Container):
    '''Container for the A2C agent.''' 

    def setup(self):
        '''Setup information in container.'''
        super().setup()
        self.valuePreds = []
        self._lastStateBeforeUpdate = None

    def addValuePred(self, valuePred):
        '''Add value predictions.'''
        self.valuePreds.append(valuePred)

    def reset(self):
        '''Reset states. 
        We save the last state as it is either the previous step, or
        we are done and we have already loaded the first frame 
        (see step functions).
        '''
        lastStateBeforeUpdate = self.currentState
        self.setup()
        self.states.append(lastStateBeforeUpdate)
        # We add the state as it is either the previous state,
        #   or the first state after resetting the game environment.
        #   See GameForContainer.step and Container.addEnvStep.

    def _preprocessObservation(self, observation):
        '''Make state from a new observation and append to states.'''
        observation = self._preprocessImage(observation)
        newState = np.zeros((1, self.agent.D, self.agent.D, self.agent.nbImgInState))
        isFirstState = self.isDone # isDone is true when we reset the environment.
        if (len(self.states) != 0) or isFirstState: 
            newState[..., :-1] = self.currentState[..., 1:]
        newState[..., -1] = observation
        return newState

    def _preprocessImage(self, img):
        '''Compute luminance (grayscale in range [0, 1]) and resize to (D, D).'''
        img = rgb2gray(img) # compute luminance 210x160
        img = resize(img, (self.agent.D, self.agent.D), mode='constant') # resize image
        return img

    def discountedRewards(self):
        '''Return the discounted rewards.
        Only works for pong as of now!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        '''
        r = np.vstack(self.rewards)
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * self.agent.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r



class A2C(Agent):
    '''Almost like the A3C agent, but no parallelization, and one agent doing
    all updates.

    nbReplicates: Number of replicate games it should be able to play.
    actionSpace: List of allowed actions (passed to atari).
    modelFileName: Path of storing model.
    resume: Load model and resume.
    '''

    D = 84 # Scaled images are 84x84.
    nbImgInState = 4 # We pass the last 4 images as a state.
    batchSize = 1028 # Observations before gradient update
    gamma = 0.99 # discount factor for reward
    mseBeta = 0.5 # Weighting of value mse loss.
    entropyBeta = 0.1 # Weighting of entropy loss.
    learningRate = 1e-4
    decayRate = 0.99 # decay factor for RMSProp leaky sum of grad^2

    containerClass = A2C_Container

    def __init__(self, nbReplicates, actionSpace, modelFileName, resume=False, **kwargsContainer):
        super().__init__(nbReplicates, **kwargsContainer)
        self.actionSpace = actionSpace
        self.nbActionClasses = len(actionSpace)
        self.modelFileName = modelFileName
        self.resume = resume
        self.nbUpdates = 0

        self.setupModel()
        self._makeActionClassMappings()
        self.episode = 0
        self.stepNumber = 0 # iterates every frame

    def _makeActionClassMappings(self):
        '''Make mappings between actions and class numbers.'''
        # This could be done with sklearn.preprocessing.LabelEncoder
        self.action2Class = {action: i for i, action in enumerate(self.actionSpace)}
        self.class2Action = {i: action for i, action in enumerate(self.actionSpace)}

    @property
    def fullBatch(self):
        '''True when we have enough data to make an update.'''
        sizeOfData = sum([len(container.actions) for container in self.containers])
        if sizeOfData < self.batchSize:
            return False
        return True

    def setupModel(self):
        '''Setup models:
        self.actionModel is the action predictions.
        self.valueModel is the prediction of the value function V. 
        self.model is the model with both outputs
        '''
        inputShape = (self.D, self.D, self.nbImgInState)

        inp = Input(shape=inputShape)
        x = Conv2D(32, 8, 8, subsample=(4, 4), activation='relu', border_mode='same', name='conv1')(inp)
        x = Conv2D(64, 4, 4, subsample=(2, 2), activation='relu', border_mode='same', name='conv2')(x)
        x = Conv2D(64, 3, 3, activation='relu', border_mode='same', name='conv3')(x)
        x = Flatten(name='flatten')(x)
        x = Dense(512, activation='relu', name='dense1')(x)

        action = Dense(self.nbActionClasses, activation='softmax', name='action')(x)
        self.actionModel = Model(inp, action)
        # Should we compile model?

        value = Dense(1, activation='linear', name='value')(x)
        self.valueModel = Model(inp, value)
        # Should we compile model?

        self.model = Model(inp, [action, value])

        actionAndEntropyLoss = makeActionAndEntropyLossA3C(self.entropyBeta)
        loss = {'action': actionAndEntropyLoss, 'value': 'mse'}
        loss_weights = {'action': 1, 'value': self.mseBeta}

        optim = RMSprop(self.learningRate, self.decayRate)
        self.model.compile(optim, loss) 

        if self.resume:
            self.model.load_weights(self.modelFileName)
            return

    def drawActions(self):
        '''Draw an action based on the new observation.'''
        actionPreds, valuePreds = self.predict(self.currentStates)
        self._addValuePredsToContainers(valuePreds)
        actions = self.policy(actionPreds)
        self._addActionsToContainers(actions)
        return actions

    def policy(self, pred):
        '''Sample actions from action predictions.'''
        actions = []
        for p in pred:
            sampleClass = np.random.choice(range(self.nbActionClasses), 1, p=p)[0]
            actions.append(self.class2Action[sampleClass])
        return np.stack(actions)

    def update(self):
        '''Run to do a gradient update of our model.
        We use data from all containers.
        '''
        for container in self.containers:
            if not container.isDone:
                container.rewards[-1] = container.valuePreds[-1]

        self.updateModel()

        for container in self.containers:
            container.reset()

        if self.nbUpdates % 30 == 0: 
            self.model.save_weights(self.modelFileName)

        self.nbUpdates += 1
        
    def _addValuePredsToContainers(self, valuePreds):
        '''Store valuePreds to all the containers.'''
        for container, vp in zip(self.containers, valuePreds):
            container.addValuePred(vp)

    def getValuePreds(self):
        '''Return all value predictions.'''
        valuePreds = []
        for container in self.containers:
            valuePreds.extend(container.valuePreds)
        return np.vstack(valuePreds)

    def updateModel(self):
        '''Collect data from containers and perform gradient update.'''
        discountedRewards = self.getDiscountedRewards()
        X = self.getStates()
        fakeLabels = [self.action2Class[action] for action in self.getActions().flatten()]
        Y = np.vstack(fakeLabels)
        valuePreds = self.getValuePreds()
        actionValues = discountedRewards - valuePreds
        Y = responseWithSampleWeights(Y, actionValues, self.nbActionClasses)
        self.model.train_on_batch(X, [Y, discountedRewards])



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


class A2C_withoutEntropy(A2C):
    entropyBeta = 0



if __name__ == '__main__':
    agent = A2C(16, [2, 3], 'tttest.h5')
    # agent = A2C_withoutEntropy(16, [2, 3], 'tttest.h5')
    mulGame = MultiGame('Pong-v0', agent)

    # agent = A2C(1, [2, 3], 'tttest.h5', resume=True)
    # mulGame = MultiGame('Pong-v0', agent, render=True)

    mulGame.play()



