
class Game(object):
    '''Class for playing an atari game.'''
    def __init__(self, gameName, agent, render=False):
        self.gameName = gameName
        self.render = render
        self.agent = agent
        self.env = gym.make(self.gameName)
        self.episode = -1 # becomes 0 when we start

    def initialize(self):
        self.rewardSum = 0
        self.episode += 1
        observation = self.env.reset()
        return observation

    def play(self):
        '''Play the game.'''
        observation = self.initialize()
        while True:
            if self.render: self.env.render()

            action = self.agent.drawAction(observation)

            # step the environment and get new measurements
            observation, reward, done, info = env.step(action)
            self.rewardSum += reward
            agent.appendResponse(reward, done, info) 
            
            if done: # an episode has finished
                observation = self.initialize()
                print('ep %d: reward total was %f.' % (self.episode, self.rewardSum))



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

    def appendResponse(self, reward, done, info):
        '''Should:
            - store relevant response
            - handle end of game
            - call update model if appropriate
        '''
        pass

    def drawAction(self, observation):
        self.preprocess(observation)
        pred = self.predict(self.currentState())
        action = self.policy(pred)
        self.actions.append(action)
        return action

    def predict(self, states):
        '''Returns predictions based on give states.'''
        pass

    def preprocess(self, observation):
        pass

    def currentState(self):
        '''Returns the latest state.'''
        return self.states[-1]

    def policy(self, pred):
        '''Returns an action based on given predictions.'''
        pass

    def updateModel(self):
        '''Should do all work with updating weights.'''
        pass


class KarpathyPolecyPong(Agent):
    '''Karpathy dense policy network.'''
    H = 200 # number of hidden layer neurons
    batch_size = 10 # every how many episodes to do a param update?
    learning_rate = 1e-3
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
    D = 80 * 80 # input dimensionality: 80x80 grid
    def __init__(self):
        super().__init__()
        self.prev_x = None
        self.episode = 0

    def predict(self, states):
        '''Returns predictions based on give states.'''
        return self.model.predict(states)

    def policy(self, pred):
        '''Returns an action based on given predictions.'''
        action = 2 if np.random.uniform() < pred else 3 # roll the dice!
        return action

    def appendResponse(self, reward, done, info):
        self.rewards.append(reward)
        if done:
            self.episode += 1
            self.prev_x = None
            if self.episode % self.batch_size == 0:
                self.updateModel()

    def updataModel(self):
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
            self.model.save(self.model_file_name)

    def _getModel(self):
        """Make keras model"""
        if resume:
            self.model = load_model(model_file_name)
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


    
    










