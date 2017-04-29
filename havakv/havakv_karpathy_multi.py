
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

class Karpathy_container(Container):

    def setup(self):
        '''Setup information in container.'''
        super().setup()
        self._lastStateBeforeUpdate = None
        self.prev_x = None

    @property
    def currentState(self):
        if len(self.states) == 0:
            return self._lastStateBeforeUpdate
        return self.states[-1]

    def _preprocessObservation(self, observation):
        '''Proprocess observation. And store in states list'''
        if self.prev_x is not None:
            if self.isDone:
                self.prev_x = None
        cur_x = self.preprocessImage(observation)
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.agent.D)
        self.prev_x = cur_x
        x = x.reshape((1, -1))
        self.states.append(x)

    def preprocessImage(self, I):
        '''Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector'''
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I.astype(np.float).ravel()

    def discountedRewards(self):
        '''Return the discounted rewards.
        Only works for pong as of now!!!!!!!!!!!!11
        '''
        """Take 1D float array of rewards and compute discounted reward """
        r = np.vstack(self.rewards)
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * self.agent.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def reset(self):
        '''Reset states. 
        We save the last state as it is either the previous step, or
        we are done and we have already loaded the first frame 
        (see step functions).
        '''
        lastStateBeforeUpdate = self.currentState
        self.setup()
        self._lastStateBeforeUpdate = lastStateBeforeUpdate



class KarpathyPolicyPong(Agent):
    '''Karpathy dense policy network.'''
    H = 200 # number of hidden layer neurons
    batch_size = 10 # every how many episodes to do a param update?
    # batchSize = 1024*10
    batchSize = 1024*50
    learning_rate = 1e-3
    # learning_rate = 1e-4
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
    D = 80 * 80 # input dimensionality: 80x80 grid
    containerClass = Karpathy_container

    def __init__(self, nbReplicates, modelFileName, resume=False):
        super().__init__(nbReplicates)
        self.modelFileName = modelFileName
        self.resume = resume
        self.nbUpdates = 0
        # self.prev_x = None
        self.setupModel()

    @property
    def fullBatch(self):
        '''True when we have enough data to make an update.'''
        sizeOfData = sum([len(container.actions) for container in self.containers])
        if sizeOfData < self.batchSize:
            return False
        return True

    # def policy(self, pred):
        # '''Returns an action based on given predictions.'''
        # action = 2 if np.random.uniform() < pred else 3 # roll the dice!
        # return action

    def policy(self, pred):
        actions = []
        for p in pred:
            action = 2 if np.random.uniform() < p[0] else 3 # roll the dice!
            # sampleClass = np.random.choice(range(self.nbActionClasses), 1, p=p)[0]
            # actions.append(self.class2Action[sampleClass])
            actions.append(action)
        return np.stack(actions)

    # def update(self, reward, done, info):
        # '''See update func in Agent class'''
        # self.rewards.append(reward)
        # if done:
            # self.episode += 1
            # self.prev_x = None
            # if self.episode % self.batch_size == 0:
                # self.updateModel()

    def update(self):
        '''Run to do a gradient update of our model
        We use data from all containers.
        '''
        self.updateModel()

        for container in self.containers:
            container.reset()

        if self.nbUpdates % 30 == 0: 
            self.model.save(self.modelFileName)
            # self.model.save_weights(self.modelFileName)

        self.nbUpdates += 1

    def getStates(self, removeStatesWithoutActions=True):
        '''Return all states.
        removeStatesWithoutActions: Removes the current state if we have not taken
            an action for it yet.
        '''
        idxLastResponse = self.idxLastResponse()
        states = []
        for container, idx in zip(self.containers, idxLastResponse):
            s = container.states
            nbActions = len(container.actions)
            nbStates = len(s)
            if removeStatesWithoutActions and (nbActions < nbStates):
                states.extend(s[:-(nbStates - nbActions)][:idx])
            else:
                states.extend(s[:idx])
        return np.vstack(states)

    def getActions(self):
        '''Return all actions.'''
        idxLastResponse = self.idxLastResponse()
        actions = []
        for container, idx in zip(self.containers, idxLastResponse):
            actions.extend(container.actions[:idx])
        return np.vstack(actions)

    def idxLastResponse(self):
        '''Return all actions.'''
        idx = []
        for container in self.containers:
            r = np.array(container.rewards)
            idx.append(np.arange(len(r))[r!=0][-1]+1)
        return idx

    def getDiscountedRewards(self):
        '''Returns discounted rewards from containers.'''
        idxLastResponse = self.idxLastResponse()
        dr = [container.discountedRewards()[:idx] for container, idx in zip(self.containers, idxLastResponse)]
        return np.concatenate(dr)

    def updateModel(self):
        print('Updating model...')
        discountedRewards = self.getDiscountedRewards()
        X = self.getStates()
        fakeLabels = [1 if action == 2 else 0 for action in self.getActions().flatten()]
        Y = np.vstack(fakeLabels)


        discountedRewards -= np.mean(discountedRewards)
        discountedRewards /= np.std(discountedRewards)

        # fakeLabels = [self.action2Class[action] for action in self.getActions().flatten()]
        # actionValues = discountedRewards - valuePreds
        # Y = responseWithSampleWeights(Y, actionValues, self.nbActionClasses)
        # self.model.train_on_batch(X, [Y, discountedRewards])
        self.model.train_on_batch(X, Y, sample_weight=discountedRewards.reshape((-1,)))

    # def updateModel(self):
        # '''Should do all work with updating weights.'''
        # print('Updating weights...')
        # # stack together all inputs, actions, and rewards for this episode
        # epx = np.vstack(self.states)
        # fakeLabels = [1 if action == 2 else 0 for action in self.actions]
        # epy = np.vstack(fakeLabels)
        # epr = np.vstack(self.rewards)
        # self.resetMemory()
    
        # # compute the discounted reward backwards through time
        # discounted_epr = self._discountRewards(epr)
        # # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        # discounted_epr -= np.mean(discounted_epr)
        # discounted_epr /= np.std(discounted_epr)
    
        # # update our model weights (all in one batch)
        # self.model.train_on_batch(epx, epy, sample_weight=discounted_epr.reshape((-1,)))

        # if self.episode % (self.batch_size * 3) == 0: 
            # self.model.save(self.modelFileName)

    # def _discountRewards(self, r):
        # """ take 1D float array of rewards and compute discounted reward """
        # discounted_r = np.zeros_like(r)
        # running_add = 0
        # for t in reversed(range(0, r.size)):
            # if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            # running_add = running_add * self.gamma + r[t]
            # discounted_r[t] = running_add
        # return discounted_r

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

    # @staticmethod
    # def _preprocess_image(I):
        # '''Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector'''
        # I = I[35:195] # crop
        # I = I[::2,::2,0] # downsample by factor of 2
        # I[I == 144] = 0 # erase background (background type 1)
        # I[I == 109] = 0 # erase background (background type 2)
        # I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        # return I.astype(np.float).ravel()

    # def preprocess(self, observation):
        # '''Proprocess observation. And store in states list'''
        # cur_x = self._preprocess_image(observation)
        # x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.D)
        # self.prev_x = cur_x
        # x = x.reshape((1, -1))
        # self.states.append(x)


