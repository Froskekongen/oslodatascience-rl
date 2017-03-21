def run_parallel_episodes(pp, n_episodes = 3, game = "Pong-v0",
                preprocess_function = prepro,
                get_model_func = get_dense_model):
    X,ACTION,REWARD = [],[],[]
    
    outs=pp.map(run_episodes,n_episodes*[0])
    for o in outs:
        X.extend(o[0])
        ACTION.extend(o[1])
        REWARD.extend(o[2])
        
    X = np.vstack(X)
    ACTION = np.vstack(ACTION)
    REWARD = np.vstack(REWARD)
    return X,ACTION,REWARD

def run_episodes(thr = [1,2,3]):
    n_episodes=3
    D=80*80
    model=get_model_func()
    env=gym.make(game)
    observation = env.reset()
    prev_x = None # used in computing the difference frame
    X,ACTION,REWARD = [],[],[]
    running_reward = None
    reward_sum = 0
    episode_number = 0
    loc_len=0
    while True:
        # preprocess the observation, set input to network to be difference image
        cur_x = preprocess_function(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob = model.predict(x.reshape((1, -1)))
        action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

        # record various intermediates (needed later for backprop)
        X.append(x.reshape((1, -1))) # observation
        # y = 1 if action == 2 else 0 # a "fake label" giving the action chosen
        ACTION.append(1 if action == 2 else 0) # a "fake label" giving the action chosen

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        REWARD.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
        loc_len+=1
        if done: # an episode finished (one player has reached a score of 21)
            episode_number += 1
            print(episode_number,reward_sum,loc_len)
            reward_sum=0
            loc_len=0
            if episode_number>(n_episodes-1):
                return X,ACTION,REWARD
            observation = env.reset()