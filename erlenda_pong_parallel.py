""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gym
from keras.layers import Input,Dense,Flatten,Convolution2D,Activation
from keras.models import Model,Sequential

from keras.optimizers import RMSprop,Adam
import uuid
from multiprocessing import Pool
from functools import partial

def buildmodel(opt):
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4,4), border_mode='same',input_shape=(80,80,1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2,2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(loss='binary_crossentropy',optimizer=opt)
    print("We finish building the model")
    return model

def create_perc_model(input_dim,hidden_dim):
    inp=Input(shape=(80,80,1), dtype='float32', name='main_input')
    dd=Flatten()(inp)
    dd=Dense(hidden_dim,activation='relu')(dd)
    out=Dense(1,activation='sigmoid')(dd)
    return inp,out

def create_conv_model(input_dim):
    inp=Input(shape=(80,80,1), dtype='float32', name='main_input')
    dd=Convolution2D(32,4,4,border_mode='same',activation='relu')(inp)
    dd=Convolution2D(32,4,4,border_mode='same',activation='relu')(dd)
    dd=Flatten()(dd)
    out=Dense(1,activation='sigmoid')(dd)
    return inp,out

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float32)

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    gamma = 0.99 # discount factor for reward
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def get_dense_model():
    """Make keras model"""

    learning_rate=1e-4
    inp = Input(shape=(80*80,))
    h = Dense(200, activation='relu')(inp)
    out = Dense(1, activation='sigmoid')(h)
    model = Model(inp, out)
    optim = RMSprop(learning_rate)
    model.compile(optim, 'binary_crossentropy')
    try:
        model.load_weights('mod_weights_binary.h5')
        print('weights loaded')
    except:
        pass
    return model

def run_episodes(thr):
    """
    Main issue - make environments that are stateful and can run for some
    iterations (say 4). Make fast gradient updates based on a few iterations
    on many agents.

    It's important to get the environments to keep the state.
    """
    n_episodes=3
    D=80*80
    model=get_dense_model()
    #env = gym.make("Pong-v0")
    env=gym.make("Pong-v0")
    observation = env.reset()
    prev_x = None # used in computing the difference frame
    xs,ys,drs = [],[],[]
    running_reward = None
    reward_sum = 0
    episode_number = 0
    loc_len=0
    while True:
        #if render: env.render()

        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob = model.predict(x.reshape((1, -1)))
        action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

        # record various intermediates (needed later for backprop)
        xs.append(x.reshape((1, -1))) # observation
        # y = 1 if action == 2 else 0 # a "fake label" giving the action chosen
        ys.append(1 if action == 2 else 0) # a "fake label" giving the action chosen

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
        loc_len+=1
        if done: # an episode finished (one player has reached a score of 21)
            episode_number += 1
            print(episode_number,reward_sum,loc_len)
            reward_sum=0
            loc_len=0
            if episode_number>(n_episodes-1):
                return xs,ys,drs
            observation = env.reset()

def run_training(mod):
    xs,ys,drs,inds = [],[],[],[]
    bn=0
    pp=Pool(3)
    while True:
        thr=[ iii for iii in range(3)]
        outs=pp.map(run_episodes,thr)
        for o in outs:
            xs.extend(o[0])
            ys.extend(o[1])
            drs.extend(o[2])
        # for iii in range(4):
        #     xs_n,ys_n,drs_n=run_episodes(mod)
        #     xs.extend(xs_n)
        #     ys.extend(ys_n)
        #     drs.extend(drs_n)
        print('Updating weights...')
        # stack together all inputs, actions, and rewards for this episode
        epx = np.vstack(xs)
        print(epx.shape)
        epy = np.vstack(ys)
        epr = np.vstack(drs)
        xs,ys,drs, = [],[],[] # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        # update our model weights (all in one batch)
        mod.train_on_batch(epx, epy, sample_weight=discounted_epr.reshape((-1,)))
        mod.save_weights('mod_weights_binary.h5')
        del epx, epy, epr, discounted_epr
        bn+=1


if __name__ == "__main__":
    #opt=RMSprop(lr=learning_rate,decay=decay_rate) # rho is decay rate of RMSprop
    # learning_rate=1e-4
    # opt=RMSprop(lr=learning_rate) # rho is decay rate of RMSprop
    #
    # mod=buildmodel(opt)
    mod=get_dense_model()
    run_training(mod)
