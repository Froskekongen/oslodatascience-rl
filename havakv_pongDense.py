""" 
Modified scrip from http://karpathy.github.io/2016/05/31/rl/
Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. 

Run with:
    python havavkv_pongDense.py --logfile <filename>
"""
import numpy as np
import gym
from keras.layers import Dense, Input, Flatten
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from common import LogPong
import argparse 

gym.undo_logger_setup() # Stop gym logging

resume = False # resume from previous checkpoint?
render = False
resume = True # resume from previous checkpoint?
render = True

# model initialization 
model_file_name = 'pong_gym_keras_mlp_full_batch.h5'
# model_file_name = 'test.h5'

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
D = 80 * 80 # input dimensionality: 80x80 grid

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def store_all_at_iteration(prefix='', ostfix=''):
    """Store all environment variables."""
    pickle.dump(observation, open('observation.p', 'wb'))
    pickle.dump(reward, open('reward.p', 'wb'))
    pickle.dump(done, open('done.p', 'wb'))
    pickle.dump(info, open('info.p', 'wb'))

def get_dense_model():
    """Make keras model"""
    if resume:
        return load_model(model_file_name)
    inp = Input(shape=(D,))
    h = Dense(H, activation='relu')(inp)
    out = Dense(1, activation='sigmoid')(h)
    model = Model(inp, out)
    optim = RMSprop(learning_rate, decay_rate)
    model.compile(optim, 'binary_crossentropy')
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile', type=str, help="name of log file")
    args = parser.parse_args()
    logger = LogPong(args.logfile) # log progress

    env = gym.make("Pong-v0")
    observation = env.reset()
    prev_x = None # used in computing the difference frame
    xs,ys,drs = [],[],[]
    running_reward = None
    reward_sum = 0
    episode_number = 0
    model = get_dense_model()
    while True:
        if render: env.render()
    
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
    
        if done: # an episode finished (one player has reached a score of 21)
            episode_number += 1

            # boring book-keeping
            # running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            running_reward = reward_sum if running_reward is None else running_reward * 0.9 + reward_sum * 0.1
            print('ep %d: reward total was %f. running mean: %f' % (episode_number, reward_sum, running_reward))
            logger.log(episode_number, reward_sum) # log progress
            reward_sum = 0
            observation = env.reset() # reset env
            prev_x = None

            if episode_number % batch_size == 0:
                print('Updating weights...')
                # stack together all inputs, actions, and rewards for this episode
                epx = np.vstack(xs)
                epy = np.vstack(ys)
                epr = np.vstack(drs)
                xs,ys,hs,drs, = [],[],[],[] # reset array memory
    
                # compute the discounted reward backwards through time
                discounted_epr = discount_rewards(epr)
                # standardize the rewards to be unit normal (helps control the gradient estimator variance)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
    
                # update our model weights (all in one batch)
                model.train_on_batch(epx, epy, sample_weight=discounted_epr.reshape((-1,)))
                del epx, epy, epr, discounted_epr
    
            if episode_number % (batch_size * 3) == 0: model.save(model_file_name)
    
if __name__ == '__main__':
    main()
