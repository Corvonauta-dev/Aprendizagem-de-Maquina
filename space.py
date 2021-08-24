import csv
import gym 
import random
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.agents import CEMAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(3,height, width, channels)))
    model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  enable_dueling_network=True, dueling_type='avg', 
                   nb_actions=actions, nb_steps_warmup=1000000
                  )
    return dqn

def build_agent_cem(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=1000, window_length=3)
    cema = CEMAgent(model=model,memory=memory, nb_actions=actions, nb_steps_warmup=1000000)

    return cema

env = gym.make('SpaceInvaders-v0')
height, width, channels = env.observation_space.shape
actions = env.action_space.n

env.unwrapped.get_action_meanings()

a = open('score_aleatorio.csv','w')

    

episodes = 2
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = random.choice([0,1,2,3,4,5])
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
    a.write(str(score))
    a.write('\n')
env.close()
a.close()


model = build_model(height, width, channels, actions)
model.summary()

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-4))
# dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)



dqn.save_weights('SavedWeights/dqn/dqn_weights.h5f')
dqn.load_weights('SavedWeights/dqn/dqn_weights.h5f')

scores = dqn.test(env, nb_episodes=2, visualize=True)
print(scores.history['episode_reward'])
f = open('score_dqn.csv','w')
for i in scores.history['episode_reward']:
    f.write(str(i))
    f.write('\n')
f.close()

del model

model = build_model(height, width, channels, actions)
model.summary()

cema = build_agent_cem(model, actions)
cema.compile()
# dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)



cema.save_weights('SavedWeights/cem/dqn_weights.h5f')
cema.load_weights('SavedWeights/cem/dqn_weights.h5f')

scores = cema.test(env, nb_episodes=2, visualize=True)
print(scores.history['episode_reward'])
f = open('score_cem.csv','w')
for i in scores.history['episode_reward']:
    f.write(str(i))
    f.write('\n')
f.close()
# print(np.mean(scores.history['episode_reward']))