#!/usr/bin/env python
# coding: utf-8

# In[4]:


#한글 서체 설치 # 생략 가능
#실행하려면 다음 코드 앞의 #를 삭제한 후 실행해주세요. 

#!sudo apt-get install -y fonts-nanum
#!sudo fc-cache -fv
#!rm ~/.cache/matplotlib -rf

# 설치 후 [런타임 다시 시작]을 해줘야 합니다.


# In[2]:


#한글깨짐 해결 
#이 과정이 싫다면 건너 뛰어도 상관없습니다. 실행하려면 다음 코드 앞의 #을 삭제해주세요.

#import matplotlib.pyplot as plt

#plt.rc('font', family='NanumBarunGothic')


# In[3]:


# Deep q-learning


# In[4]:


get_ipython().system('pip install gym')


# In[5]:


import numpy as np
import random
from IPython.display import clear_output
from collections import deque

import gym
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam


# In[6]:


env = gym.make("Taxi-v3").env
env.render()

print('취할 수 있는 상태 수: {}'.format(env.observation_space.n))
print('취할 수 있는 행동 수: {}'.format(env.action_space.n))


# In[7]:


class Agent:
    def __init__(self, env, optimizer):
        self._state_size = env.observation_space.n
        self._action_size = env.action_space.n
        self._optimizer = optimizer
        self.expirience_replay = deque(maxlen=2000)
        
        self.gamma = 0.6
        self.epsilon = 0.1

        self.q_network = self.build_compile()
        self.target_network = self.build_compile()
        self.target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.append((state, action, reward, next_state, terminated))

    def build_compile(self):
        model = Sequential()
        model.add(Embedding(self._state_size, 10, input_length=1))
        model.add(Reshape((10,)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.expirience_replay, batch_size)
        for state, action, reward, next_state, terminated in minibatch:
            target = self.q_network.predict(state)
            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)
            self.q_network.fit(state, target, epochs=1, verbose=0)


# In[8]:


optimizer = Adam(learning_rate=0.01)
agent = Agent(env, optimizer)
batch_size = 32
num_of_episodes = 10
timesteps_per_episode = 10
agent.q_network.summary()


# In[9]:


for e in range(0, num_of_episodes):
    state = env.reset()
    state = np.reshape(state, [1, 1])

    reward = 0
    terminated = False

    for timestep in range(timesteps_per_episode):
        action = agent.act(state)
        next_state, reward, terminated, info = env.step(action) 
        next_state = np.reshape(next_state, [1, 1])
        agent.store(state, action, reward, next_state, terminated)
        state = next_state

        if terminated:
            agent.target_model()
            break

        if len(agent.expirience_replay) > batch_size:
            agent.retrain(batch_size)
        
    if (e + 1) % 10 == 0:
        print("**********************************")
        print("Episode: {}".format(e + 1))
        env.render()
        print("**********************************")


# In[10]:


#12.5.2 몬테카를로 트리 검색을 적용한 틱택토 게임 구현하기


# In[11]:


boarder = {'1': ' ' , '2': ' ' , '3': ' ' ,
            '4': ' ' , '5': ' ' , '6': ' ' ,
            '7': ' ' , '8': ' ' , '9': ' ' }

board_keys = []

for key in boarder:
    board_keys.append(key)


# In[12]:


def visual_Board(board_num):
    print(board_num['1'] + '|' + board_num['2'] + '|' + board_num['3'])
    print('-+-+-')
    print(board_num['4'] + '|' + board_num['5'] + '|' + board_num['6'])
    print('-+-+-')
    print(board_num['7'] + '|' + board_num['8'] + '|' + board_num['9'])


# In[13]:


def game():
    turn = 'X'
    count = 0
    
    for i in range(8):
        visual_Board(boarder)
        print("당신 차례입니다," + turn + ". 어디로 이동할까요?")
        move = input()        
        if boarder[move] == ' ':
            boarder[move] = turn
            count += 1
        else:
            print("이미 채워져있습니다.\n어디로 이동할까요?")
            continue

        if count >= 5:
            if boarder['1'] == boarder['2'] == boarder['3'] != ' ': 
                visual_Board(boarder)
                print("\n게임 종료.\n")                
                print(" ---------- " +turn + "가 승리했습니다. -----------")               
                break

            elif boarder['4'] == boarder['5'] == boarder['6'] != ' ': 
                visual_Board(boarder)
                print("\n게임 종료.\n")                
                print(" ---------- " +turn + "가 승리했습니다. -----------")
                break

            elif boarder['7'] == boarder['8'] == boarder['9'] != ' ': 
                visual_Board(boarder)
                print("\n게임 종료.\n")                
                print(" ---------- " +turn + "가 승리했습니다. -----------")
                break

            elif boarder['1'] == boarder['4'] == boarder['7'] != ' ': 
                visual_Board(boarder)
                print("\n게임 종료.\n")                
                print(" ---------- " +turn + "가 승리했습니다. -----------")
                break

            elif boarder['2'] == boarder['5'] == boarder['8'] != ' ': 
                visual_Board(boarder)
                print("\n게임 종료.\n")                
                print(" ---------- " +turn + "가 승리했습니다. -----------")
                break

            elif boarder['3'] == boarder['6'] == boarder['9'] != ' ': 
                visual_Board(boarder)
                print("\n게임 종료.\n")                
                print(" ---------- " +turn + "가 승리했습니다. -----------")
                break 

            elif boarder['1'] == boarder['5'] == boarder['9'] != ' ': 
                visual_Board(boarder)
                print("\n게임 종료.\n")                
                print(" ---------- " +turn + "가 승리했습니다. -----------")
                break

            elif boarder['3'] == boarder['5'] == boarder['7'] != ' ': 
                visual_Board(boarder)
                print("\n게임 종료.\n")                
                print(" ---------- " +turn + "가 승리했습니다. -----------")
                break 

        if count == 9:
            print("\n게임 종료.\n")                
            print("동점입니다")


        if turn =='X':
            turn = 'Y'
        else:
            turn = 'X'        

if __name__ == "__main__":
    game()


# In[ ]:




