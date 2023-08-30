import gym
# 강화학습 환경을 만들어주는 프레임워크
from model import DeepQNetwork, Agent
from utils import plotLearning
import numpy as np

if __name__ == '__main__':
    env = gym.make('SpaceInvaders-v0')
    brain = Agent(gamma=0.95, epsilon=1.0,
                  alpha=0.003, maxMemorySize=5000,
                  replace=None)
    
    while brain.memCntr < brain.memSize:
        observation = env.reset()
        done = False
        while not done:
            # 0 : no action
            # 1 : fire
            # 2 : move right
            # 3 : move left
            # 4 : move right fire
            # 5 : move left fire
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            if done and info['ale.lives'] == 0:
                reward = -100 # penalty
            brain.storeTransition(np.mean(observation[15:200, 30:125], axis=2), # current state
                                  action, reward, 
                                  np.mean(observation_[15:200, 30:125], axis=2)) # successor state
            observation = observation_
    print('done initializing memory')

    scores = []
    epsHistory = [] 
    # to keep track of the history of epsilons as it decreased over time 
    # because we want to know the relationship between the score and the epsilon
    numGames = 50
    batch_size = 32

    for i in range(numGames):
        print('starting game ', i+1, 'epsilon : %.4f'%brain.EPSILON)
        epsHistory.append(brain.EPSILON)
        done = False
        observation = env.reset()

        frames = [np.sum(observation[15:200, 30:125], axis = 2)]
        score = 0
        lastAction = 0

        while not done:
            if len(frames) == 3: # 3개의 프레임 입력하여 다음 액션 결정
                action = brain.chooseAction(frames) 
                frames = []
            else:
                action = lastAction

            observation_, reward, done, info = env.step(action)
            score += reward
            frames.append(np.sum(observation[15:200, 30:125], axis = 2))

            if done and info['ale.lives'] == 0:
                reward = -100

            brain.storeTransition(np.mean(observation[15:200, 30:125], axis=2), # current state
                                  action, reward, 
                                  np.mean(observation_[15:200, 30:125], axis=2)) # successor state 
            observation = observation_
            brain.learn(batch_size)
            lastAction = action
            # env.render()
        scores.append(score)
        print('score : ', score)
        x = [i + 1 for i in range(numGames)]
        fileName = 'test'+ str(numGames) + '.png'
        plotLearning(x, scores, epsHistory, fileName)

# epsilon decreases linearly over time & the agent's performance gradually increases over time




