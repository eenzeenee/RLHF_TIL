import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Modlue):
    def __init__(self, ALPHA):
        super(DeepQNetwork, self).__init()
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128*19*8, 512)
        self.fc2 = nn.Linear(512, 6) # 6 actions
        
        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)
        self.loss =nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda_is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation): # observation = sequence of observations
        observation = T.Tensor(observation).to(self.device)
        observation = observation.view(-1, 1, 185, 95) # resize
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        observation = observation.view(-1, 128*19*8) # flatten
        observation = F.relu(self.fc1(observation))

        actions = self.fc2(observation)

        return actions

class Agent(object):
    def __init__(self, gamma, epsilon, alpha, maxMemorySize, epsEnd=0.05, replace=10000, actionSpace=[0, 1, 2, 3, 4, 5]):
        """
        gamma : discount factor for future award
        epsilon : epsilon greedy action selection
        alpha : learning rate
        """
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.actionSpace = actionSpace
        self.memSize = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = [] # numpy array 활용 시 보다 효율적
        self.memCntr = 0
        self.repalce_target_cnt = replace
        self.Q_eval = DeepQNetwork(alpha)
        self.Q_next = DeepQNetwork(alpha)

    def storeTransition(self, state, action, reward, state_):
        if self.memCntr < self.memSize:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.memCntr%self.memSize] = [state, action, reward, state_]
        self.memCntr += 1

    def chooseAction(self, observation):
        # observation = sequence of observations
        rand = np.random.random() # for epsilon greedy action selection
        actions = self.Q_eval.forward(observation)
        if rand < 1 - self.EPSILON: # 1 - self.EPSILON = probability of choosing the maximum action
            # actions = Matrix
            # the number of row = the number of frames we pass in
            # columns = each of six actions 
            action = T.argmax(actions[1]).item()
        else:
            actions = np.random.choice(self.actionSpace)
        self.steps += 1
        return action

    def learn(self, batch_size):
        self.Q_eval.optimizer.zero_grad() # for batch optimization instead of full optimization
        if self.replace_target_cnt is not None and \
           self.learn_step_counter % self.repalce_target_cnt == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())
           
        if self.memCntr + batch_size < self.memSize:
            memStart = int(np.random.choice(range(self.memCntr)))
        else:
            memStart = int(np.random.choice(range(self.memCntr - batch_size - 1)))
        
        miniBatch = self.memory[memStart:memStart+batch_size]
        memory = np.array(miniBatch)

        Qpred = self.Q_eval.forward(list(memory[:, 0][:])).to(self.Q_eval.device) # current state
        Qnext = self.Q_next.forward(list(memory[:, 3][:])).to(self.Q_eval.device) # successor state
        # using the memory subsample

        maxAction = T.argmax(Qnext, dim=1).to(self.Q_eval.device) # maximum action for the successor state
        rewards = T.Tensor(list(memory[:, 2])).to(self.Q_eval.device)
        Qtarget = Qpred
        Qtarget[:, maxAction] = rewards + self.GAMMA*T.max(Qnext[1]) # update value of Q target of the max action
        # Qtarget = max action for the next succesor state
        if self.steps > 500:
            if self.EPSILON - 1e-4 > self.EPS_END:
                self.EPSILON -= 1e-4
            else:
                self.EPSILON = self.EPS_END
        
        loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1
