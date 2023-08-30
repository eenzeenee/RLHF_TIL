# Reinforcement learning network

- evaluation network : to evaluate the current state and see which action to take
- target network : to calculate the value of maximal actions during the learning step

- why need two network
    - boil down to eliminating bias in the estimates of the values of the actions
    - the weight of target network are periodically updated with the weights of the evaluation network so that esimates of the maximal actions can get more accurate over time

# what is reinforcement learning

- to get the labeled data for supervised learing = limitation
- learning actively by doing = reinforcement learning

- Agent : the thing that does the learning
    - No labels required 
- actions
- envirionment
- reward / penalty
- policy = the algorithm that dictates how the agent will act in any given situation or state of the environment 
    - expressed as a probability of choosing some action a given environment in some state s
    - these probabilities are not the same as the state transition probabilities

- Bellman equation
    - mathematical relationship between state transitions rewards and the policy
    - it tells us the value meaning the expected future reward of a policy for some state of the environment

- Reinforcement Learning
    - maximizing / solving the bellman equation
    - there is the dilemma (explore-exploit dilemma)
        - short-term reward by exploiting the best-known action
        - be adventurous and choose actions whose reward appears smaller or maybe even unknown
    - one popular solution (epsilon greedy policy)
        - choose the best known action most of the time 
        - occasionally choose a sub-optimal action to see if there's something better

- model-based vs model-free
    - model-based : dynamic programming
    - model-free : Q-learning, Deep Q-learning

# Markov Decision Processes
- 누적 보상을 최대로 하기 위한 최적의 정책을 구하는 문제
- discounted return : 시간 스텝 t 이후 미래에 얻을 수 있는 보상의 총합
    - G_t = r(x_t, a_t)+gamma\*r(x_{t+1}, a_{t+1}) + ... + gamma^{T-t}\*r(x_T, a_T)
        - gamma : 감가율
            - 감가율이 작을수록 agent가 먼 미래에 받을 보상보다 가까운 미래에 받을 보상에 더 큰 가중치를 둔다는 뜻

# The Explore Exploit Dilemma
