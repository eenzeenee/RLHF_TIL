# UNIT 02

## 1. 가치 기반 강화학습 종류
- 정책 기반 vs 가치 기반
    - 정책 기반 : 정책을 직접 학습하여 최적의 정책($\pi^*$) 탐색 
    - 가치 기반 : 최적의 가치 함수 ($Q^*$혹은 $V^*$)를 탐색하는 것 = 최적의 정책
        - 대부분의 가치 기반 방법에서는 Exploration/Exploitation Tradeoff를 처리하는 Epsilon-Greedy 정책 활용
- 상태 가치 함수
    - $V_{\pi}(s) = E_{\pi}[G_t|S_t=s]$
        - $V_{\pi}$ : Value of state $s$
        - $E_{\pi}$ : Expected return
        - $S_t=s$ : if the agent starts as state $s$
    - 각 상태에 대해 상태 가치 함수는 agent가 해당 state에서 시작한 다음 영원히 특정 policy를 따랐을 때 예상되는 기대 수익을 반환
- 행동 가치 함수
    - $Q_{\pi}(s,a) = E_{\pi}[G_t|S_t=s, A_t=a]$
        - $Q_{\pi}$ : Value of state-action pair $s, a$
        - $E_{\pi}$ : Expected return
        - $S_t=s$ : if the agent starts as state $s$
        - $A_t=a$ : if the agent chooses action $a$
    - 상태-행동 pair에 대해 해당 상태에서 해당 행동을 취할 경우 예상되는 기대 수익을 반환
- 반환 결과는 상태, 행동 가치함수 관계 없이 예상되는 기대 보상
- 모든 상태, 상태-행동 쌍에 대한 기대 보상을 계산하는 것은 매우 큰 계산량을 필요로 함
    - 이를 위해 Bellman 방정식이 필요

## 2. Bellman Equation
- 가치 추정을 단순화하는 방법
- $V_{\pi}=E_{\pi}[R_{t+1}+\gamma * V_{\pi}(S_{t+1})|S_t=s]$
    - $V_{\pi}$ : Value of state $s$
    - $E_{\pi}$ : Expected value of **immediate reward**
    - $\gamma * V_{\pi}(S_{t+1})$ : the discounted value of next state ($S_{t+1}$)
    - $S_t=s$ : if the agent starts as state $s$
- 즉각적인 보상 + 할인된 미래 보상

## 3. 몬테카를로 vs 시간 차 학습 (Monte Carlo vs Temporal Difference Learning)
- Monte Carlo
    - 이전 모든 episode의 정보 활용
    - $V(S_t) <- V(S_t)+ \alpha [G_t-V(S_t)]$
        - first $V(S_t)$ : New value of state t
        - second & third $V(S_t)$ : Former estimation of value of state t (Expected return starting as the state $s$)
        - $\alpha$ : learning rate
        - $G_t$ : Return at timestep $t$

- Temporal Difference Learning
    - ($S_t, A_t, R_{t+1}, S_{t+1}$)만 활용
    - 1번의 상호작용만을 기다림
        - $TD$ is to update the $V(S_t)$ at each step
    - Bootstrapping
        - $TD$의 업데이트가 모든 샘플인 $G_t$에 기반한 것이 아니라 일부 샘플링한 기존 추정치인 $V_(S_{t+1})$에 기반하므로
    - $V(S_t) <- V(S_t)+ \alpha[R_{t+1}+\gamma V(S_{t+1})-V(S_t)]$
        - first $V(S_t)$ : New value of state $t$
        - second & third $V(S_t)$ : Former estimation of value of state t (Expected return starting as the state $s$)
        - $\alpha$ : learning rate
        - $R_{t+1}$ : reward
        - $\gamma V(S_{t+1})$ : discounted value of next state
    - TD target = $R_{t+1}$ & $\gamma V(S_{t+1})$
    - 업데이트한 가치 함수를 기반으로 주어진 환경과 계속 상호 작용 진행
- Monte Carlo와 Temporal Difference Learning의 차이점
    - Monte Carlo : 전체 에피소드에서 가치 함수를 업데이트하므로 에피소드의 실제 정확한 discounted reward 활용
    - Temporal Difference Learning : 각 단계에서 가치 함수를 업데이트하고 다음 업데이트 시 해당 함수로 대체 적용, $G_t$라고 하는 실제 예상 수익을 알지 못함

## 4. Q-learning
- 행동 가치 함수를 학습하기 위해 TD learning 방식을 활용하는 방법 (off-policy value-based method)
    - value-based : 각 상태, 상태-행동 쌍에 대해 가치 함수 훈련
    - TD learning : 에피소드의 끝이 아니라 각 단계에서 가치함수 업데이트
    - off-policy : ...?
- 특정 상태에서 특정 조치를 취하는 것의 가치를 결정하는 행동 가치 함수인 Q-function을 학습하는 데에 활용하는 알고리즘
- 가치(value)와 보상(reward)의 차이
    - 가치 (value) : state, state-action 에서 시작하여 정책에 따라 행동하는 경우 예상되는 누적 보상
    - 보상 (reward) : state에서 action을 수행한 후 환경으로부터 얻는 피드백
- Q-learning 단계
    - 내부적으로 모든 state-action 쌍 value를 포함하는 Q-table인, Q-function(action-value function) 학습
    - 주어진 state-action에서 Q-function은 Q-table에서 해당 쌍을 탐색
    - 학습이 완료되면 최적의 Q-function, 즉, 최적의 Q-table을 갖게 됨
    - 최적의 Q-function이 있다면 각 상태에서 취할 최선의 조치를 알고 있기 때문에 최적의 policy를 가지게 된 것과 동일
    - 초기의 Q-table : 임의의 state-action 쌍 값을 가짐
        - agent가 환경을 탐색하고 Q-function/Q-table을 업데이트하며 최적의 policy에 대한 근사치 제공
- Value와 Policy의 관계
    - $\pi^*(s)=argmax_aQ^*(s,a)$
    - finding an optimal value function -> having an optimal policy

- Q-learning 알고리즘
    1. Q-table 초기화 (zero initializing)
    2. epsilon-greedy strategy 활용하여 action 선택
        - $1-\epsilon$의 확률로 exploitation (착취)
        - $\epsilon$의 확률로 exploration (탐색)
        - 학습 초기에는 $\epsilon$값이 매우 크기 때문에 대부분 탐색 시행 -> 학습이 진행됨에 따라 Q-table의 추정이 더욱 좋아지고 탐색이 더 적게 필요해지므로 $\epsilon$값을 줄이게 됨
    3. $A_t$행동 수행하여 보상인 $R_{t+1}$과 다음 상태인 $S_{t+1}$을 얻게 됨
    4. $Q(S_t, A_t)$ 업데이트
        - $Q(S_t, A_t) <- Q(S_t, A_t) + \alpha [R_{t+1}+\gamma max_aQ(S_{t+1},a)-Q(S_t, A_t)]$
            - first $Q(S_t, A_t)$ : New Q-value estimation
            - second & third $Q(S_t, A_t)$ : Former Q-value estimation
            - $\alpha$ : learning rate
            - $R_{t+1}$ : immediate reward
            - $\gamma max_aQ(S_{t+1},a)$ : discounted estimate optimal Q-value of next state
            - TD target : $R_{t+1}$ & $\gamma max_aQ(S_{t+1},a)$
            - TD error : $R_{t+1}$ & $\gamma max_aQ(S_{t+1},a)$ & third $Q(S_t, A_t)$
    - 이러한 단계로 이뤄지기 때문에 Q-learning은 off-policy 알고리즘
- off-policy vs on-policy
    - off-policy : using a **different policy** for acting (inference) and updating (training).
        - Q-learning에서 
            - action policy : epsilon-greedy 방식
            - updating policy : greedy 방식
    - on-policy : using the **same policy** for acting and updating.
        - Sarsa에서
            - epsilon-greedy 방식을 통해 action 선택 및 update 진행