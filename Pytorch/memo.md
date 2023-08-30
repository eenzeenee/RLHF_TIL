# DQN algorithm
- $Q^*: State \times Action \rightarrow \mathbb{R}$
    - 주어진 state에서 action을 취하면 reward를 최대화하는 policy를 쉽게 구성 가능
    - $\pi^*(s) = \arg\!\max_a \ Q^*(s, a)$
- error : $\delta = Q(s, a) - (r + \gamma \max_a' Q(s', a))$
- error 최소화를 위해 **Huber Loss** 활용
    - $\mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta)$
    - $\text{where} \quad \mathcal{L}(\delta) = \begin{cases}
     \frac{1}{2}{\delta^2}  & \text{for } |\delta| \le 1, \\
     |\delta| - \frac{1}{2} & \text{otherwise.}
   \end{cases}$
    - **Huber Loss** : robust regression에서 활용하는 손실함수
        - squared error 손실함수에 비해 이상치에 비교적 덜 민감

# PPO

- PPO : proximality constraints에서 expected return을 최대화하도록 policy 학습
    -  데이터 batch 수집 & 직접 소비되는 policy gradient algorithm
    -  빠르고 효율적인 online, on-policy reinforcement algorithm
    -  TorchRL : loss-module 제공 -> 정책 교육마다 wheel을 새로 만드는 대신 활용 가능

- ClipPPOLoss
    -  주어진 환경에서 policy를 실행하여 데이터 batch를 sampling
    -  REINFORCE 손실의 clipped 버전을 활용하여 이 batch의 랜덤 하위 샘플로 주어진 수의 최적화 단계 실행
    -  clipping은 손실의 최저 bound 제공 -> 더 낮은 return 추정치가 더 높은 return에 비해 선호될 것 

- $L(s,a,\theta_k,\theta) = \min\left(
    \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), \;\;
    g(\epsilon, A^{\pi_{\theta_k}}(s,a))
    \right),$
    - $\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a)$ : 단순히 REINFORCE loss의 importance-weighted 버전
    - $g(\epsilon, A^{\pi_{\theta_k}}(s,a))
    $ : 주어진 threshold 쌍을 초과/미달할 때 비율을 잘라내어 손실 근사

- 탐색
    - stochastic policy 활용
    - $f_{\theta}(\text{observation}) = \mu_{\theta}(\text{observation}), \sigma^{+}_{\theta}(\text{observation})$
    - 정책 설계 단계
        1. 신경망 정의 : D_obs -> 2*D_action
            - mu(loc)와 sigma(scale) 모두 D_action의 차원 가짐
        2. $a$를 추가하여 NormalParamExtractor 위치와 배율 추출
        3. TensorDictModule이 분포를 생성하고 샘플링할 수 있는 확률 제작

# 공부 중 궁금한 점

1. 우리만의 환경을 정의하기 위해서는 어떻게 해야하지?
2. PPO 너무 어려움......