# RNN

Recurrent Neural Networks

현재 입력과 과거 입력을 동시에 고려하는 형태(Feedback 구조)
히든 노드가 방향을 가진 엣지로 연결되어 순환 구조(directed cycle)을 이룸
음성, 문자 등 순차 데이터 처리에 적합
Sequence 길이에 관계없이 Input, Output을 처리할 수 있는 네트워크 구조 - 유연함

순차데이터 처리 : 다음글자 예측가능(Word2Vec)
기본 구조
Hidden Layer 간 순전파(Forward Propagation)로 Parameter 갱신,
정답 Label 기반으로 역전파(Back Propagation)로 Parameter 갱신


관련정보와 해당정보사용지점 사이 거리가 멀 경우, 
역전파 시에 Gradient가 점차 줄어들어 학습능력이 크게 저하(거리가 가까울 수록 학습능력 좋음) - Vanishing Gradient Problem

LSTM으로 극복
(RNN의 Hidden State에 Cell-State를 추가) - 손실함수 변
Forget Gate : Sigmoid(0,1)로 과거 상태 정보를 잊을 지, 기억할 지 선택
Input Gate : Sigmoid(0,1)로 강도, + Hyperbolic Tangent + Hadamard Product 연산(-1 ~ 1)로 방향


