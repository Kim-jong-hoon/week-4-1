# CNN
:CNN은 이미지, 영상, 음성 등과 같은 2차원 또는 3차원 구조의 데이터를 처리하기 위해 고안된 딥러닝 신경망 구조입니다.

# 🧠 CNN 용어 정리 (Convolutional Neural Network Glossary)

이 저장소는 인공지능 분야, 특히 합성곱 신경망(CNN, Convolutional Neural Network)에 관련된 핵심 용어들을 정리한 자료입니다. 인공지능을 처음 접하거나 CNN 구조를 이해하고자 하는 분들에게 도움이 되기를 바랍니다.

---

## 📌 목차

1. [CNN 기본 구조 용어](#cnn-기본-구조-용어)
2. [합성곱 연산 관련 용어](#합성곱-연산-관련-용어)
3. [정규화 및 활성화 함수](#정규화-및-활성화-함수)
4. [학습 및 최적화 관련 용어](#학습-및-최적화-관련-용어)
5. [성능 평가 지표](#성능-평가-지표)

---

## CNN 기본 구조 용어

| 용어 | 설명 |
|------|------|
| Input Layer | 모델에 입력되는 원본 이미지나 데이터 |
| Convolutional Layer | 이미지의 특징을 추출하는 계층, 필터(커널)를 사용 |
| Filter (Kernel) | 입력 이미지에 대해 합성곱 연산을 수행하는 행렬 |
| Feature Map | 필터를 통해 얻어진 출력 이미지 |
| Padding | 경계 데이터를 보존하기 위해 입력 주변에 0을 추가 |
| Stride | 필터가 이동하는 간격 |
| Pooling Layer | Feature Map의 크기를 줄이는 층 (예: Max Pooling) |
| Flatten | 다차원 데이터를 1차원으로 변환 |
| Fully Connected Layer (FC) | 분류나 회귀를 위한 전결합 층 |

---

## 합성곱 연산 관련 용어

| 용어 | 설명 |
|------|------|
| Receptive Field | 출력 뉴런 하나가 입력에서 바라보는 영역 |
| Depth | 한 레이어에서 출력되는 feature map의 개수 |
| Channel | RGB 이미지에서는 R, G, B 각각이 하나의 채널 |
| Zero Padding | 입력 가장자리에 0을 추가하여 출력 크기 유지 |
| Valid Padding | 패딩 없이 출력 계산 (출력 크기 감소) |
| Same Padding | 입력과 출력 크기를 동일하게 유지하는 패딩 방식 |

---

## 정규화 및 활성화 함수

| 용어 | 설명 |
|------|------|
| ReLU (Rectified Linear Unit) | 음수는 0으로, 양수는 그대로 반환하는 비선형 함수 |
| Sigmoid | 0~1 사이로 출력, 이진 분류에서 자주 사용됨 |
| Tanh | -1~1 사이 출력, Sigmoid보다 중심이 0에 가까움 |
| Softmax | 출력값을 확률로 변환 (다중 클래스 분류) |
| Batch Normalization | 미니배치 단위로 정규화하여 학습 안정화 |

---

## 학습 및 최적화 관련 용어

| 용어 | 설명 |
|------|------|
| Loss Function | 예측값과 실제값 사이의 오차를 계산 |
| Cross-Entropy Loss | 분류 문제에 많이 쓰이는 손실 함수 |
| Optimizer | 파라미터를 업데이트하여 손실을 줄이는 알고리즘 |
| SGD (Stochastic Gradient Descent) | 확률적 경사하강법 |
| Adam | 학습률을 자동 조절하는 고급 최적화 알고리즘 |
| Epoch | 전체 데이터를 한 번 학습한 횟수 |
| Batch Size | 한 번에 학습하는 데이터 샘플 수 |
| Overfitting | 학습 데이터에 과적합되어 일반화 성능 저하 |
| Dropout | 일부 뉴런을 무작위로 제거하여 과적합 방지 |

---

## 성능 평가 지표

| 용어 | 설명 |
|------|------|
| Accuracy | 전체 중 맞춘 비율 |
| Precision | 양성으로 예측한 것 중 실제 양성 비율 |
| Recall | 실제 양성 중 예측이 양성인 비율 |
| F1 Score | Precision과 Recall의 조화 평균 |
| Confusion Matrix | 예측 결과를 TP, FP, FN, TN으로 분류한 행렬 |

---

## 📚 참고 자료

- [DeepLearning.ai](https://www.deeplearning.ai/)
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- Goodfellow et al., Deep Learning (2016)

## CNN 컨벌루션 에니메이션 
https://claude.ai/public/artifacts/2cebc728-66b5-414a-9e97-991f60a2a7e1

### 수직 엣지 필터
-> 노란색 행렬과 수직 엣지 필터의 각 자릿 수를 곱한 뒤 더한다
<br> 열 순서대로 -1 , 0 ,1 ,값을 가진다.
<img width="1462" height="498" alt="수직선 에지 필터" src="https://github.com/user-attachments/assets/b7efb386-489e-4389-9967-0c8024c45110" />

### 수평 엣지 필터
행 순서대로 -1 , 0, 1값을 가진다.
<img width="1695" height="685" alt="수평에지 필터" src="https://github.com/user-attachments/assets/2e16fd8c-324a-411c-bbef-de518ad23c82" />

### 블러 필터
이미지를 부드럽게 해주는 필터이다.
<img width="1688" height="687" alt="블러필터" src="https://github.com/user-attachments/assets/ec7338fb-5003-4cc0-ab5c-80c7642161be" />

### 샤프닝 필터 
이미지를 선명하게 해주는 필터
<img width="1692" height="677" alt="샤프닝 필터" src="https://github.com/user-attachments/assets/3fe429d5-6122-40e7-9b1f-193a33e37bcc" />

## 📋 포함된 내용

1. **기본 개념**: 샤프닝 필터란 무엇인가
2. **구조와 원리**: 어떻게 작동하는가
3. **다양한 종류**: 기본/강한/부드러운 샤프닝
4. **비교표**: 블러 vs 샤프닝 차이점
5. **실제 사용 예시**: 사진 보정, 컴퓨터 비전
6. **주의사항**: 부작용과 올바른 사용법
7. **신호등 검출**: 실제 프로젝트에서의 활용
