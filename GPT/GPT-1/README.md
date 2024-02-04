# GPT-1

[](https://www.mikecaptain.com/resources/pdf/GPT-1.pdf)

## Improving Language Understanding by Generative Pre-Training

Abstract

자연어 이해는 텍스트 유추, 질문 응답, 의미 유사성 평가, 문서 분류 등과 같은 다양한 과제를 포함하는 넓은 범위의 작업을 포함합니다. 비라벨링 된 대규모 텍스트 말뭉치는 풍부하지만, 이러한 특정 작업을 학습하기 위한 라벨링 된 데이터는 부족하기 때문에 구분적으로 훈련된 모델이 적절하게 수행되기 어려운 어려움이 있습니다. 우리는 대규모 라벨이 없는 텍스트 코퍼스에서 언어 모델의 생성적 사전 훈련을 한 다음 각 특정 작업에 대한 구분적인 세부 조정을 통해 이러한 작업에서 큰 성과를 얻을 수 있다는 것을 보여줍니다. 이전 접근 방식과는 달리, 모델 아키텍처를 최소한으로 변경하면서도 효과적인 전이를 달성하기 위해 미세 조정 중에 과제에 대한 인식적 입력 변환을 활용합니다. 우리는 자연어 이해를 위한 다양한 벤치마크에서 우리의 접근 방식의 효과를 증명합니다. 우리의 일반적인 과제 중립 모델은 각 작업에 특별히 제작된 아키텍처를 사용하는 구분적으로 훈련된 모델보다 12개의 작업 중 9개에서 혁신을 이루어냅니다. 예를 들어, 우리는 상식적 추론 (Stories Cloze Test)에서 8.9%의 절대 개선, 질문 응답 (RACE)에서 5.7%의 절대 개선 및 텍스트 의미론적 연역 (MultiNLI)에서 1.5%의 절대 개선을 달성합니다.

본 논문의 저자들은 unlabeled text data의 활용 문제를 개선하기 위한 Semi-supervised language model, GPT를 제안한다.

GPT는 기본적으로 Transformer 구조로 이루어져 있어, text sequence의 long-term dependency를 다룰 수 있다.

또한 GPT는 두 가지 학습 단계 1)Unsupervised pre-training, 2)Supervised fine-tuning을 통해 최소한의 구졸 변화로 target task에 transfer 가능한 언어 모델이다.

### 1) Unsupervised pre-training

Given an unsupervised corpus of tokens $U = \{u_1, u_2, …, u_n\}$, we use a standard language modeling objective to maximize the following likelihood:

$L_1(U) = \sum_i logP(u_i|u_{i-k}, ..., u_{i-1}; \theta)$

즉, **unlabeled token**을 통해 일반적인 언어모델의 목적함수를 최대화 하는 과정이다.

$k$는 context window의 크기이며, 조건부 확률 P는 $\theta$를 parameter로 갖는 신경망으로 모델링 된다.

우선 Input token의 context vector $U = (u_{-k}, …, u_{-1})$에 embedding matrix $W_e$를 곱한 후, Positional embedding $W_p$를 더해 Masked_self-attention을 위한 input $h_0$을 만들어준다.

$h_0 = UW_e + W_p$

GPT는 n개의 transformer의 decoder가 stack되어 있는 형태로 본 논문에서는 12개의 decoder block을 사용했다. $l$ 번째 decoder block의 hidden state $h_l$은 이전 decoder block의 hidden state $h_{l-1}$을 입력으로 받아 계산된다.

$h_l =$ transformer_block$(h_{l-1}) \ \forall i \in [1,n]$

마지막 n번째 decoder block의 hidden state output $$ $h_n$에 다시 $W_e$를 곱하여 softmax 함수를 적용하면 output probability $P(u)$를 계산할 수 있다.

$P(u) = softmax(h_nW_e^T)$

![image](https://github.com/jw9603/LM_review/assets/70795645/b976d49f-82d3-4d12-a06f-f02fb2d580f9)


### 2) Supervised fine-tuning

After training the model with the objective in Eq. 1, we adapt the parameters to the supervised target task.

We assume a labeled dataset $C$, where each instance consists of a sequence of input tokens, $x^1, …, x^m$, along with a label $y.$

C의 input token에 대한 GPT의 마지막 decoder block hidden state $h_l^m$을 얻기 위해 앞선 단계에서 얻은 pre-trained model 에 Input token들을 통과시킨다.

그리고 파라미터 $W_y$를 갖는 하나의 linear layer에  $h_l^m$을 통과시켜 softmax probability $P(y|x^1,…, x^m)$를 계산한다.

$P(y|x^1,…, x^m)$ = softmax$(h_l^mW_y)$

따라서 label y에 대해서 지도 학습을 진행할 수 있다.

$L_2(C) = \sum_{(x,y)}logP(y|x^1,…, x^m)$

저자들은 또한 unsupervised pre-training의 목적함수 $L_1$을 supervised fine-tuning을 위한 auxiliary objective로서 추가하였다. 이 때 기존의 $L_1$은 unlabeled dataset $U$에 대한 $L_1(U)$로 계산되었지만, auxiliary objective로서 $L_1$은 Labeled dataset $C$에 대한 $L_1(C)$로 계산된다. $\lambda$는 $L_1(C)$의 반영정도를 정하기 위한 weight hyper-parameter이다.

전반적으로 fine-tuning하는 동안 학습되는 파라미터는 $W_y$와 task-specific에 따라 결정되는 delimiter토큰이다.

### Task-specific input transformations

GPT는 최소한의 구조 변화로 Target task에 적용 가능한 언어 모델이다.
![image](https://github.com/jw9603/LM_review/assets/70795645/687e0142-b28b-4c13-8b81-da1305ad790c)


1. Classification

단순한 classification에 대해서는 기존의 방법 그대로 fine-tuning을 진행한다.

1. Entailment

Entailment는 전제 Premise를 통해 가설 Hypothesis의 참, 거짓을 밝히는 task이다. 따라서 Delimiter로 나누어진 Premise, Hypothesis token 을 delimiter token($)를 기준으로 연결해서 fine-tuning을 진행한다.

1. Similarity

SImilarity task는 문장간의 순서가 존재하지 않다. 따라서 가능한 두가지의 순서 [(문장 1,문장 2),(문장 2,문장 1)]를 모두 고려해야 한다. 두 가지 경우를 input 하여 독립적으로 얻은 결과 을 최종적으로 element-wise addition한 결과로 fine-tuning을 진행한다.

1. QA, Multiple Choice

QA task는 기본적으로 context document $z$에 대한 question $q$가 제시된다. 그리고 주어진 상황에서 가능한 다양한 Answer$\{a_k\}$가 존재한다. 따라서 QA task는 가능한 다양한 답변을 delimiter $와 함께 [z;q;$;$a_k$]로 concat하여 input한다. 각각의 경우는 독립적으로 학습되며, 최종적으로 softmax함수를 통해 답변의 분포를 계산한다.

### Experiment & Result

BooksCorpus dataset을 통해 pre-training을 진행했다.

저자들은 GPT를 통한 semi-supervised 방법의 효과를 증명하기 위해 Natural language inference, QA, sentence similarity, Classification의 다양한 benckmark dataset을 사용하여 실험을 진행했다.

![image](https://github.com/jw9603/LM_review/assets/70795645/884f7cb6-abe1-421f-8ede-a79942b61380)


Natural language inference에서는 거의 대부분의 dataset에서 GPT가 큰 차이로 우수한 성능을 보였다. (3x)는 앙상블 모델을 의미한다.

유일하게 저조한 성능을 보인 RTE dataset은 크기가 작은 데이터셋이다. 따라서 NLI task의 fine tuning은 상대적으로 데이터셋이 클수록 좋은 성능을 보임을 알 수 있다.

![image](https://github.com/jw9603/LM_review/assets/70795645/4cd8abc9-0a6e-4e86-9c0b-4ce52a90b1ea)


두 번째 실험은 QA task에 대한 성능 비교이다.

![image](https://github.com/jw9603/LM_review/assets/70795645/af4cbfa7-baed-42bd-8ae1-ed3fe8afa717)


세 번째 실험은 classification과 similarity task이다.

![image](https://github.com/jw9603/LM_review/assets/70795645/6ea849a4-11e9-4d46-a158-14334d223705)


![image](https://github.com/jw9603/LM_review/assets/70795645/b4a0f75d-17ec-4797-9f63-a87ce6de584e)


왼쪽 그래프는 unsupervised pre-training에서 transformer layer 수에 따른 결과비교다.

layer 수에 따른 유의미한 성능 향상이 있음을 알 수 있다.

오른쪽 그래프는 Transformer 유무와 pre-training에 따른 각각 task의 Zero-shot 성능 비교다. 실선은 Transformer를 사용한 모델이며, 점선은 LSTM을 사용한 모델이다.

대부분의 task가 Transformer를 사용했을 때 더 좋은 성능을 보였으며, pretraining을 거듭할수록 그 차이는 커졌다.

특히 Zero-shot에 대한 LSTM 성능의 분산이 더 컸다는 결과를 통해, pre-training이 전반적인 일반화 능력을 제공한다는 것을 알 수 있다.

![image](https://github.com/jw9603/LM_review/assets/70795645/3eb5f43f-9e50-4dae-b7c4-697e1f88d139)


### Conclusion

우리는 생성적 사전 훈련과 구분적 미세 조정을 통해 하나의 과제 중립 모델을 통해 강력한 자연어 이해를 달성하기 위한 프레임워크를 제안했습니다. 다양한 텍스트의 긴 연속 부분으로 사전 훈련하면 우리의 모델은 중요한 세계 지식을 습득하고, 먼 거리의 종속성을 처리할 능력을 획득하게 되며, 이러한 능력은 질문 응답, 의미 유사성 평가, 의미론적 연역 결정, 텍스트 분류와 같은 구분적인 작업을 성공적으로 해결하기 위해 전이됩니다. 이로써 우리는 연구 대상인 12개의 데이터셋 중 9개에서 현재의 최첨단을 개선합니다. 비지도 (사전) 훈련을 사용하여 구분적 작업의 성능을 향상시키는 것은 오랫동안 기계 학습 연구의 중요한 목표였습니다. 우리의 연구는 실제로 중요한 성능 향상을 달성하는 것이 가능하며, 이 방법론과 가장 잘 맞는 모델 (트랜스포머) 및 데이터 세트 (긴 범위 종속성을 갖는 텍스트)가 무엇인지에 대한 힌트를 제공합니다. 이를 통해 자연어 이해 및 다른 도메인에서 비지도 학습에 대한 새로운 연구를 가능하게 하고, 비지도 학습이 어떤 상황에서 어떻게 작동하는지에 대한 우리의 이해를 더욱 향상시키는 데 도움이 되길 기대합니다.

### References

1. RADFORD, Alec, et al. Improving language understanding by generative pre-training. 2018.
