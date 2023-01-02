# Leo

## Resume
The projects conducted before from Colab.

## Installation
The deep learning structures are based on keras and tensorflow.

```bash
pip install keras
pip intsall tensorflow
```
## Model Introduction
### LSTM
Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) that is able to remember past information and use it to help process and predict future outcomes. This is achieved through the use of internal memory cells, which can store information over a longer period of time and gates that control the flow of information into and out of the memory cells.

The LSTM architecture was developed to address the problem of vanishing gradients in traditional RNNs, which made it difficult to train networks to remember information over long periods of time. LSTMs are able to overcome this problem by using the gates to selectively retain or discard information from the memory cells, allowing them to maintain a useful context for longer periods of time.

LSTMs have been successful in a wide range of tasks, including language translation, language modeling, sentiment analysis, and more. They are particularly well-suited for tasks that require the model to remember and use long-term dependencies, such as predicting the next word in a sentence based on the context of the entire sentence.

### Embedding
Embedding is a mapping from a discrete set of symbols or items to a continuous vector space. The purpose of an embedding is to represent the symbols in a way that captures some of the relationships or structure present in the data.

One common use of embeddings is in natural language processing tasks, where words or phrases in a text are represented as vectors. These vectors capture some of the meaning of the words and their relationships to other words in the text. For example, the vectors for the words "king" and "queen" might be similar, since they are both related to royalty, while the vectors for "king" and "car" might be very different.

Embeddings can also be used to represent other kinds of symbols, such as user or item identifiers in a recommendation system, or nodes in a graph. The structure of the embedding can be learned from data, or it can be specified by the designer of the model.

In general, embeddings are useful because they can provide a more compact and structured representation of data, which can make it easier to perform tasks such as classification, clustering, or similarity search. They can also provide a way to incorporate prior knowledge about the structure of the data into the model.

### A2C Reinforcement Learning
A2C, also known as Advantage Actor-critic, is a reinforcement learning algorithm that combines the ideas of the actor-critic method and the Q-learning algorithm. It uses value-based method for the critic part and policy-based method for the actor part.

The goal of the actor is to learn a policy that maximizes the expected reward by selecting the best action given the current state. The critic, on the other hand, estimates the value function, which represents the expected future reward for a given state.

The A2C algorithm uses these two components to learn a policy by interacting with the environment. At each time step, the actor takes an action based on the current state, and the environment transitions to a new state and returns a reward. The critic then uses this information to update its estimate of the value function. The actor also updates its policy based on the advantage, which is the difference between the estimated value and the expected reward.

Overall, A2C is an efficient and stable algorithm for learning policies in continuous action spaces, and it has been applied to a wide range of problems in reinforcement learning, including games, robotics, and more.

