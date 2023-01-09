# Leo

## Resume
The projects conducted before from Colab.

## Installation
The deep learning structures are based on keras and tensorflow.

```bash
pip install keras
pip intsall tensorflow
```
## ML Model Introduction
### LSTM
Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) that is able to remember past information and use it to help process and predict future outcomes. This is achieved through the use of internal memory cells, which can store information over a longer period of time and gates that control the flow of information into and out of the memory cells.

The LSTM architecture was developed to address the problem of vanishing gradients in traditional RNNs, which made it difficult to train networks to remember information over long periods of time. LSTMs are able to overcome this problem by using the gates to selectively retain or discard information from the memory cells, allowing them to maintain a useful context for longer periods of time.

LSTMs have been successful in a wide range of tasks, including language translation, language modeling, sentiment analysis, and more. They are particularly well-suited for tasks that require the model to remember and use long-term dependencies, such as predicting the next word in a sentence based on the context of the entire sentence.

### TCN
Temporal Convolutional Networks (TCN) are a type of deep learning model designed for processing sequential data. They are based on the idea of using convolutional neural networks (CNNs) to extract features from sequential data, similar to how CNNs are used to extract features from images.

TCNs are composed of multiple layers of convolutional filters that operate on an input sequence over time. Each layer processes the input using a fixed-size kernel, which slides across the input sequence and applies the same set of filters at each time step. The outputs of the convolutional filters are then combined and passed through a nonlinear activation function before being passed on to the next layer.

One advantage of TCNs is that they are able to effectively capture long-range dependencies in the input sequence, thanks to their use of dilated convolutions. Dilated convolutions allow the kernel to skip over input elements, effectively increasing the size of the kernel and allowing it to cover a larger portion of the input sequence.

TCNs have been applied to a variety of tasks, including natural language processing, speech recognition, and time series forecasting. They have been shown to perform well on these tasks and to be able to learn useful features from the data.

### Embedding
Embedding is a mapping from a discrete set of symbols or items to a continuous vector space. The purpose of an embedding is to represent the symbols in a way that captures some of the relationships or structure present in the data.

One common use of embeddings is in natural language processing tasks, where words or phrases in a text are represented as vectors. These vectors capture some of the meaning of the words and their relationships to other words in the text. For example, the vectors for the words "king" and "queen" might be similar, since they are both related to royalty, while the vectors for "king" and "car" might be very different.

Embeddings can also be used to represent other kinds of symbols, such as user or item identifiers in a recommendation system, or nodes in a graph. The structure of the embedding can be learned from data, or it can be specified by the designer of the model.

In general, embeddings are useful because they can provide a more compact and structured representation of data, which can make it easier to perform tasks such as classification, clustering, or similarity search. They can also provide a way to incorporate prior knowledge about the structure of the data into the model.

### Time Embedding
Time embedding is a mathematical technique used to represent a time-series data set in a higher-dimensional space. It involves constructing a sequence of vectors, called "time-embedded vectors" based on the original time-series data. Each time-embedded vector is constructed by selecting a sequence of consecutive points from the time series and using them to form a vector.

For example, consider a time series of length 10, represented by the sequence of values {x1, x2, x3, ..., x10}. A time-embedded vector of dimension 3 could be constructed by selecting every third point from the time series to form a vector:

[x1, x4, x7]

[x2, x5, x8]

[x3, x6, x9]

[x4, x7, x10]

In this way, the original time-series data is transformed into a set of time-embedded vectors, which can be analyzed using techniques such as dimensionality reduction or clustering. Time embedding can be useful for uncovering patterns or structure in time-series data that might not be apparent when the data is analyzed in its original form.

### tSNE
t-Distributed Stochastic Neighbor Embedding (t-SNE) is a dimensionality reduction technique that is commonly used to visualize high-dimensional data. It is particularly well-suited for visualizing complex, non-linear relationships in data.

t-SNE works by minimizing the divergence between a high-dimensional probability distribution (representing the data in its original space) and a lower-dimensional probability distribution (representing the data in the reduced space). The minimization is done using an optimization algorithm, which adjusts the positions of the points in the reduced space in order to minimize the divergence.

One key feature of t-SNE is that it tries to preserve the local structure of the data, meaning that points that are close together in the high-dimensional space will also be close together in the reduced space. This can make it effective at revealing patterns and relationships in the data that may not be apparent in other dimensionality reduction techniques.

t-SNE is a powerful tool for data visualization, but it has some limitations. It can be sensitive to the choice of hyperparameters, and it can be computationally expensive for large datasets. It is also not suitable for use as a general-purpose dimensionality reduction technique, as it does not preserve the global structure of the data in the same way that some other techniques do.

### A2C Reinforcement Learning
A2C, also known as Advantage Actor-critic, is a reinforcement learning algorithm that combines the ideas of the actor-critic method and the Q-learning algorithm. It uses value-based method for the critic part and policy-based method for the actor part.

The goal of the actor is to learn a policy that maximizes the expected reward by selecting the best action given the current state. The critic, on the other hand, estimates the value function, which represents the expected future reward for a given state.

The A2C algorithm uses these two components to learn a policy by interacting with the environment. At each time step, the actor takes an action based on the current state, and the environment transitions to a new state and returns a reward. The critic then uses this information to update its estimate of the value function. The actor also updates its policy based on the advantage, which is the difference between the estimated value and the expected reward.

Overall, A2C is an efficient and stable algorithm for learning policies in continuous action spaces, and it has been applied to a wide range of problems in reinforcement learning, including games, robotics, and more.

### WGAN
Wasserstein Generative Adversarial Networks (WGANs) are a type of generative model that use a variant of the generative adversarial network (GAN) framework. GANs are a type of model that consists of two parts: a generator network and a discriminator network. The generator tries to produce synthetic data that is similar to a training dataset, while the discriminator tries to distinguish the synthetic data from the real data. The two networks are trained together, with the generator trying to produce synthetic data that is good enough to fool the discriminator and the discriminator trying to accurately distinguish the synthetic data from the real data.

WGANs are an improvement over traditional GANs in that they use the Wasserstein distance, also known as the Earth Mover's distance, as the loss function for training the generator. This distance measure is better suited to training generative models than the traditional GAN loss function, which can be difficult to optimize.

WGANs have been used to generate a variety of types of data, including images, text, and audio. They have been shown to be effective at producing high-quality synthetic data and to be more stable and easier to train than traditional GANs.

### Gradient Penalty
The gradient penalty is a method used to enforce the Lipschitz constraint in the Wasserstein Generative Adversarial Network (WGAN) training algorithm. The Lipschitz constraint states that the output of the discriminator network should not change too quickly as the input changes. Enforcing this constraint helps to stabilize the training process and can improve the quality of the generated samples.

To enforce the Lipschitz constraint, the gradient penalty adds a term to the loss function that penalizes the discriminator for producing large gradients. This is done by sampling random points on the data manifold and computing the gradient of the discriminator with respect to these points. The penalty is then computed as the square of the norm of the gradients, and it is added to the loss function with a weight factor.

The gradient penalty has been shown to be effective at stabilizing the training process and improving the quality of the generated samples in WGANs. It has also been used in other generative models that use the GAN framework, such as the Improved Wasserstein GAN (IWGAN) and the Least Squares GAN (LSGAN).

### CycleGAN
CycleGAN is a type of generative adversarial network (GAN) designed for the purpose of image-to-image translation. It is particularly useful for converting images from one domain (e.g. horses) to another domain (e.g. zebras) without requiring a large dataset of paired images.

The idea behind CycleGAN is to use two GANs, one to translate images from domain A to domain B and another to translate images from domain B back to domain A. These two GANs are trained to generate images that are "realistic" and that can be "reconstructed" back to the original domain. The reconstruction is achieved using a process called "cycle consistency," which means that the images produced by one GAN should be able to be fed into the other GAN and produce an image that is similar to the original.

One of the key advantages of CycleGAN is that it can learn to translate between two domains even if there is no direct mapping between the two. This makes it a powerful tool for tasks such as style transfer and image-to-image translation.

## Quantitive Finance Model Introduction
### Black-Scholes model
The Black-Scholes model relies on several key assumptions, including the assumption that the underlying asset follows a lognormal distribution of price changes, and that the option can only be exercised at expiration. Using these assumptions, the model can be used to calculate the theoretical price of the option, as well as various risk measures such as delta, gamma, theta, and vega.

While the Black-Scholes model has been widely used, it has also been criticized for its assumptions, which may not always hold in practice. Despite this, the model remains a popular tool for pricing options and is widely used in financial markets.

### Heston model
The Heston model is based on the assumption that the underlying asset price follows a geometric Brownian motion process, similar to the Black-Scholes model. However, it also assumes that the volatility of the asset follows a mean-reverting process, meaning that it tends to move back towards a long-term average value over time. The Heston model can be used to calculate the theoretical price of a European call or put option, as well as various risk measures such as delta, gamma, theta, and vega.

Like the Black-Scholes model, the Heston model has been widely used in finance, but has also been criticized for its assumptions, which may not always hold in practice. Despite this, the model remains a popular tool for pricing options and other derivative securities.

### Morte Carlo Simulation
Monte Carlo simulation is a statistical technique that involves using random sampling to perform mathematical calculations. It is named after the city of Monte Carlo in Monaco, which is famous for its casinos.

In a Monte Carlo simulation, random values are generated and used to perform a calculation multiple times. The results of these calculations are then analyzed to determine the likely outcome of the overall process. This technique can be used to model complex systems or to solve problems that cannot be easily solved using traditional methods.

Monte Carlo simulation is used in a wide range of fields, including finance, engineering, and science. It is particularly useful for problems where the probability distribution of the input variables is known or can be approximated, but the problem itself is too complex to solve analytically. By using random sampling to model the problem, it is possible to estimate the likely outcome and to evaluate the impact of different input variables on the result.

### Variance Reduction Methods
There are several variance reduction techniques that can be used to improve the accuracy and efficiency of Monte Carlo simulations. One such technique is stratified sampling, in which the sample space is divided into a number of "strata," and random sampling is used to select a representative sample from each stratum. This can help to reduce the variance in the results by ensuring that the sample is more evenly distributed across the sample space.

Another technique is importance sampling, in which the sampling distribution is chosen to be proportional to the importance of the different parts of the sample space. This can help to reduce the variance by giving more weight to the parts of the sample space that are more important to the final result.

Other variance reduction techniques include control variates, antithetic variates, and common random numbers. These techniques can be used in combination to further reduce the variance in the results of a Monte Carlo simulation.

### Kalman Filter
The Kalman filter is an algorithm that is used to estimate the state of a system based on a series of noisy measurements. It is widely used in engineering, economics, and other fields to filter out noise and to provide accurate estimates of the state of a system.

The Kalman filter works by using a prediction-correction approach to estimate the state of a system. It first makes a prediction about the state of the system based on the previous state and any known inputs. It then uses the measurements of the system to correct the prediction and to produce an improved estimate of the state. This process is repeated over time, with the Kalman filter using the updated estimates of the state to make more accurate predictions.

The Kalman filter is known for its ability to provide accurate estimates of the state of a system even in the presence of noise and other uncertainties. It is often used in situations where the system being modeled is subject to errors or uncertainties, and where it is important to have a reliable estimate of the state of the system.
