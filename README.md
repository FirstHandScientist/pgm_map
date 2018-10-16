# GANs

## TODO Tse/ minmax problem:

1.  robust linear clissifier + feature mapping: for robustness against noise and attack

2.  Minmax learning for remote prediction, find the connection to anto-encoder and GAN, do information bottleneck work, try to see (mutual information estimation in estimation information flow in DNN helps or not)

## TODO ROBUST probability learning, try to use measure that is robust to noisy samples or outlier, such as beta-estimation, beta divergence, useVAE do the generator may solve the probability of g

### TODO Does current design EOT-GAN help robust classification design/large-scale imaginary classification/ semi-supervised learning?

## TODO Coverlutional lay bounded design, for lipschitz property

<del>First step for the reference(random circulate coverlutional matrix), see if useful</del>

## TODO Discrete GAN or RBM or Autoencoder

# Record of reading

## Causal Inference

<sup id="ebbae5f70288dc30ee111f6185f56769"><a href="#pearl2018theoretical" title="Pearl, Theoretical Impediments to Machine Learning With Seven Sparks from the Causal Revolution, {arXiv preprint arXiv:1801.04016}, v(), (2018).">pearl2018theoretical</a></sup> explains the theoretical limits of current
state-of-art machine learning that are mostly based on statistical methods.

# Bayesian

## EP

### TODO replace gaussian assumption with graphic model, optimize it as least as good as gaussian

## The Probabilistic Graphical Models

Chapter~15 introduce Bayesian Networks, undirected graphical models/Markov random field, factor graph, and message-passing algorithms including sum-product and max-product.<sup id="46925c57259dfc7c3b5f0d20747e4ba7"><a href="#theodoridis2015machine" title="Theodoridis, Machine learning: a Bayesian and optimization perspective, Academic Press (2015).">theodoridis2015machine</a></sup>.

### TODO Check **normal factor graph**, a variant of the factor graphs has been recently introduced, where edges represent variables and vertices represent factors.

### TODO Max-product and Max-sum, could be used to detect input signal structures, such as location of objects in pictures.

### TODO Could the back-tracking be combined with two-direction message flowing, in order to get optima of input signal? How to find the optima in just two-direction message flowing? May should also pay attention to the hardware requirement(such as memory cost in message-passing inference).

### TODO Back-tracking based method for cause input signal identification? Is it possible that after back-tracking, then try to reLearn the input nearby causal input signal? To improve the detection procession?

### TODO redundant part signal of input detection? If the redundant part can be identified, the input can be simplified (accelerate the prediction speed?), or the pre-process transformation of input can be identified analytically and then be implied before each detection?

### TODO Grouping the different parts of output signal, stop errors back-propagating to irrelevant input parts? This can also be benefited when the causal-output decision relationship is made.

### TODO What is the discrimination dimension of overlaying typical activation functions into complex form, for given dimension of input?

### TODO Is there metrics as replacement of loss function to better get the causal-output relationship?

## Interpretable Methods and Explanations

A general framework for learning different kinds of explanations for black box algorithms is proposed and experimentedcite:fong2017interpretable.
Google's interpretability tool: [lucid@github](https://github.com/tensorflow/lucid).

### TODO Use lucid to study the inference propagation over CNN or its variants

### TODO What is the relationship between salience map and neural network sparsity.

<sup id="669be089a35564ac92c6144f7d35dd91"><a href="#fong2017interpretable" title="Fong \&amp; Vedaldi, Interpretable explanations of black boxes by meaningful perturbation, {arXiv preprint arXiv:1704.03296}, v(), (2017).">fong2017interpretable</a></sup> proposes two test rules for leanring/inference algorithms: 1. classification itself 2. rotation perturbation on input. Regulation formulas are proposed. Deletion, noise and bluring on input images are experimented and discussed.

## Inference and generative models

Imitating human recognition process, when class label is given, features of this class label is generated in mind and then compared to the input data x, to see of which class it belongs to?

## Bayesian Learning

### TODO Use Occam rule to balance the generalization and accuracy of algorithms and accuracy. A specific problem here could be to use this rule to get the best stacked ELM structures. May be it is interesting to link the regulation parameter lambda with Occam rule.

### TODO Use EM philosophy to design the generalizing ability of inference. EM can handle the missing data case. Thus it is possible to embed this into inference algorithm design, by taking missing data as future data for prediction:

1.  1. assuming the joint possible distribution, then embed it for training

2.  1\*. joint distribution in most cases is not available, try Monte Carlo?

3.  2. In batch data feeding procedure, use generative models to generate relevant pseodo-input data, manipulate this percentage consist. (I think I can test it on CNN algorithms first)

# Reference


# Bibliography
<a id="pearl2018theoretical"></a>[pearl2018theoretical] Pearl, Theoretical Impediments to Machine Learning With Seven Sparks from the Causal Revolution, <i>{arXiv preprint arXiv:1801.04016}</i>, <b>()</b>, (2018). <a href="">link</a>. <a href="http://dx.doi.org/">doi</a>. [↩](#ebbae5f70288dc30ee111f6185f56769)

<a id="theodoridis2015machine"></a>[theodoridis2015machine] Theodoridis, Machine learning: a Bayesian and optimization perspective, Academic Press (2015). [↩](#46925c57259dfc7c3b5f0d20747e4ba7)

<a id="fong2017interpretable"></a>[fong2017interpretable] Fong \& Vedaldi, Interpretable explanations of black boxes by meaningful perturbation, <i>{arXiv preprint arXiv:1704.03296}</i>, <b>()</b>, (2017). <a href="">link</a>. <a href="http://dx.doi.org/">doi</a>. [↩](#669be089a35564ac92c6144f7d35dd91)
