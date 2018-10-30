<div id="table-of-contents">
<h2>Table of Contents</h2>
<div id="text-table-of-contents">
<ul>
<li><a href="#sec-1">1. GANs</a>
<ul>
<li><a href="#sec-1-1">1.1. minmax problem:</a></li>
<li><a href="#sec-1-2">1.2. ROBUST probability learning</a></li>
<li><a href="#sec-1-3">1.3. Current design EOT-GAN help robust classification design/large-scale imaginary classification/ semi-supervised learning?</a></li>
<li><a href="#sec-1-4">1.4. Coverlutional lay bounded design, for lipschitz property</a></li>
<li><a href="#sec-1-5">1.5. Discrete GAN or RBM or Autoencoder</a></li>
<li><a href="#sec-1-6">1.6. OT autoencoder</a></li>
<li><a href="#sec-1-7">1.7. Wasserstein coreset/ barycenters and Boosting</a></li>
</ul>
</li>
<li><a href="#sec-2">2. Bayesian</a>
<ul>
<li><a href="#sec-2-1">2.1. Bayesian ELBO</a></li>
<li><a href="#sec-2-2">2.2. EP</a></li>
<li><a href="#sec-2-3">2.3. EM</a></li>
<li><a href="#sec-2-4">2.4. Occam rule</a></li>
<li><a href="#sec-2-5">2.5. The Probabilistic Graphical Models</a></li>
</ul>
</li>
<li><a href="#sec-3">3. Interpretable Methods and Explanations</a>
<ul>
<li><a href="#sec-3-1">3.1. Bayesian Learning</a></li>
</ul>
</li>
<li><a href="#sec-4">4. Record of reading</a>
<ul>
<li><a href="#sec-4-1">4.1. Causal Inference</a></li>
</ul>
</li>
<li><a href="#sec-5">5. Reference</a></li>
</ul>
</div>
</div>


# GANs<a id="sec-1" name="sec-1"></a>

## minmax problem:<a id="sec-1-1" name="sec-1-1"></a>

1.  robust linear clissifier + feature mapping: for robustness against noise and attack

2.  Minmax learning for remote prediction, find the connection to anto-encoder and GAN, do information bottleneck work, try to see (mutual information estimation in estimation information flow in DNN helps or not)
3.  Find OT distance in pix space and feature space, what is the condition for the eqvilence? then use ot upper bound to try the minmax problem&#x2026; target: extend the problem into general case, not just linear dicision rule.

## ROBUST probability learning<a id="sec-1-2" name="sec-1-2"></a>

1.  try to use measure that is robust to noisy samples or outlier, such as beta-estimation, beta divergence, useVAE do the generator may solve the probability of g

## Current design EOT-GAN help robust classification design/large-scale imaginary classification/ semi-supervised learning?<a id="sec-1-3" name="sec-1-3"></a>

## Coverlutional lay bounded design, for lipschitz property<a id="sec-1-4" name="sec-1-4"></a>

<del>First step for the reference(random circulate coverlutional matrix), see if useful</del>

## Discrete GAN or RBM or Autoencoder<a id="sec-1-5" name="sec-1-5"></a>

## OT autoencoder<a id="sec-1-6" name="sec-1-6"></a>

OT is equivalent or leq than autoencoder structured autoencoder:
1.  consider the concatenation/progressive adding more mapping. See if each concatenation has complexity reduction, error bounding&#x2026; \(W(P_X,P_Y) \leq W(P_X, G1(Z1)) \leq W(X, G2(Z2)) \leq \cdots\)
2.  \(W(P_X,P_Y) \leq W(P_X, G2(Z2)) \leq W(G1(Z1), G2(Z2))\), i.e. do alternative mapping twice, what is the benefits of solving \(W(G1(Z1), G2(Z2))\).
3.  Consider adaboosing for condition of going deeper
4.  <sup id="a8524115555a6a2efed287e2f5d16ba4"><a href="#NIPS2017_7126" title="@incollection{NIPS2017_7126,
    title = {AdaGAN: Boosting Generative Models},
    author = {Tolstikhin, Ilya O and Gelly, Sylvain and Bousquet, Olivier and SIMON-GABRIEL, Carl-Johann and Sch\{o}lkopf, Bernhard},
    booktitle = {Advances in Neural Information Processing Systems 30},
    editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
    pages = {5424--5433},
    year = {2017},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/7126-adagan-boosting-generative-models.pdf}
    }">NIPS2017_7126</a></sup> use beta-divergence for each mixture component generator optimization. This allow a training generator to omit tail samples during training. Empirical samples that are not captured during previous generator training will be put more weight and become high-weight samples for next generator training.

## Wasserstein coreset/ barycenters and Boosting<a id="sec-1-7" name="sec-1-7"></a>

# Bayesian<a id="sec-2" name="sec-2"></a>

## Bayesian ELBO<a id="sec-2-1" name="sec-2-1"></a>

generative mode, adversarial optimization, optimize both the bonds and also the gap, is that possible???

## EP<a id="sec-2-2" name="sec-2-2"></a>

replace gaussian assumption with graphic model, optimize it as least as good as gaussian

## EM<a id="sec-2-3" name="sec-2-3"></a>

Use EM philosophy to design the generalizing ability of inference. EM can handle the missing data case. Thus it is possible to embed this into inference algorithm design, by taking missing data as future data for prediction:

1.  1. assuming the joint possible distribution, then embed it for training

2.  1\*. joint distribution in most cases is not available, try Monte Carlo?

3.  2. In batch data feeding procedure, use generative models to generate relevant pseodo-input data, manipulate this percentage consist. (I think I can test it on CNN algorithms first)

## Occam rule<a id="sec-2-4" name="sec-2-4"></a>

Use Occam rule to balance the generalization and accuracy of algorithms and accuracy. A specific problem here could be to use this rule to get the best stacked ELM structures. May be it is interesting to link the regulation parameter lambda with Occam rule.

## The Probabilistic Graphical Models<a id="sec-2-5" name="sec-2-5"></a>

Chapter~15 introduce Bayesian Networks, undirected graphical models/Markov random field, factor graph, and message-passing algorithms including sum-product and max-product.<sup id="46925c57259dfc7c3b5f0d20747e4ba7"><a href="#theodoridis2015machine" title="Theodoridis, Machine learning: a Bayesian and optimization perspective, Academic Press (2015).">theodoridis2015machine</a></sup>.

1.  Check **normal factor graph**, a variant of the factor graphs has been recently introduced, where edges represent variables and vertices represent factors.
2.  Max-product and Max-sum, could be used to detect input signal structures, such as location of objects in pictures.
3.  Could the back-tracking be combined with two-direction message flowing, in order to get optima of input signal? How to find the optima in just two-direction message flowing? May should also pay attention to the hardware requirement(such as memory cost in message-passing inference).
4.  Back-tracking based method for cause input signal identification? Is it possible that after back-tracking, then try to reLearn the input nearby causal input signal? To improve the detection procession?
5.  redundant part signal of input detection? If the redundant part can be identified, the input can be simplified (accelerate the prediction speed?), or the pre-process transformation of input can be identified analytically and then be implied before each detection?

6.  Grouping the different parts of output signal, stop errors back-propagating to irrelevant input parts? This can also be benefited when the causal-output decision relationship is made.

7.  What is the discrimination dimension of overlaying typical activation functions into complex form, for given dimension of input?

8.  Is there metrics as replacement of loss function to better get the causal-output relationship?

# Interpretable Methods and Explanations<a id="sec-3" name="sec-3"></a>

A general framework for learning different kinds of explanations for black box algorithms is proposed and experimentedcite:fong2017interpretable.
Google's interpretability tool: [lucid@github](https://github.com/tensorflow/lucid).

1.  Use lucid to study the inference propagation over CNN or its variants
2.  What is the relationship between salience map and neural network sparsity.
    
    <sup id="669be089a35564ac92c6144f7d35dd91"><a href="#fong2017interpretable" title="Fong \&amp; Vedaldi, Interpretable explanations of black boxes by meaningful perturbation, {arXiv preprint arXiv:1704.03296}, v(), (2017).">fong2017interpretable</a></sup> proposes two test rules for leanring/inference algorithms: 1. classification itself 2. rotation perturbation on input. Regulation formulas are proposed. Deletion, noise and bluring on input images are experimented and discussed.

## Bayesian Learning<a id="sec-3-1" name="sec-3-1"></a>

# Record of reading<a id="sec-4" name="sec-4"></a>

## Causal Inference<a id="sec-4-1" name="sec-4-1"></a>

<sup id="ebbae5f70288dc30ee111f6185f56769"><a href="#pearl2018theoretical" title="Pearl, Theoretical Impediments to Machine Learning With Seven Sparks from the Causal Revolution, {arXiv preprint arXiv:1801.04016}, v(), (2018).">pearl2018theoretical</a></sup> explains the theoretical limits of current
state-of-art machine learning that are mostly based on statistical methods.

# Reference<a id="sec-5" name="sec-5"></a>


# Bibliography
<a id="NIPS2017_7126"></a>[NIPS2017_7126] @incollection{NIPS2017_7126,
title = {AdaGAN: Boosting Generative Models},
author = {Tolstikhin, Ilya O and Gelly, Sylvain and Bousquet, Olivier and SIMON-GABRIEL, Carl-Johann and Sch\"{o}lkopf, Bernhard},
booktitle = {Advances in Neural Information Processing Systems 30},
editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
pages = {5424--5433},
year = {2017},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/7126-adagan-boosting-generative-models.pdf}
} [↩](#a8524115555a6a2efed287e2f5d16ba4)

<a id="theodoridis2015machine"></a>[theodoridis2015machine] Theodoridis, Machine learning: a Bayesian and optimization perspective, Academic Press (2015). [↩](#46925c57259dfc7c3b5f0d20747e4ba7)

<a id="fong2017interpretable"></a>[fong2017interpretable] Fong \& Vedaldi, Interpretable explanations of black boxes by meaningful perturbation, <i>{arXiv preprint arXiv:1704.03296}</i>, <b>()</b>, (2017). <a href="">link</a>. <a href="http://dx.doi.org/">doi</a>. [↩](#669be089a35564ac92c6144f7d35dd91)

<a id="pearl2018theoretical"></a>[pearl2018theoretical] Pearl, Theoretical Impediments to Machine Learning With Seven Sparks from the Causal Revolution, <i>{arXiv preprint arXiv:1801.04016}</i>, <b>()</b>, (2018). <a href="">link</a>. <a href="http://dx.doi.org/">doi</a>. [↩](#ebbae5f70288dc30ee111f6185f56769)
