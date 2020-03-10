
# Table of Contents

1.  [Bayesian](#orgc3b9d53)
    1.  [Book and Thesis:](#org71a0f6e)
    2.  [Inference methods and techniques](#orgddff408)
        1.  [Application Consideration](#org6449f4b)
        2.  [Classical Inference Methods](#org5390841)
    3.  [Neural network based methods](#org77ce468)
        1.  [Graphical Neural Networks](#org2ab4a05)
        2.  [Variational mehtods](#orgec5896d)
    4.  [Learning of Graphical Models](#orgf23e0a4)
        1.  [Parameter Learning](#orga2306b3)
        2.  [Structure/graph Learning](#orgb667a11)
    5.  [Sparks](#orga75a8d2)
        1.  [Applying RENN for Conditional Random Field](#org4d9f97a)
        2.  [Hierarchical model: RCN + RENN](#orgbe2f888)
        3.  [HMM+GMs](#org16f5903)
        4.  [HRCF for bio-medical application](#org792e6b7)
        5.  [flow+EM](#org287cd9e)
        6.  [flow-model based classification](#orgd92a434)
2.  [Application of Graphical Models and Teaching](#org0c2a08a)
    1.  [With HW](#org8c145d4)
3.  [GANs](#orgbf3885c)
    1.  [Redefine the target of GAN](#orga1cd009)
    2.  [Current design EOT-GAN help robust classification design/large-scale imaginary classification/ semi-supervised learning?](#org14eece6)
    3.  [Coverlutional lay bounded design, for lipschitz property](#org3b7220a)
    4.  [Discrete GAN or RBM or Autoencoder](#org12cb0c4)
    5.  [OT incremental building](#orgbc392cd)
        1.  [Additive Building:](#orga969cae)
        2.  [seems one-to-mutiple barycenter computation is a base-line of mixture](#orgebef844)
        3.  [Concatenating Building](#org857136f)
    6.  [User GAN to learn context noise](#orgecb2b03)
    7.  [HMM+GMM+OT/GAN](#org7775f7c)
    8.  [Using EOT for Coreset finding or generating](#orga00a4dc)
    9.  [convex duality (Farnia):](#orgdc1bf51)
4.  [Robustness](#orga773456)
    1.  [ROBUST probability learning](#orged64c7e)
    2.  [minmax problem:](#org181cf6b)
    3.  [Discussion with Hossein](#org698caa0)
5.  [Interpretable Methods and Explanations](#org62c3c41)
    1.  [Bayesian Learning](#orgf428af7)
6.  [Record of reading](#org2f7dd41)
    1.  [Causal Inference](#orge7f1e52)
7.  [Reference](#org5df7c51)



<a id="orgc3b9d53"></a>

# Bayesian


<a id="org71a0f6e"></a>

## Book and Thesis:

> Sutton, 2010, [An Introduction to Conditional Random Fields](https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf)

> Wainwright, 2008, [Graphical Models, Exponential Families, and Variational Inference](file:///home/dong/Documents/my_eBooks/mLearning/graphical_models_wainwright.pdf)
> Koller, 2009, [Probabilistic graphical models: principles and techniques](file:///home/dong/Documents/my_eBooks/mLearning/probabilistic_graphical_models_principles_techniques.pdf)

**Thesis**:

> Yingzhen Li, 2018, [Approximate Inference: New Visions](https://www.repository.cam.ac.uk/handle/1810/277549)

> Adrian Weller, 2014, [Methods for Inference in Graphical Models](http://mlg.eng.cam.ac.uk/adrian/phd_FINAL.pdf)

Ihler, Alexander:

> Lou, Qi, 2018, [Anytime Approximate Inference in Graphical Models](https://escholarship.org/uc/item/7sc0m97f)

> Ping, Wei, 2016, [Learning and Inference in Latent Variable Graphical Models](https://escholarship.org/uc/item/7q90z4b5)

> Forouzan, Sholeh, 2015, [Approximate Inference in Graphical Models](https://escholarship.org/uc/item/5n4733cz)

> Qiang, Liu, 2014, [Reasoning and Decisions in Probabilistic Graphical Models - A Unified Framework](https://escholarship.org/uc/item/92p8w3xb)

Minka:

> Yuan Qi, 2005, [Extending Expectation Propagation for Graphical Models](https://affect.media.mit.edu/pdfs/05.qi-phd.pdf)

> Thomas P Minka, 2001, [A family of algorithms for approximate Bayesian inference](https://tminka.github.io/papers/ep/minka-thesis.pdf)


<a id="orgddff408"></a>

## Inference methods and techniques


<a id="org6449f4b"></a>

### Application Consideration

1.  What will I get by applying the RNN's mean field explanation to RNN augmented Kalman filter?
    
    Ref1: [Satorras, 2019, Combining Generative and Discriminative Models for Hybrid Inference](https://papers.nips.cc/paper/9532-combining-generative-and-discriminative-models-for-hybrid-inference.pdf)
    
    Ref2: [Zheng, 2019, Conditional Random Fields as Recurrent Neural Networks](https://arxiv.org/pdf/1502.03240.pdf)
    
    Ref3: [Krahenbuhl, 2011, Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials](https://arxiv.org/abs/1210.5644)

2.  Can I interpret tree-reweighted RENN as a mixture method of MF, LoopyBP and GBP (or even including TWBP) after convexcify its corresponding hypergraph?


<a id="org5390841"></a>

### Classical Inference Methods

1.  Hand-crafted message passing, BP, GBP, Tree-reweighted BP and EP, PowerEP

2.  <font color="green">Alpha belief propagation</font>
    
    <font color="green"> convergence property of Alpha belief propagation</font>

3.  Tree-reweighted BP for MAP problems (for HW map problem complete graph)
    
    Refer to [exact MAP estimates by hypertree agreement](https://papers.nips.cc/paper/2206-exact-map-estimates-by-hypertree-agreement.pdf)
    
    [tree-reweighted belief propagation algorithms and approximated ML esimation by pseudo-moment matching](http://ssg.mit.edu/group/willsky/publ_pdfs/166_pub_AISTATS.pdf)

4.  Generalized BP for marginal distributions, [yedidis 2005, constructing free energy approximations and Generalized belief propagation algorithms](https://www.cs.princeton.edu/courses/archive/spring06/cos598C/papers/YedidaFreemanWeiss2004.pdf)
    
    What about GBP for MAP problem?

5.  Tree-structured EP, [Tree-structured approximations by expectation propagation](https://tminka.github.io/papers/eptree/minka-eptree.pdf)

6.  Convergence Analysis, Roosta, 2008, [Convergence Analysis of Reweighted Sum-Product Algorithms](https://ieeexplore.ieee.org/document/4599175)


<a id="org77ce468"></a>

## Neural network based methods


<a id="org2ab4a05"></a>

### Graphical Neural Networks

1.  literature development path
    
    > Half-automated message passing, [Learning to Pass Expectation Propagation Messages](https://papers.nips.cc/paper/5070-learning-to-pass-expectation-propagation-messages.pdf) , message-level automation
    
    > Training neural network to do message passing, [Inference in Probabilistic Graphical Models by Graph Neural Networks](https://arxiv.org/abs/1803.07710) , train NN for message updates, and also NN for mapping messages to estimations. A good property observed in the work, trained NNs can be used for different factor graphs with different potentials and structures
    Same track, [GMNN: Graph Markov Neural Networks](https://arxiv.org/abs/1905.06214), semi-supervised learning, EM is used for training.
    
    > More generalized computation power: [Graph Net](https://github.com/deepmind/graph_nets), A graph network takes a graph as input and returns a graph as output. The input graph has edge- (E ), node- (V ), and global-level (u) attributes. The output graph has the same structure, but updated attributes. Graph networks are part of the broader family of "graph neural networks".
    
    Idea to investigate: i. Using graph net or graphical neural network for belief updates, is it possible to train one graph net, such that it take factor graph in and output factor graph with belief converged already?
    
    ii. using graph net, especially the GMNN, solves HW's symbol detection problem. Pilot symbols as labeled data, rest detection rely on the inference of semi-supervised learning.

2.  alpha belief propagation with GAN ?
    
    Reference:
    
    > [Adversarial Message Passing For Graphical Models](https://arxiv.org/abs/1612.05048)
    
    > [Training Generative Adversarial Networks from Incomplete Observations using Factorised Discriminators](https://arxiv.org/pdf/1905.12660.pdf)

3.  RENN for MAP problem?

More reference:

> [Scarselli, 2009, The graph neural network model](https://persagen.com/files/misc/scarselli2009graph.pdf)


<a id="orgec5896d"></a>

### Variational mehtods

> NIPS, Tutorial 2016, [Variational Inference](https://media.nips.cc/Conferences/2016/Slides/6199-Slides.pdf)


<a id="orgf23e0a4"></a>

## Learning of Graphical Models


<a id="orga2306b3"></a>

### Parameter Learning

1.  Learning graphical model parameters by approximate inference
    
    [Learning Graphical Model Parameters with Approximate Marginal Inference](https://ieeexplore.ieee.org/abstract/document/6420841)
    
    [Bethe Learning of Conditional Random Fields via MAP Decoding](https://arxiv.org/abs/1503.01228)

2.  Learning of MRF with neural networks
    
    > Wiseman and Kim, 2019, [Amortized Bethe Free Energy Minimization for Learning MRFs](https://papers.nips.cc/paper/9687-amortized-bethe-free-energy-minimization-for-learning-mrfs.pdf)
    
    > Kuleshov and Ermon, 2017, [Neural Variational Inference and Learning in Undirected Graphical Models](https://arxiv.org/abs/1711.02679)

1.  Learning of Directed Graphs
    
    > Chongxuan Li, 2020, [To Relieve Your Headache of Training an MRF, Take AdVIL](https://arxiv.org/abs/1901.08400)
    
    > Mnih and Gregor, 2014, [Neural Variational Inference and Learning in Belief Networks](https://arxiv.org/abs/1402.0030)
    
    > NIPS, Tutorial 2016, [Variational Inference](https://media.nips.cc/Conferences/2016/Slides/6199-Slides.pdf)


<a id="orgb667a11"></a>

### Structure/graph Learning

Todo: add refereces, RCN, AndOr graphs etc.


<a id="orga75a8d2"></a>

## Sparks


<a id="org4d9f97a"></a>

### Applying RENN for Conditional Random Field

1.  RENN for conditional RBM

2.  RENN for high-order HMM


<a id="orgbe2f888"></a>

### Hierarchical model: RCN + RENN


<a id="org16f5903"></a>

### HMM+GMs

1.  Apply to acoustic signal detection
    
    1.1 <font color="green">Derivatives Done</font>
    
    1.2 <font color="orange">Experiments underoging</font>
    
    1.3 [A Hidden Markov Model Variant for Sequence Classification](https://www.ijcai.org/Proceedings/11/Papers/203.pdf), variant classification tricks from HMM

2.  Apply to planning


<a id="org792e6b7"></a>

### HRCF for bio-medical application

1.  Pure tractable density functions + BP or RENN

2.  NN based emission probability + BP or RENN


<a id="org287cd9e"></a>

### flow+EM

1.  <font color="green"> EM guides mixture building of probabilistic model</font>
2.  How about using DCT/wavelet transform for our generative models?
3.  Shall try EM with Ordinary Differential Equation?


<a id="orgd92a434"></a>

### flow-model based classification

1.  <font color="green"> Maximum likelihood estimation done</font>
2.  reform input x and class label as [x, c], the send [x, c] to go through invertible flow-model. To maximize the mutual information between x and c


<a id="org0c2a08a"></a>

# Application of Graphical Models and Teaching


<a id="org8c145d4"></a>

## With HW

1.  >  [Donoho, 2010, Message passingfor compressed sensing](https://ieeexplore.ieee.org/document/5503193) and [Donoho, 2009, AMP for compressed sensing](https://arxiv.org/abs/0907.3574)
    
    > [Rangan, 2018, VAMP](https://arxiv.org/abs/1610.03082)

2.  OAMPNet, MMNet, [Adaptive Neural Signal Detection for Massive MIMO](https://arxiv.org/abs/1906.04610)
    
    How about to bring the VAMP (or generalized AMP by Rangan, 2012, prefer VAMP) into OAMPNet? Better than OAMPNet?

3.  SSFN seems to be able as candidate ITERATIVE detection method for MIMO as MMNet.

4.  Use RENN with and without readout net for MIMO detection

5.  If NN based method does not give very good performance on non-binary support cases, may just use the equivalence condition to convert the non-binary MRF binary MRF, solve the problem and cast the solution back.


<a id="orgbf3885c"></a>

# GANs


<a id="orga1cd009"></a>

## Redefine the target of GAN

1.  Try to define the targets of GAN as combinational conditionals distributions/combination of sample logics instead of joint decisions. Then the complex decision can be made by combination of simple logics.


<a id="org14eece6"></a>

## Current design EOT-GAN help robust classification design/large-scale imaginary classification/ semi-supervised learning?


<a id="org3b7220a"></a>

## Coverlutional lay bounded design, for lipschitz property

<del>First step for the reference(random circulate coverlutional matrix), see if useful</del>


<a id="org12cb0c4"></a>

## Discrete GAN or RBM or Autoencoder


<a id="orgbc392cd"></a>

## OT incremental building


<a id="orga969cae"></a>

### Additive Building:

[Discussion with Baptiste on additive para \(\gamma\), remaining question: how to optimize Q](images/GAN/incremental_building/P81115-111945.jpg)

<del>**\*** How about using barycenters model to do the incremental building?</del>


<a id="orgebef844"></a>

### seems one-to-mutiple barycenter computation is a base-line of mixture


<a id="org857136f"></a>

### Concatenating Building

OT is equivalent or leq than autoencoder structured autoencoder:

1.  consider the concatenation/progressive adding more mapping. See if each concatenation has complexity reduction, error bounding&#x2026; \(W(P_X,P_Y) \leq W(P_X, G1(Z1)) \leq W(X, G2(Z2)) \leq \cdots\)
2.  \(W(P_X,P_Y) \leq W(P_X, G2(Z2)) \leq W(G1(Z1), G2(Z2))\), i.e. do alternative mapping twice, what is the benefits of solving \(W(G1(Z1), G2(Z2))\).
3.  Consider adaboosing for condition of going deeper
4.  cite:NIPS2017<sub>7126</sub> use beta-divergence for each mixture component generator optimization. This allow a training generator to omit tail samples during training. Empirical samples that are not captured during previous generator training will be put more weight and become high-weight samples for next generator training.

5.  Use Gaussian random encoder, benefit: the latent divergence with prior (gaussian prior) can be analytically studied.


<a id="orgecb2b03"></a>

## User GAN to learn context noise

User GAN to learn context noise distribution instead of signal itself. Then apply learned noise to signal. 


<a id="org7775f7c"></a>

## HMM+GMM+OT/GAN

HMM+GMM models perform good enough in clean/non-noise scenarios/context. But in heavy-noise scenario, it works poor.
\(P_X\), the signal distribution itself or the feature distribution after MFCC, is not GMM but is modeled as GMM. So, how about learning the transformation \(P_X \rightarrow Q_X\) to make \(Q_X\) is Mixture Gaussian.

[Discussion with Saikat on application of OT to HMM](images/GAN/hmm/hmm_ot.jpg)


<a id="orga00a4dc"></a>

## Using EOT for Coreset finding or generating

1.  Using EOT to compute coreset
2.  Using EOT to train generative model to generate coreset. It is ok for mode collapse.
3.  How about using beta-divergence for coreset problem?


<a id="orgdc1bf51"></a>

## convex duality (Farnia):


<a id="orga773456"></a>

# Robustness


<a id="orged64c7e"></a>

## ROBUST probability learning

1.  try to use measure that is robust to noisy samples or outlier, such as beta-estimation, beta divergence, useVAE do the generator may solve the probability of g


<a id="org181cf6b"></a>

## minmax problem:

1.  robust linear clissifier + feature mapping: for robustness against noise and attack

2.  Minmax learning for remote prediction, find the connection to anto-encoder and GAN, do information bottleneck work, try to see (mutual information estimation in estimation information flow in DNN helps or not)
3.  Find OT distance in pix space and feature space, what is the condition for the eqvilence? then use ot upper bound to try the minmax problem&#x2026; target: extend the problem into general case, not just linear dicision rule.


<a id="org698caa0"></a>

## [Discussion with Hossein](images/robustness/adversarial_sample.jpg)


<a id="org62c3c41"></a>

# Interpretable Methods and Explanations

A general framework for learning different kinds of explanations for black box algorithms is proposed and experimentedcite:fong2017interpretable.
Google's interpretability tool: [lucid@github](https://github.com/tensorflow/lucid).

1.  Use lucid to study the inference propagation over CNN or its variants
2.  What is the relationship between salience map and neural network sparsity.
    
    cite:fong2017interpretable proposes two test rules for leanring/inference algorithms: 1. classification itself 2. rotation perturbation on input. Regulation formulas are proposed. Deletion, noise and bluring on input images are experimented and discussed.


<a id="orgf428af7"></a>

## Bayesian Learning


<a id="org2f7dd41"></a>

# Record of reading


<a id="orge7f1e52"></a>

## Causal Inference

cite:pearl2018theoretical explains the theoretical limits of current
state-of-art machine learning that are mostly based on statistical methods.


<a id="org5df7c51"></a>

# Reference

bibliographystyle:unsrt
bibliography:mLearningMemo.bib

