
# Table of Contents

1.  [Book and Thesis:](#org900b041)
2.  [Inference and Learning of PGMs](#orgb9e78b8)
    1.  [Inference methods and techniques](#org109157d)
        1.  [Application Consideration](#org8a2a84f)
        2.  [Classical Inference Methods](#org2801ae0)
    2.  [Neural network based methods](#org25834a2)
        1.  [Graphical Neural Networks](#orga24d9fe)
        2.  [Learning messages](#org4e653cd)
        3.  [Variational mehtods](#org30aebfd)
    3.  [Learning of Graphical Models](#orgdd1af7a)
        1.  [Parameter Learning](#org7195232)
        2.  [Structure/graph Learning](#org74025f9)
    4.  [Sparks](#orgf5c95a5)
        1.  [Applying RENN for Conditional Random Field](#org049e070)
        2.  [Hierarchical model: RCN + RENN](#org7f6946a)
        3.  [HMM+GMs](#orgda3bb38)
        4.  [HRCF for bio-medical application](#orga91b1e3)
        5.  [flow+EM](#org7b45191)
        6.  [flow-model based classification](#org2f69a3c)
3.  [Application of Graphical Models and Teaching](#org5f6da32)
    1.  [With HW](#org1493765)
4.  [GANs](#org8b2e138)
    1.  [Redefine the target of GAN](#org4fc808a)
    2.  [Current design EOT-GAN help robust classification design/large-scale imaginary classification/ semi-supervised learning?](#org29b5542)
    3.  [Coverlutional lay bounded design, for lipschitz property](#orga5b23b7)
    4.  [Discrete GAN or RBM or Autoencoder](#org81a179b)
    5.  [OT incremental building](#org22e95e7)
        1.  [Additive Building:](#org344b4be)
        2.  [seems one-to-mutiple barycenter computation is a base-line of mixture](#orgfaf508b)
        3.  [Concatenating Building](#orgd30e3b1)
    6.  [User GAN to learn context noise](#org9c6638a)
    7.  [HMM+GMM+OT/GAN](#orgc2cf903)
    8.  [Using EOT for Coreset finding or generating](#org8018bea)
    9.  [convex duality (Farnia):](#org5c4b6a2)
5.  [Robustness](#org236111f)
    1.  [ROBUST probability learning](#org980dc47)
    2.  [minmax problem:](#org6fc1acc)
    3.  [Discussion with Hossein](#orga6f4363)
6.  [Interpretable Methods and Explanations](#org2e88eb4)
    1.  [Bayesian Learning](#org95f1ae3)
7.  [Record of reading](#org470c5d7)
    1.  [Causal Inference](#orgdb9f549)
8.  [Reference](#org12a41d0)



<a id="org900b041"></a>

# Book and Thesis:

Book CACHE:

Komodakis etc, 2016, [(Hyper)-Graphs Inference through Convex Relaxations and Move Making Algorithms: Contributions and Applications in Artificial Vision](https://www.nowpublishers.com/article/Details/CGV-066)

Bogdan Savchynskyy, 2019, [Discrete Graphical Models &#x2013; An Optimization Perspective](file:///home/dong/Documents/my_eBooks/mLearning/discrete_graphical_models_an_optimization_perspective.pdf) < 

Kingma and Welling, 2019, [An Introduction to Variational Autoencoders](file:///home/dong/Documents/my_eBooks/mLearning/introduction_to_variatinal_autoencoders.pdf) <

Angelino, 2016, [Patterns of Scalable Bayesian Inference](https://www.nowpublishers.com/article/Details/MAL-052)

Nowozin, 2011, [Structured Learning and Prediction in Computer Vision](http://www.nowozin.net/sebastian/papers/nowozin2011structured-tutorial.pdf) <

Books:

Sutton, 2010, [An Introduction to Conditional Random Fields](https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf)

Wainwright, 2008, [Graphical Models, Exponential Families, and Variational Inference](file:///home/dong/Documents/my_eBooks/mLearning/graphical_models_wainwright.pdf)

Koller, 2009, [Probabilistic graphical models: principles and techniques](file:///home/dong/Documents/my_eBooks/mLearning/probabilistic_graphical_models_principles_techniques.pdf)

**Thesis CACHE**:

Mark Rowland, 2018, [Structure in Machine Learning: Graphical Models and Monte Carlo Methods](https://www.repository.cam.ac.uk/handle/1810/287479)

**Thesis**:

Yingzhen Li, 2018, [Approximate Inference: New Visions](https://www.repository.cam.ac.uk/handle/1810/277549)

Adrian Weller, 2014, [Methods for Inference in Graphical Models](http://mlg.eng.cam.ac.uk/adrian/phd_FINAL.pdf)

Ihler, Alexander:

Lou, Qi, 2018, [Anytime Approximate Inference in Graphical Models](https://escholarship.org/uc/item/7sc0m97f)

Ping, Wei, 2016, [Learning and Inference in Latent Variable Graphical Models](https://escholarship.org/uc/item/7q90z4b5)

Forouzan, Sholeh, 2015, [Approximate Inference in Graphical Models](https://escholarship.org/uc/item/5n4733cz)

Qiang, Liu, 2014, [Reasoning and Decisions in Probabilistic Graphical Models - A Unified Framework](https://escholarship.org/uc/item/92p8w3xb)

Minka:

Yuan Qi, 2005, [Extending Expectation Propagation for Graphical Models](https://affect.media.mit.edu/pdfs/05.qi-phd.pdf)

Thomas P Minka, 2001, [A family of algorithms for approximate Bayesian inference](https://tminka.github.io/papers/ep/minka-thesis.pdf)


<a id="orgb9e78b8"></a>

# Inference and Learning of PGMs


<a id="org109157d"></a>

## Inference methods and techniques


<a id="org8a2a84f"></a>

### Application Consideration

1.  What will I get by applying the RNN's mean field explanation to RNN augmented Kalman filter?
    
    Ref1: [Satorras, 2019, Combining Generative and Discriminative Models for Hybrid Inference](https://papers.nips.cc/paper/9532-combining-generative-and-discriminative-models-for-hybrid-inference.pdf)
    
    Ref2: [Zheng, 2019, Conditional Random Fields as Recurrent Neural Networks](https://arxiv.org/pdf/1502.03240.pdf)
    
    Ref3: [Krahenbuhl, 2011, Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials](https://arxiv.org/abs/1210.5644)

2.  Can I interpret tree-reweighted RENN as a mixture method of MF, LoopyBP and GBP (or even including TWBP) after convexcify its corresponding hypergraph?


<a id="org2801ae0"></a>

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


<a id="org25834a2"></a>

## Neural network based methods


<a id="orga24d9fe"></a>

### Graphical Neural Networks

1.  literature development path
    
    Half-automated message passing, [Learning to Pass Expectation Propagation Messages](https://papers.nips.cc/paper/5070-learning-to-pass-expectation-propagation-messages.pdf) , message-level automation
    
    Training neural network to do message passing, [Inference in Probabilistic Graphical Models by Graph Neural Networks](https://arxiv.org/abs/1803.07710) , train NN for message updates, and also NN for mapping messages to estimations. A good property observed in the work, trained NNs can be used for different factor graphs with different potentials and structures
    Same track, [GMNN: Graph Markov Neural Networks](https://arxiv.org/abs/1905.06214), semi-supervised learning, EM is used for training.
    
    More generalized computation power: [Graph Net](https://github.com/deepmind/graph_nets), A graph network takes a graph as input and returns a graph as output. The input graph has edge- (E ), node- (V ), and global-level (u) attributes. The output graph has the same structure, but updated attributes. Graph networks are part of the broader family of "graph neural networks".
    
    Idea to investigate: i. Using graph net or graphical neural network for belief updates, is it possible to train one graph net, such that it take factor graph in and output factor graph with belief converged already?
    
    ii. using graph net, especially the GMNN, solves HW's symbol detection problem. Pilot symbols as labeled data, rest detection rely on the inference of semi-supervised learning.

2.  alpha belief propagation with GAN ?
    
    Reference:
    
    [Adversarial Message Passing For Graphical Models](https://arxiv.org/abs/1612.05048)
    
    [Training Generative Adversarial Networks from Incomplete Observations using Factorised Discriminators](https://arxiv.org/pdf/1905.12660.pdf)

3.  RENN for MAP problem?

More reference:

[Scarselli, 2009, The graph neural network model](https://persagen.com/files/misc/scarselli2009graph.pdf)


<a id="org4e653cd"></a>

### Learning messages

Lin, 2015, [Deeply Learning the Messages in Message Passing Inference](http://papers.nips.cc/paper/5791-deeply-learning-the-messages-in-message-passing-inference.pdf)


<a id="org30aebfd"></a>

### Variational mehtods

NIPS, Tutorial 2016, [Variational Inference](https://media.nips.cc/Conferences/2016/Slides/6199-Slides.pdf)

Kingma and Welling, 2014, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

Kuleshov and Ermon, 2017, [Neural Variational Inference and Learning in Undirected Graphical Models](https://arxiv.org/abs/1711.02679)

Li, etc, 2020, [To Relieve Your Headache of Training an MRF, Take AdVIL](https://arxiv.org/abs/1901.08400)

Lazaro-Gredilla, 2019 (Vicarious AI), [Learning undirected models via query training](https://arxiv.org/abs/1912.02893)


<a id="orgdd1af7a"></a>

## Learning of Graphical Models


<a id="org7195232"></a>

### Parameter Learning

1.  Learning graphical model parameters by approximate inference
    
    Domke, 2013, [Learning Graphical Model Parameters with Approximate Marginal Inference](https://ieeexplore.ieee.org/abstract/document/6420841)
    
    Tang, 2015, [Bethe Learning of Conditional Random Fields via MAP Decoding](https://arxiv.org/abs/1503.01228)
    
    You Lu, 2019, [Block Belief Propagation for Parameter Learning in Markov Random Fields](https://www.aaai.org/ojs/index.php/AAAI/article/view/4357)
    
    Hazan, 2016, [Blending Learning and Inference in Conditional Random Fields](http://www.jmlr.org/papers/v17/13-260.html)

2.  Learning of MRF with neural networks
    
    Wiseman and Kim, 2019, [Amortized Bethe Free Energy Minimization for Learning MRFs](https://papers.nips.cc/paper/9687-amortized-bethe-free-energy-minimization-for-learning-mrfs.pdf)
    
    Kuleshov and Ermon, 2017, [Neural Variational Inference and Learning in Undirected Graphical Models](https://arxiv.org/abs/1711.02679)

1.  Learning of Directed Graphs
    
    Chongxuan Li, 2020, [To Relieve Your Headache of Training an MRF, Take AdVIL](https://arxiv.org/abs/1901.08400)
    
    Mnih and Gregor, 2014, [Neural Variational Inference and Learning in Belief Networks](https://arxiv.org/abs/1402.0030)
    
    NIPS, Tutorial 2016, [Variational Inference](https://media.nips.cc/Conferences/2016/Slides/6199-Slides.pdf)


<a id="org74025f9"></a>

### Structure/graph Learning

Todo: add refereces, RCN, AndOr graphs etc.


<a id="orgf5c95a5"></a>

## Sparks


<a id="org049e070"></a>

### Applying RENN for Conditional Random Field

1.  RENN for conditional RBM

2.  RENN for high-order HMM


<a id="org7f6946a"></a>

### Hierarchical model: RCN + RENN


<a id="orgda3bb38"></a>

### HMM+GMs

1.  Apply to acoustic signal detection
    
    1.1 <font color="green">Derivatives Done</font>
    
    1.2 <font color="orange">Experiments underoging</font>
    
    1.3 [A Hidden Markov Model Variant for Sequence Classification](https://www.ijcai.org/Proceedings/11/Papers/203.pdf), variant classification tricks from HMM

2.  Apply to planning


<a id="orga91b1e3"></a>

### HRCF for bio-medical application

1.  Pure tractable density functions + BP or RENN

2.  NN based emission probability + BP or RENN


<a id="org7b45191"></a>

### flow+EM

1.  <font color="green"> EM guides mixture building of probabilistic model</font>
2.  How about using DCT/wavelet transform for our generative models?
3.  Shall try EM with Ordinary Differential Equation?


<a id="org2f69a3c"></a>

### flow-model based classification

1.  <font color="green"> Maximum likelihood estimation done</font>
2.  reform input x and class label as [x, c], the send [x, c] to go through invertible flow-model. To maximize the mutual information between x and c


<a id="org5f6da32"></a>

# Application of Graphical Models and Teaching


<a id="org1493765"></a>

## With HW

1.  >  [Donoho, 2010, Message passingfor compressed sensing](https://ieeexplore.ieee.org/document/5503193) and [Donoho, 2009, AMP for compressed sensing](https://arxiv.org/abs/0907.3574)
    
    > [Rangan, 2018, VAMP](https://arxiv.org/abs/1610.03082)

2.  OAMPNet, MMNet, [Adaptive Neural Signal Detection for Massive MIMO](https://arxiv.org/abs/1906.04610)
    
    How about to bring the VAMP (or generalized AMP by Rangan, 2012, prefer VAMP) into OAMPNet? Better than OAMPNet?

3.  SSFN seems to be able as candidate ITERATIVE detection method for MIMO as MMNet.

4.  Use RENN with and without readout net for MIMO detection

5.  If NN based method does not give very good performance on non-binary support cases, may just use the equivalence condition to convert the non-binary MRF binary MRF, solve the problem and cast the solution back.


<a id="org8b2e138"></a>

# GANs


<a id="org4fc808a"></a>

## Redefine the target of GAN

1.  Try to define the targets of GAN as combinational conditionals distributions/combination of sample logics instead of joint decisions. Then the complex decision can be made by combination of simple logics.


<a id="org29b5542"></a>

## Current design EOT-GAN help robust classification design/large-scale imaginary classification/ semi-supervised learning?


<a id="orga5b23b7"></a>

## Coverlutional lay bounded design, for lipschitz property

<del>First step for the reference(random circulate coverlutional matrix), see if useful</del>


<a id="org81a179b"></a>

## Discrete GAN or RBM or Autoencoder


<a id="org22e95e7"></a>

## OT incremental building


<a id="org344b4be"></a>

### Additive Building:

[Discussion with Baptiste on additive para \(\gamma\), remaining question: how to optimize Q](images/GAN/incremental_building/P81115-111945.jpg)

<del>**\*** How about using barycenters model to do the incremental building?</del>


<a id="orgfaf508b"></a>

### seems one-to-mutiple barycenter computation is a base-line of mixture


<a id="orgd30e3b1"></a>

### Concatenating Building

OT is equivalent or leq than autoencoder structured autoencoder:

1.  consider the concatenation/progressive adding more mapping. See if each concatenation has complexity reduction, error bounding&#x2026; \(W(P_X,P_Y) \leq W(P_X, G1(Z1)) \leq W(X, G2(Z2)) \leq \cdots\)
2.  \(W(P_X,P_Y) \leq W(P_X, G2(Z2)) \leq W(G1(Z1), G2(Z2))\), i.e. do alternative mapping twice, what is the benefits of solving \(W(G1(Z1), G2(Z2))\).
3.  Consider adaboosing for condition of going deeper
4.  cite:NIPS2017<sub>7126</sub> use beta-divergence for each mixture component generator optimization. This allow a training generator to omit tail samples during training. Empirical samples that are not captured during previous generator training will be put more weight and become high-weight samples for next generator training.

5.  Use Gaussian random encoder, benefit: the latent divergence with prior (gaussian prior) can be analytically studied.


<a id="org9c6638a"></a>

## User GAN to learn context noise

User GAN to learn context noise distribution instead of signal itself. Then apply learned noise to signal. 


<a id="orgc2cf903"></a>

## HMM+GMM+OT/GAN

HMM+GMM models perform good enough in clean/non-noise scenarios/context. But in heavy-noise scenario, it works poor.
\(P_X\), the signal distribution itself or the feature distribution after MFCC, is not GMM but is modeled as GMM. So, how about learning the transformation \(P_X \rightarrow Q_X\) to make \(Q_X\) is Mixture Gaussian.

[Discussion with Saikat on application of OT to HMM](images/GAN/hmm/hmm_ot.jpg)


<a id="org8018bea"></a>

## Using EOT for Coreset finding or generating

1.  Using EOT to compute coreset
2.  Using EOT to train generative model to generate coreset. It is ok for mode collapse.
3.  How about using beta-divergence for coreset problem?


<a id="org5c4b6a2"></a>

## convex duality (Farnia):


<a id="org236111f"></a>

# Robustness


<a id="org980dc47"></a>

## ROBUST probability learning

1.  try to use measure that is robust to noisy samples or outlier, such as beta-estimation, beta divergence, useVAE do the generator may solve the probability of g


<a id="org6fc1acc"></a>

## minmax problem:

1.  robust linear clissifier + feature mapping: for robustness against noise and attack

2.  Minmax learning for remote prediction, find the connection to anto-encoder and GAN, do information bottleneck work, try to see (mutual information estimation in estimation information flow in DNN helps or not)
3.  Find OT distance in pix space and feature space, what is the condition for the eqvilence? then use ot upper bound to try the minmax problem&#x2026; target: extend the problem into general case, not just linear dicision rule.


<a id="orga6f4363"></a>

## [Discussion with Hossein](images/robustness/adversarial_sample.jpg)


<a id="org2e88eb4"></a>

# Interpretable Methods and Explanations

A general framework for learning different kinds of explanations for black box algorithms is proposed and experimentedcite:fong2017interpretable.
Google's interpretability tool: [lucid@github](https://github.com/tensorflow/lucid).

1.  Use lucid to study the inference propagation over CNN or its variants
2.  What is the relationship between salience map and neural network sparsity.
    
    cite:fong2017interpretable proposes two test rules for leanring/inference algorithms: 1. classification itself 2. rotation perturbation on input. Regulation formulas are proposed. Deletion, noise and bluring on input images are experimented and discussed.


<a id="org95f1ae3"></a>

## Bayesian Learning


<a id="org470c5d7"></a>

# Record of reading


<a id="orgdb9f549"></a>

## Causal Inference

cite:pearl2018theoretical explains the theoretical limits of current
state-of-art machine learning that are mostly based on statistical methods.


<a id="org12a41d0"></a>

# Reference

bibliographystyle:unsrt
bibliography:mLearningMemo.bib

