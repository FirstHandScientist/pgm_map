
# Table of Contents

1.  [Book and Monograph:](#orgaa726f5)
2.  [Inference and Learning of PGMs](#orgda36ec6)
    1.  [Inference methods and techniques](#org84564a6)
        1.  [Application Consideration](#orga2413ea)
        2.  [Classical Inference Methods](#org43c9e4b)
    2.  [Neural network based methods](#orgd19ee4d)
        1.  [Graphical Neural Networks](#org8661587)
        2.  [Learning messages](#orgd867ca6)
        3.  [Variational methods](#org7ce1012)
        4.  [Neural density function estimation](#org657febd)
    3.  [Learning of Graphical Models](#org3a8c1dc)
        1.  [Parameter Learning](#org95447f8)
3.  [PGM and Decision-making in Dynamic Systems](#org1480a13)
    1.  [Courses](#orgc376de3)
4.  [In connecting with others](#orgf1a0560)
    1.  [GANs](#orgd99e34b)
    2.  [Discrete GAN or RBM or Autoencoder](#orgc543275)
    3.  [Optimal Transport (likelihood-free learning)](#org817015e)



<a id="orgaa726f5"></a>

# Book and Monograph:

Book CACHE:

Komodakis etc, 2016, [(Hyper)-Graphs Inference through Convex Relaxations and Move Making Algorithms: Contributions and Applications in Artificial Vision](https://www.nowpublishers.com/article/Details/CGV-066)

Bogdan Savchynskyy, 2019, [Discrete Graphical Models &#x2013; An Optimization Perspective](file:///home/dong/Documents/my_eBooks/mLearning/discrete_graphical_models_an_optimization_perspective.pdf) < 

Angelino, 2016, [Patterns of Scalable Bayesian Inference](https://www.nowpublishers.com/article/Details/MAL-052)

Nowozin, 2011, [Structured Learning and Prediction in Computer Vision](http://www.nowozin.net/sebastian/papers/nowozin2011structured-tutorial.pdf) <

Books or Monograph:

Kingma and Welling, 2019, [An Introduction to Variational Autoencoders](file:///home/dong/Documents/my_eBooks/mLearning/introduction_to_variatinal_autoencoders.pdf) 

Sutton, 2010, [An Introduction to Conditional Random Fields](https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf)

Wainwright, 2008, [Graphical Models, Exponential Families, and Variational Inference](file:///home/dong/Documents/my_eBooks/mLearning/graphical_models_wainwright.pdf)

Koller, 2009, [Probabilistic graphical models: principles and techniques](file:///home/dong/Documents/my_eBooks/mLearning/probabilistic_graphical_models_principles_techniques.pdf)

Mark Rowland, 2018, [Structure in Machine Learning: Graphical Models and Monte Carlo Methods](https://www.repository.cam.ac.uk/handle/1810/287479)

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


<a id="orgda36ec6"></a>

# Inference and Learning of PGMs


<a id="org84564a6"></a>

## Inference methods and techniques


<a id="orga2413ea"></a>

### Application Consideration

1.  What will I get by applying the RNN's mean field explanation to RNN augmented Kalman filter?
    
    Ref1: [Satorras, 2019, Combining Generative and Discriminative Models for Hybrid Inference](https://papers.nips.cc/paper/9532-combining-generative-and-discriminative-models-for-hybrid-inference.pdf)
    
    Ref2: [Zheng, 2019, Conditional Random Fields as Recurrent Neural Networks](https://arxiv.org/pdf/1502.03240.pdf)
    
    Ref3: [Krahenbuhl, 2011, Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials](https://arxiv.org/abs/1210.5644)


<a id="org43c9e4b"></a>

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


<a id="orgd19ee4d"></a>

## Neural network based methods


<a id="org8661587"></a>

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

More reference:

[Scarselli, 2009, The graph neural network model](https://persagen.com/files/misc/scarselli2009graph.pdf)


<a id="orgd867ca6"></a>

### Learning messages

Lin, 2015, [Deeply Learning the Messages in Message Passing Inference](http://papers.nips.cc/paper/5791-deeply-learning-the-messages-in-message-passing-inference.pdf)


<a id="org7ce1012"></a>

### Variational methods

NIPS, Tutorial 2016, [Variational Inference](https://media.nips.cc/Conferences/2016/Slides/6199-Slides.pdf)

Kingma and Welling, 2014, Autoencoder: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

Kuleshov and Ermon, 2017, NVIL: [Neural Variational Inference and Learning in Undirected Graphical Models](https://arxiv.org/abs/1711.02679)

Li, etc, 2020, AdVIL: [To Relieve Your Headache of Training an MRF, Take AdVIL](https://arxiv.org/abs/1901.08400)

Lazaro-Gredilla, 2019 (Vicarious AI), [Learning undirected models via query training](https://arxiv.org/abs/1912.02893)

Sobolev and Vetrov, 2019, (Section 3 gives interesting discussion on literature works) [Importance Weighted Hierarchical Variational Inference](http://papers.nips.cc/paper/8350-importance-weighted-hierarchical-variational-inference)

Kingma, et al, 2016, [Improved Variational Inference with Inverse Autoregressive Flow](https://papers.nips.cc/paper/6581-improved-variational-inference-with-inverse-autoregressive-flow)

Rezende, Mohamed, 2015, [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770)


<a id="org657febd"></a>

### Neural density function estimation

Chen et al, 2018, ODE: [Neural Ordinary Differential Equations](https://papers.nips.cc/paper/7892-neural-ordinary-differential-equations)

Kingma, Dhariwal, 2018, [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)

Dinh, Sohl-Dickstein, Bengio, 2017, [Density Estimation using Real NVP](https://arxiv.org/pdf/1605.08803.pdf)

Dinh, Krueger, Bengio, 2014, [NICE: Non-linear independent component estimation](https://arxiv.org/abs/1410.8516)

Inverse autoregreeeive flow as in previous subsection.


<a id="org3a8c1dc"></a>

## Learning of Graphical Models


<a id="org95447f8"></a>

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


<a id="org1480a13"></a>

# PGM and Decision-making in Dynamic Systems

-   Sutton, Barto, 2018, [Reinforcement learning (2ed edition)](https://github.com/FirstHandScientist/Reinforcement-Learning-2nd-Edition-by-Sutton-Exercise-Solutions)

-   Bubeck, Cesa-Bianchi, 2012, [Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/SurveyBCB12.pdf), Now publisher, Foundations and trends in machine learning

-   Ziebart, 2010, [Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy](https://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf)

-   Levin, 2018, [Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/abs/1805.00909)

-   Haarnoja, et al 2017, [Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/pdf/1702.08165.pdf)

-   Martin L. Puterman, 2014, Markov Decision Processes: Discrete Stochastic Dynamic Programming

-   Szepesvari, 2009, [Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs-lecture.pdf)


<a id="orgc376de3"></a>

## Courses

-   [Reinforcement Learning (UCL)](https://www.davidsilver.uk/teaching/)
-   [Deep Reinforcement Learning (CS285)](http://rail.eecs.berkeley.edu/deeprlcourse/)
-   [Advanced Deep Learning & Reinforcement Learning](https://www.youtube.com/playlist?list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs)


<a id="orgf1a0560"></a>

# In connecting with others

-   [Repos: Advanced-variational-inference-paper](https://github.com/otokonoko8/implicit-variational-inference)

-   [Repos: Deep-Bayesian-nonparametrics-papers](https://github.com/otokonoko8/deep-Bayesian-nonparametrics-papers)


<a id="orgd99e34b"></a>

## GANs


<a id="orgc543275"></a>

## Discrete GAN or RBM or Autoencoder


<a id="org817015e"></a>

## Optimal Transport (likelihood-free learning)

&#x2026; Matthed Thorpe, 2018, [Introduction to Optimal Transport](http://www.math.cmu.edu/~mthorpe/OTNotes)
&#x2026; Peyre, Cuturi, 2018, Computational Optimal Transport, [Codes and slides for OT](https://optimaltransport.github.io/resources/)

