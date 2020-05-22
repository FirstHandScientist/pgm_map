
# Table of Contents

1.  [Book and Monograph on PGMs](#orgcb43868)
    1.  [Books / Monograph:](#orgebc8da3)
    2.  [Cached Region](#org9400507)
2.  [Inference and Learning of PGMs Papers](#org196e5fe)
    1.  [Inference methods and techniques](#org26f980b)
        1.  [Classical Inference Methods](#org3620de0)
        2.  [Improvements](#org9ded5ee)
        3.  [Application](#org2805e72)
    2.  [Neural network based methods](#org6c82a76)
        1.  [Deep learning based methods](#org2233b38)
        2.  [Variational methods](#org70e3e71)
        3.  [Neural density function estimation](#orgb90a336)
    3.  [Learning of Graphical Models](#orge806179)
        1.  [Parameter Learning](#org6726361)
3.  [PGM and Decision-making in Dynamic Systems](#orgf6c7d94)
4.  [In Connecting with Others](#org61aa681)
    1.  [Repos](#org79e9351)
    2.  [Courses](#org52ea76d)
    3.  [Optimal Transport (likelihood-free learning)](#org959a747)

****The collection of literature work on Probabilistic Graphical Models (PGMs). Source file can be found at git repository [pgm-map](https://github.com/FirstHandScientist/pgm_map).****


<a id="orgcb43868"></a>

# Book and Monograph on PGMs


<a id="orgebc8da3"></a>

## Books / Monograph:

-   Kingma and Welling, 2019, [An Introduction to Variational Autoencoders](file:///home/dong/Documents/my_eBooks/mLearning/introduction_to_variatinal_autoencoders.pdf)
-   Sutton, 2010, [An Introduction to Conditional Random Fields](https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf)
-   Wainwright, 2008, [Graphical Models, Exponential Families, and Variational Inference](file:///home/dong/Documents/my_eBooks/mLearning/graphical_models_wainwright.pdf)
-   Koller, 2009, [Probabilistic graphical models: principles and techniques](file:///home/dong/Documents/my_eBooks/mLearning/probabilistic_graphical_models_principles_techniques.pdf)
-   Mark Rowland, 2018, [Structure in Machine Learning: Graphical Models and Monte Carlo Methods](https://www.repository.cam.ac.uk/handle/1810/287479)
-   Yingzhen Li, 2018, [Approximate Inference: New Visions](https://www.repository.cam.ac.uk/handle/1810/277549)
-   Adrian Weller, 2014, [Methods for Inference in Graphical Models](http://mlg.eng.cam.ac.uk/adrian/phd_FINAL.pdf)
    
    Ihler, Alexander:

-   Lou, Qi, 2018, [Anytime Approximate Inference in Graphical Models](https://escholarship.org/uc/item/7sc0m97f)
-   Ping, Wei, 2016, [Learning and Inference in Latent Variable Graphical Models](https://escholarship.org/uc/item/7q90z4b5)
-   Forouzan, Sholeh, 2015, [Approximate Inference in Graphical Models](https://escholarship.org/uc/item/5n4733cz)
-   Qiang, Liu, 2014, [Reasoning and Decisions in Probabilistic Graphical Models - A Unified Framework](https://escholarship.org/uc/item/92p8w3xb)
    
    Minka:

-   Yuan Qi, 2005, [Extending Expectation Propagation for Graphical Models](https://affect.media.mit.edu/pdfs/05.qi-phd.pdf)
-   Thomas P Minka, 2001, [A family of algorithms for approximate Bayesian inference](https://tminka.github.io/papers/ep/minka-thesis.pdf)


<a id="org9400507"></a>

## Cached Region

-   Komodakis etc, 2016, [(Hyper)-Graphs Inference through Convex Relaxations and Move Making Algorithms: Contributions and Applications in Artificial Vision](https://www.nowpublishers.com/article/Details/CGV-066)
-   Bogdan Savchynskyy, 2019, [Discrete Graphical Models &#x2013; An Optimization Perspective](file:///home/dong/Documents/my_eBooks/mLearning/discrete_graphical_models_an_optimization_perspective.pdf)
-   Angelino, 2016, [Patterns of Scalable Bayesian Inference](https://www.nowpublishers.com/article/Details/MAL-052)
-   Nowozin, 2011, [Structured Learning and Prediction in Computer Vision](http://www.nowozin.net/sebastian/papers/nowozin2011structured-tutorial.pdf)


<a id="org196e5fe"></a>

# Inference and Learning of PGMs Papers


<a id="org26f980b"></a>

## Inference methods and techniques


<a id="org3620de0"></a>

### Classical Inference Methods

Hand-crafted message passing, BP, GBP, Tree-reweighted BP and EP, PowerEP

-   Wainwright and Willsky, 2003, [Exact MAP estimates by hypertree agreement](https://papers.nips.cc/paper/2206-exact-map-estimates-by-hypertree-agreement.pdf)
-   Wainwright et al, 2003, [tree-reweighted belief propagation algorithms and approximated ML esimation by pseudo-moment matching](http://ssg.mit.edu/group/willsky/publ_pdfs/166_pub_AISTATS.pdf)
-   Generalized BP for marginal distributions, Yedidis, et al, 2005, [Constructing free energy approximations and Generalized belief propagation algorithms](https://www.cs.princeton.edu/courses/archive/spring06/cos598C/papers/YedidaFreemanWeiss2004.pdf)
-   Tree-structured EP, Minka and Qi, [Tree-structured approximations by expectation propagation](https://tminka.github.io/papers/eptree/minka-eptree.pdf)
-   Convergence Analysis, Roosta, 2008, [Convergence Analysis of Reweighted Sum-Product Algorithms](https://ieeexplore.ieee.org/document/4599175)


<a id="org9ded5ee"></a>

### Improvements

-   Conditioning and Clamping
    -   Eaton and Ghahramani, 2009, [Choosing a Variable to Clamp](http://mlg.eng.cam.ac.uk/pub/pdf/EatGha09.pdf)
    -   Geier et al, 2015, [Locally Conditioned Belief Propagation](http://auai.org/uai2015/proceedings/papers/158.pdf)
    -   Weller and Jebara, 2014, [Clamping Variables and Approximate Inference](https://papers.nips.cc/paper/5529-clamping-variables-and-approximate-inference.pdf)

-   Linear Response. Welling and Teh, [Linear Response Algorithms for Approximate Inference in Graphical Models](https://www.ics.uci.edu/~welling/publications/papers/LR2.pdf)

-   Combining with Particle/Stochastic Methods
    -   Liu et al, 2015, [Probabilistic Variational Bounds for Graphical Models](https://papers.nips.cc/paper/5695-probabilistic-variational-bounds-for-graphical-models)
    -   Noorshams and Wainwright, 2013, [stochastic belief propagation: a low-complexity alternative to the sum-product algorithm](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6373728)

-   Mixture/multi-modal
    -   Baque et al, 2017, [Multi-Modal Mean-Fields via Cardinality-Based Clamping](http://openaccess.thecvf.com/content_cvpr_2017/papers/Baque_Multi-Modal_Mean-Fields_via_CVPR_2017_paper.pdf)
    -   Hao Xiong et al, 2019, [One-Shot Marginal MAP Inference in Markov Random Fields](http://auai.org/uai2019/proceedings/papers/19.pdf)


<a id="org2805e72"></a>

### Application

-   [Satorras, 2019, Combining Generative and Discriminative Models for Hybrid Inference](https://papers.nips.cc/paper/9532-combining-generative-and-discriminative-models-for-hybrid-inference.pdf)
-   [Zheng, 2019, Conditional Random Fields as Recurrent Neural Networks](https://arxiv.org/pdf/1502.03240.pdf)
-   [Krahenbuhl, 2011, Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials](https://arxiv.org/abs/1210.5644)


<a id="org6c82a76"></a>

## Neural network based methods


<a id="org2233b38"></a>

### Deep learning based methods

-   Karaletsos, 2016, [Adversarial Message Passing For Graphical Models](https://arxiv.org/abs/1612.05048)
-   Stoller et al, 2020, [Training Generative Adversarial Networks from Incomplete Observations using Factorised Discriminators](https://arxiv.org/pdf/1905.12660.pdf)
    
    Learning messages

-   Heess et al, [Learning to Pass Expectation Propagation Messages](https://papers.nips.cc/paper/5070-learning-to-pass-expectation-propagation-messages.pdf), half-automated message passing, message-level automation
-   Yoon et al, 2018, [Inference in Probabilistic Graphical Models by Graph Neural Networks](https://arxiv.org/abs/1803.07710)
-   Lin, 2015, [Deeply Learning the Messages in Message Passing Inference](http://papers.nips.cc/paper/5791-deeply-learning-the-messages-in-message-passing-inference.pdf)

Graphical Neural Networks

-   [GMNN: Graph Markov Neural Networks](https://arxiv.org/abs/1905.06214), semi-supervised learning, EM is used for training.
-   More generalized computation power: [Graph Net Library](https://github.com/deepmind/graph_nets), A graph network takes a graph as input and returns a graph as output.
-   Related, [Deep Graph Library](https://github.com/dmlc/dgl), for deep learning on graphs
-   Scarselli et al, 2009, [The graph neural network model](https://persagen.com/files/misc/scarselli2009graph.pdf)


<a id="org70e3e71"></a>

### Variational methods

-   NIPS tutorial 2016, [Variational Inference](https://media.nips.cc/Conferences/2016/Slides/6199-Slides.pdf)
-   Kingma and Welling, 2014, Autoencoder: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
-   Kuleshov and Ermon, 2017, NVIL: [Neural Variational Inference and Learning in Undirected Graphical Models](https://arxiv.org/abs/1711.02679)
-   Li, etc, 2020, AdVIL: [To Relieve Your Headache of Training an MRF, Take AdVIL](https://arxiv.org/abs/1901.08400)
-   Lazaro-Gredilla, 2019 (Vicarious AI), [Learning undirected models via query training](https://arxiv.org/abs/1912.02893)
-   Sobolev and Vetrov, 2019, (Section 3 gives interesting discussion on literature works) [Importance Weighted Hierarchical Variational Inference](http://papers.nips.cc/paper/8350-importance-weighted-hierarchical-variational-inference)
-   Kingma, et al, 2016, [Improved Variational Inference with Inverse Autoregressive Flow](https://papers.nips.cc/paper/6581-improved-variational-inference-with-inverse-autoregressive-flow)
-   Rezende, Mohamed, 2015, [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770)


<a id="orgb90a336"></a>

### Neural density function estimation

-   Chen et al, 2018, ODE: [Neural Ordinary Differential Equations](https://papers.nips.cc/paper/7892-neural-ordinary-differential-equations)
-   Kingma, Dhariwal, 2018, [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)
-   Dinh, Sohl-Dickstein, Bengio, 2017, [Density Estimation using Real NVP](https://arxiv.org/pdf/1605.08803.pdf)
-   Dinh, Krueger, Bengio, 2014, [NICE: Non-linear independent component estimation](https://arxiv.org/abs/1410.8516)
-   Inverse autoregreeeive flow as in previous subsection.


<a id="orge806179"></a>

## Learning of Graphical Models


<a id="org6726361"></a>

### Parameter Learning

Learning graphical model parameters by approximate inference

-   Domke, 2013, [Learning Graphical Model Parameters with Approximate Marginal Inference](https://ieeexplore.ieee.org/abstract/document/6420841)
-   Tang, 2015, [Bethe Learning of Conditional Random Fields via MAP Decoding](https://arxiv.org/abs/1503.01228)
-   You Lu, 2019, [Block Belief Propagation for Parameter Learning in Markov Random Fields](https://www.aaai.org/ojs/index.php/AAAI/article/view/4357)
-   Hazan, 2016, [Blending Learning and Inference in Conditional Random Fields](http://www.jmlr.org/papers/v17/13-260.html)

Learning of MRF with neural networks

-   Wiseman and Kim, 2019, [Amortized Bethe Free Energy Minimization for Learning MRFs](https://papers.nips.cc/paper/9687-amortized-bethe-free-energy-minimization-for-learning-mrfs.pdf)
-   Kuleshov and Ermon, 2017, [Neural Variational Inference and Learning in Undirected Graphical Models](https://arxiv.org/abs/1711.02679)

Learning of Directed Graphs

-   Chongxuan Li, 2020, [To Relieve Your Headache of Training an MRF, Take AdVIL](https://arxiv.org/abs/1901.08400)
-   Mnih and Gregor, 2014, [Neural Variational Inference and Learning in Belief Networks](https://arxiv.org/abs/1402.0030)
-   NIPS tutorial 2016, [Variational Inference](https://media.nips.cc/Conferences/2016/Slides/6199-Slides.pdf)


<a id="orgf6c7d94"></a>

# PGM and Decision-making in Dynamic Systems

-   Sutton, Barto, 2018, [Reinforcement learning (2ed edition)](https://github.com/FirstHandScientist/Reinforcement-Learning-2nd-Edition-by-Sutton-Exercise-Solutions)

-   Bubeck, Cesa-Bianchi, 2012, [Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/SurveyBCB12.pdf)

-   Ziebart, 2010, [Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy](https://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf)

-   Levin, 2018, [Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/abs/1805.00909)

-   Haarnoja, et al 2017, [Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/pdf/1702.08165.pdf)

-   Martin L. Puterman, 2014, Markov Decision Processes: Discrete Stochastic Dynamic Programming

-   Szepesvari, 2009, [Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs-lecture.pdf)


<a id="org61aa681"></a>

# In Connecting with Others


<a id="org79e9351"></a>

## Repos

-   Repos: [Advanced-variational-inference-paper](https://github.com/otokonoko8/implicit-variational-inference)
-   Repos: [Deep-Bayesian-nonparametrics-papers](https://github.com/otokonoko8/deep-Bayesian-nonparametrics-papers)


<a id="org52ea76d"></a>

## Courses

-   [Reinforcement Learning (UCL)](https://www.davidsilver.uk/teaching/)
-   [Deep Reinforcement Learning (CS285)](http://rail.eecs.berkeley.edu/deeprlcourse/)
-   [Advanced Deep Learning & Reinforcement Learning](https://www.youtube.com/playlist?list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs)


<a id="org959a747"></a>

## Optimal Transport (likelihood-free learning)

-   Matthed Thorpe, 2018, [Introduction to Optimal Transport](http://www.math.cmu.edu/~mthorpe/OTNotes)
-   Peyre, Cuturi, 2018, Computational Optimal Transport, [Codes and slides for OT](https://optimaltransport.github.io/resources/)

