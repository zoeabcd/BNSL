# BNSL
QI+X course project in 2023
**Task:** Develop a quantum algorithm approach to the Bayesian Network Structure Learning Problem.

## Introduction
To understand causal relations between the measured features of a population, researchers have intro-
duced graphical models such as Bayesian Networks to decompose the entire joint probability distribu-
tion of the data into the salient conditional dependences between the data. A real world motivation
for such model to study the impacts of overdoses in nutritional supplements is provided in Appendix
Given such conditional independences, learning algorithms optimize the probability distributions to
fit the observations from a given dataset.

However, the assumed conditional independences mostly follow human-designed heuristics, because
they are hard to learn themselves. Theoretically, there are exponentially many possible structures with
different implications on the conditional independences innate to the data. Because this structure
learning problem has received quite little attention in research so far, we dedicated our course project
to the potential of quantum algorithms to provide a computational speed-up both on current noisy
intermediate-scale quantum (NISQ) devices as on more powerful quantum computers in the future.

## $H_{score}$ calculation
About the calculation of H_score: (Zoe 23/12/21)
the K2 function in score-based method (whose goal is maximize the score) is
$$s(G,D) = \sum_{i = 1}^{n}s_i(G,D)$$
where
$$s_i(G,D) = \sum_{j = 1}^{q_i}\bigg[\log(r_i - 1)! - \log(N_{ij} + r_i - 1)! + \sum_{k=1}^{r_i}\log(N_{ijk}!)\bigg].$$
And $N_{ijk}$ means the number of datas when node $i$ is in status $k$ and it's parents in $G$ are in status $j$. In our example $r_i = 2\forall i$, which means
$$s_i(G,D) = -\sum_{j = 1}^{q_i}\bigg[\log(N_{ij} + 1)! - \sum_{k=1}^{r_i}\log(N_{ijk}!)\bigg].$$
When we use the score to build up hamiltonian H_score, the score should be minimized instead. Consider
$$S_i(K) = -s_i(K \text{ is i's parents}, D) = \sum_{j = 1}^{q_i}\bigg[\log(N_{ij} + 1)! - \sum_{k=1}^{r_i}\log(N_{ijk}!)\bigg]$$
and construct
$$H^{i}_{score}(d_i) = S_i(G(d_i), D) = \sum_{J\subset\{1,\dots,n\}/\{i\}}\big( \omega_i(J)\Pi_{j\in J}d_{ji} \big)$$
where
$$\omega_i(J) = \sum_{l=0}^{|J|}(-1)^{|J|-l}\sum_{K\subset J, |K| = l}S_i(K)$$
