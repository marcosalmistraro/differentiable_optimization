This repository explores several concepts linked to **constrained optimization**, with a specific focus on their applicability in the context of neural networks. The motivation for this work stems from the **OptNet** approach, proposing a closed-form solution for the back-propagation of QP-only parameters.

The simulations available here provide a flexible tool for implementing differentiale optimization with regards to generally non-linear problem instances. In order to do demonstrate the functionality of the approach, I have developed an autoencoder architecture where the decoding stage is constituted by a parameterized non-linear optimization task. The model's architecture is described by the following diagram.

![Model](https://github.com/marcosalmistraro/differentiable_optimization/blob/main/imgs/FIG_5.png)

As for gradient back-propagation, this is done by means of a local quadratic approximation of the objective function, as well as linear approximations of the constraints. The rationale employed for structuring the `PyTorch` graph is presented below here.

![Model](https://github.com/marcosalmistraro/differentiable_optimization/blob/main/imgs/FIG_8.png)
