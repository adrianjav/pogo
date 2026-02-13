# A Proximal One-step Geometric Orthoptimizer

This a [PyTorch](https://pytorch.org>) implementation of the POGO algorithm proposed in  [An Embarrassingly Simple Way to Optimize Orthogonal Matrices at Scale](TODO).
This is a lightweight and easy-to-use library containing the optimizer as a normal Pytorch Optimizer, and two base optimizers to choose for from now: SGD and [Vector Adam](https://arxiv.org/abs/2205.13599).

**Beware:** You should pass only orthogonal parameters to POGO and initialize them as such. See below!

## Installation

You can install `pogo` from pip by simply running

```bash
pip install pogo-torch
```

Or, if you are using `uv`, you can add it to your project with
```bash
uv add pogo-torch
```

Alternatively, you can install it directly from the repository:

```bash
pip install git+https://github.com/adrianjav/pogo
```

## Getting started

POGO is implemented as a Pytorch optimizer, so it should be quite intuitive to use. Moreover, the default parameters should work for most use cases. If that were not the case, feel free to check the docstrings and ultimately open an issue.

To initialize your parameters as orthogonal, you can either use [`torch.init.orthogonal_`](https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.orthogonal_) (but make sure it does what you intend given the shape of your parameters) or use any other initialization method and then project them to be orthogonal. For example:

```python
X = ... # My parameter
U, S, VT = torch.linalg.svd(X, full_matrices=False)
X.data = U @ VT
```

Then, you can use POGO as any other optimizer:

```python
from pogo import base, POGO

model = ...  # Ensure that your parameters are initialized as orthogonal!
optimizer = POGO(model.parameters(), base.VectorAdam(), learning_rate)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    ...
    loss.backward()
    optimizer.step()
```

By default, POGO expects matrices of the form `[num_matrices, p, n]` with $p < n$ and such that `torch.bmm(X, X)` yields `num_matrices` identity matries of size $p \times p$. If that does not fit your needs (e.g. you have more than one leading axis, or you need colum-orthogonal matrices), then please do check POGO's `flatten_fn` and `rows` parameters.

## Citation

```bibtex
@article{javaloy2026pogo,
    title   = {An Embarrassingly Simple Way to Optimize Orthogonal Matrices at Scale},
    author  = {Javaloy, Adri{\'a}n and Vergari, Antonio},
    year    = 2025,
	journal = {ArXiv preprint},
	volume  = {TODO},
	url     = {https://arxiv.org/abs/TODO}
}
```
