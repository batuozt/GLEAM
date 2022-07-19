"""
Implementations of various algorithms.
Written by Christopher M. Sandino (sandino@stanford.edu), 2020.
"""

import torch
from torch import nn


class ConjugateGradient(nn.Module):
    """
    Implementation of conjugate gradient algorithm to invert
    the linear system:
        y = A x
    where A is a symmetric, positive-definite matrix.
    In multi-coil MRI reconstruction, A is not symmetric. However, we can
    form the normal equations to get the problem in the form above:
        A^T y = A^T A x
    Based on code by Jon Tamir.
    """

    def __init__(self, A, num_iter, dbprint=False):
        super(ConjugateGradient, self).__init__()

        self.A = A
        self.num_iter = num_iter
        self.dbprint = dbprint

    def zdot(self, x1, x2):
        """
        Complex dot product between tensors x1 and x2.
        """
        return torch.sum(x1.conj() * x2)

    def zdot_single(self, x):
        """
        Complex dot product between tensor x and itself
        """
        return self.zdot(x, x).real

    def _update(self, iter):
        def update_fn(x, p, r, rsold):
            Ap = self.A(p)
            pAp = self.zdot(p, Ap)
            alpha = (rsold / pAp)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = self.zdot_single(r)
            beta = (rsnew / rsold)
            rsold = rsnew
            p = beta * p + r

            # print residual
            if self.dbprint:
                print(f'CG Iteration {iter}: {rsnew}')

            return x, p, r, rsold

        return update_fn

    def forward(self, x, y):
        # Compute residual
        r = y - self.A(x)
        rsold = self.zdot_single(r)
        p = r

        for i in range(self.num_iter):
            x, p, r, rsold = self._update(i)(x, p, r, rsold)

        return x
    
    def reverse(self, x):
        out = (1/self.lamb) * (self.A(x) + self.lamb*x - self.Aty)
        return out

class PowerMethod(nn.Module):
    """
    Implementation of power method to compute singular values of batch
    of matrices.
    """
    def __init__(self, num_iter, eps=1e-6):
        super(PowerMethod, self).__init__()

        self.num_iter = num_iter
        self.eps = eps

    def forward(self, A):
        # get data dimensions
        batch_size, m, n = A.shape

        # initialize random eigenvector directly on device
        v = torch.rand((batch_size, n, 1), dtype=torch.complex64, device=A.device)

        # compute A^H A
        AhA = torch.bmm(A.conj().permute(0, 2, 1), A)

        for _ in range(self.num_iter):
            v = torch.bmm(AhA, v)
            eigenvals = (torch.abs(v) ** 2).sum(1).sqrt()
            v = v / (eigenvals.reshape(batch_size, 1, 1) + self.eps)

        return eigenvals.reshape(batch_size)