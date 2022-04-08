from __future__ import print_function, division  
import numpy as np
from numpy.linalg import norm


def _rbf(X, Y, gamma='auto'):

    if gamma == 'auto' or gamma == 0:
        n_features = np.shape(X)[1] if len(np.shape(X)) > 1 else 1
        gamma = 1 / n_features
    assert gamma > 0, ""
    return np.exp(- gamma * norm(X - Y)**2)


def rbf(gamma='auto'):

    if not isinstance(gamma, str) and gamma > 0:
        def kernel(X, Y):
            return np.exp(- gamma * norm(X - Y)**2)
        return kernel
    else:
        return _rbf


def _linear(X, Y):
    """ Linear kernel, simply the dot product K(X, Y) = (X . Y).
    """
    return np.inner(X, Y)


def linear():
    """ Linear kernel, simply the dot product K(X, Y) = (X . Y).
    """
    return _linear


def _poly(X, Y, degree=3, coef0=1):
    """ Parametrized version of the polynomial kernel, K(X, Y) = (X . Y + coef0)^degree.

    - Default coef0 is 1.
    - Default degree is 3. Computation time is CONSTANT with d, but that's not a reason to try huge values. degree = 2,3,4,5 should be enough.
    - Using degree = 1 is giving a (possibly) non-homogeneous linear kernel.
    """
    assert degree > 0, "[ERROR] kernels.poly: using a degree < 0 will fail (the kernel is not p.d.)."
    return (np.dot(X, Y) + coef0) ** degree


def poly(degree=3, coef0=1):
    """ Return the polynomial kernel of degree d (X, Y -> K(X, Y) = (X . Y + coef0)^d), degree = 3 by default.
    """
    assert degree > 0, "[ERROR] kernels.poly: using a degree < 0 will fail (the kernel is not p.d.)."

    def kernel(X, Y):
        """ Parameter-free version of the polynomial kernel, K(X, Y) = (X . Y + coef0)^d.
        """
        return (np.dot(X, Y) + coef0) ** degree
    return kernel


