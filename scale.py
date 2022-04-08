from __future__ import print_function, division  # Python 2 compatibility if needed
import numpy as np


def scale(X, copy=True):
    r""" Preprocessing of the data X : center and reduce (scale).

    - Center to the mean, and scale to unit variance the WHOLE array, NOT component wise,
    - X has to be a numpy array (or array-like),
    - if copy is False, do not create a new array, but change X inplace, EXPERIMENTAL (default is to give a fresh copy, copy=True).
    """
    mu = np.mean(X)
    sigma = np.std(X)
    assert sigma >= 0, "[ERROR] scale.scale got an array of shape {} with np.std < 0, that should not happen.".format(np.shape(X))
    if not copy:  # Use X and modify it directly.
        X -= mu     # Center
        X /= sigma  # Reduce
        return X  # Should be useless
    else:
        if sigma > 0:
            Y = (X - mu) / sigma  # Center and reduce
        else:
            Y = X - mu  # Just center
        return Y

# End of scale.py