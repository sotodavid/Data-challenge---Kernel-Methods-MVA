#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division  # Python 2 compatibility if needed
import numpy as np
from numpy.linalg import norm

from cvxopt import matrix as cvxMat
from cvxopt.solvers import qp as QPSolver
from cvxopt.solvers import options as QPSolverOptions
QPSolverOptions['show_progress'] = True
QPSolverOptions['maxiters'] = 250
QPSolverOptions['abstol'] = 1e-8
QPSolverOptions['reltol'] = 1e-6
QPSolverOptions['feastol'] = 1e-8

try:
    from joblib import Parallel #, delayed
    parallel_backend = 'threading'  # XXX it works, but does NOT speed up the projection/prediction (but does speed up the multi-class QP solver)
    USE_JOBLIB = False
    print("Succesfully imported joblib.Parallel, it will be used to try to run training/projecting of multi-class SVC in parallel ...")
except ImportError:
    parallel_backend = None
    USE_JOBLIB = False
    print("[WARNING] Failed to import joblib.Parallel, it will not be used ...")


try:
    from numba import jit
    USE_NUMBA = True
    print("Succesfully imported numba.jit, it will be used to try to speed-up function calls ...")
except ImportError:
    def jit(function):
        """ Fake decorator in case numba.jit is not available. """
        return function
    USE_NUMBA = False
    print("[WARNING] Failed to import numba.jit, it will not be used ...")


import kernels 
all_kernels = {'rbf': kernels.rbf}




def classification_score(yc, ytrue):

    error = norm(yc - ytrue, 0)  # Use L0-norm
    score = (1 - error/np.size(ytrue))
    return score


# Inspired from http://www.mblondel.org/journal/2010/09/19/support-vector-machines-in-python/


MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-3


class BinarySVC(object):

    def __init__(self, kernel='rbf', C=1.0,
                 degree=3, gamma='auto', coef0=1.0,
                 threshold=MIN_SUPPORT_VECTOR_MULTIPLIER,
                 verbose=1, cache_size=200):

        assert (C is None) or (C >= 0), "[ERROR] BinarySVC require a strictly positive value for C (regularization parameter)."
        self._c = C
        if self._c is not None:
            self._c = float(self._c)
        assert (gamma == 'auto') or (gamma >= 0), "[ERROR] BinarySVC require a positive value for gamma or 'auto'."
        self._gamma = gamma
        self._degree = degree
        self._coef0 = coef0
        self._threshold = threshold
        self._verbose = verbose
        self._cache_size = cache_size
        # Kernel parameter, can be a string or a callable
        if isinstance(kernel, str):
            self._kernel_name = kernel
            k = all_kernels[kernel]
            if kernel in ['rbf']:
                self._kernel = k(gamma=gamma)
            elif kernel in ['poly']:
                self._kernel = k(degree=degree, coef0=coef0)
            else:  # 'linear'
                self._kernel = k()
        else:
            self._kernel_name = "User defined (%s)" % str(kernel)
            self._kernel = kernel
        self._log("  A new {} object has been created:\n    > {}".format(self.__class__.__name__, self))

    # Print the model if needed:
    def __str__(self):
        """ Print the parameters of the classifier, if possible."""
        return "BinarySVC(kernel={}, C={}, degree={}, gamma={}, coef0={}, threshold={}, verbose={}, cache_size={})".format(self._kernel_name, self._c, self._degree, self._gamma, self._coef0, self._threshold, self._verbose, self._cache_size)

    __repr__ = __str__

    def _log(self, *args):
        """ Print only if self._verbose > 0. """
        if self._verbose > 0:
            print(*args)

    def _logverbose(self, *args):
        """ Print only if self._verbose > 1. """
        if self._verbose > 1:
            print(*args)

    def _logverbverbose(self, *args):
        """ Print only if self._verbose > 2. """
        if self._verbose > 2:
            print(*args)

    # Learn the model
    def fit(self, X, y, K=None):

        n_samples, n_features = np.shape(X)
        self._log("  Training BinarySVC... on X of shape {} ...".format(np.shape(X)))
        if set(y) == {0, 1}:  # In O(n_samples) but not more
            self._logverbose("[WARNING] BinarySVC.fit: y were into {0, 1} and not {-1, +1}, be careful when using the BinarySVC.predict method.")
            y = np.array((y * 2) - 1, dtype=int)
        if set(y) != {-1, +1}:  # In O(n_samples) but not more
            raise ValueError("BinarySVC.fit: y are not binary, they should belong to {-1, +1}.")
        if n_samples > 500:
            self._logverbose("[WARNING] BinarySVC.fit: X n_samples is large (> 500), the computation of the Gram matrix K might use a lot of memory.")
        if n_features > 50:
            self._logverbose("[WARNING] BinarySVC.fit: X n_features is large (> 50), the computation of the Gram matrix K might require a lot of time.")
        if (n_samples > 500) or (n_features > 10):
            self._logverbose("[WARNING] BinarySVC.fit: X n_samples is large (> 500) or n_features is large (> 10), the training of the BinarySVC will take time ...")
        # Compute the Gram matrix, or use the one given in argument
        # (easy way to speed up the multi-class SVC)
        if K is None:
            K = self._gram_matrix(X)
        else:
            self._log("  Using the given Gram matrix K.")


        self._log("  Using the QP solver from cvxopt (cvxopt.qp) ...")
        a, b = self._solve_qp(X, y, K)

        # Support vectors have non zero lagrange multipliers
        sv = a > self._threshold
        # XXX we should not have 100% of X as support vectors, so self._threshold should not be too small !
        self._ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        self._log("  => {} support vectors out of {} points.".format(len(self.a), n_samples))


        b = 0
        for n in range(len(self.a)):
            b += self.sv_y[n]
            b -= np.sum(self.a * self.sv_y * K[self._ind[n], sv])
        if len(self.a) > 0:  # XXX len(self.a) == 0 should never happen
            b /= len(self.a)
        self.b = b

        # Weight vector
        if self._kernel_name == 'linear':
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
            self._log("  Weight vector of shape {}.".format(np.shape(self.w)))
        else:
            self.w = None
            self._log("  No weight vector, non-linear kernel, will use the Lagrange multipliers self.a ...")

        # Compute and store number of support vector
        self.n_support_ = len(self.sv)
        self._log("  Keeping {} support vectors.".format(self.n_support_))
        return self.n_support_

    def _solve_qp(self, X, y, K):


        n_samples, _ = np.shape(X)
        P = cvxMat(np.outer(y, y) * K)
        q = cvxMat(np.ones(n_samples) * (-1))

        if (self._c is None) or (self._c <= 0):
            # Hard-margin, 0 <= a_i. Hard-margin is like having C=+oo.
            G = cvxMat(np.diag(np.ones(n_samples) * (-1)))
            h = cvxMat(np.zeros(n_samples))
        else:
            # Soft-margin, 0 <= a_i <= c
            # -a_i <= 0
            G_top = np.diag(np.ones(n_samples) * (-1))
            h_left = np.zeros(n_samples)
            # a_i <= c
            G_bot = np.identity(n_samples)
            h_right = np.ones(n_samples) * self._c
            G = cvxMat(np.vstack((G_top, G_bot)))
            h = cvxMat(np.hstack((h_left, h_right)))

        A = cvxMat(y, (1, n_samples))  # Matrix of observations
        b = cvxMat(0.0)  # Bias = 0

        self._log("  More information on http://cvxopt.org/userguide/coneprog.html#quadratic-programming if needed")
        # Solve QP problem, by calling the QP solver (quadratic program)
        solution = QPSolver(P, q, G, h, A, b)

        # Lagrange multipliers (optimal dual variable)
        a = np.ravel(solution['x'])
        self._log("  The QP solver found Lagrange multipliers of shape {} !".format(np.shape(a)))
        return a, 0.0



    # Get the score
    def score(self, X, y):
        """ Compute the classification error for this classifier (compare the predicted labels of X to the exact ones given by y)."""
        ypredicted = self.predict(X)
        score = classification_score(ypredicted, y)
        if not 0 <= score <= 1:
            self._log("clf.score(X, y): error the computed score is not between 0 and 1 (score = {})".format(score))
        elif score == 1:
            self._log("[SUCCESS UNLOCKED] clf.score(X, y): the computed score is exactly 1 ! Yeepee ! Exact prediction ! YOUUUU")
        return score


    def _gram_matrix(self, X):

        self._log("  Computing Gram matrix for a BinarySVC for data X of shape {} ...".format(np.shape(X)))
        n_samples, _ = np.shape(X)
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            # nb_change_on_K += 1
            K[i, i] = self._kernel(x_i, x_i)
            # for j, x_j in enumerate(X):
            for j in range(i+1, n_samples):
                 K[i, j] = K[j, i] = self._kernel(x_i, X[j])
        return K

    def project(self, X):
        """ Computes the SVM projection on the given features X. """
        self._log("  Projecting on a BinarySVC for data X of shape {} ...".format(np.shape(X)))
        if (np.shape(X)[0] > 500) or (np.shape(X)[1] > 10):
            self._logverbose("[WARNING] BinarySVC.project: X n_samples is large (> 500) or n_features is large (> 10), projecting with this trained BinarySVC will take time ...")
        if self.w is not None:
            # Use self.w because it is simpler and quicker than using _kernel
            self._log("    Linear kernel, using self.w...")
            return self.b + np.dot(X, self.w)
        else:
            self._log("    Non-linear kernel, using self.a, self.sv and self.sv_y ...")
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                y_predict[i] = sum(a * sv_y * self._kernel(X[i], sv) for a, sv, sv_y in zip(self.a, self.sv, self.sv_y))
            return y_predict + self.b

    def predict(self, X):

        self._log("  Predicting on a BinarySVC for data X of shape {} ...".format(np.shape(X)))
        predictions = np.sign(self.project(X))
        self._log("  Stats about the predictions: (0 should never be predicted, labels are in {-1,+1})\n", list((k, np.sum(predictions == k)) for k in [-1, 0, +1]))
        return predictions


    decision_function = project

    @property
    def intercept_(self):
        """ Constant term (bias b) in the SVM decision function. """
        return self.b

    @property
    def coef_(self):
        """ Weight term (w) in the SVM decision function. """
        if self.w is not None:
            return self.w
        else:
            raise ValueError("BinarySVC.coef_ is only available when using a linear kernel")

    @property
    def support_(self):
        """ Indices of support vectors. """
        return self._ind

    @property
    def support_vectors_(self):
        """ Support vectors. """
        return self.sv


class mySVC(BinarySVC):


    def __init__(self, max_n_classes=10, n_jobs=1, **kwargs):
   
        self._max_n_classes = max_n_classes
        self.n_classes_ = None
        self._n_jobs = n_jobs
        super(mySVC, self).__init__(**kwargs)

    def __str__(self):
        """ Print the parameters of the classifier, if possible."""
        return "mySVC(kernel={}, C={}, n_classes_={}, max_n_classes={}, degree={}, gamma={}, coef0={}, threshold={}, verbose={}, n_jobs={}, cache_size={})".format(self._kernel_name, self._c, self.n_classes_, self._max_n_classes, self._degree, self._gamma, self._coef0, self._threshold, self._verbose, self._n_jobs, self._cache_size)

    __repr__ = __str__

    def fit(self, X, y, n_jobs=None):

        if n_jobs is None:
            n_jobs = self._n_jobs
        # n_samples, n_features = np.shape(X)
        self._log("  Training mySVC... on X of shape {}.".format(np.shape(X)))
        # 1. Detect how many classes there is
        n_classes = len(np.unique(y))
        self._log("  Using y of shape {} with {} different classes".format(np.shape(y), n_classes))
        # If too few classes
        if n_classes == 1:
            raise ValueError("mySVC.fit: n_classes guessed to be = 1 from y, the SVM cannot learn to classify points X if it sees only one class.")
        if n_classes == 2:
            raise ValueError("mySVC.fit: n_classes guessed to be = 2 from y, you should use the BinarySVC class.")
        # If too many classes
        if n_classes > self._max_n_classes:
            raise ValueError("mySVC.fit: too many classes in the input labels vector y (there is {}, max authorized is {}).\n  - Try to check that y is indeed discrete labels, and maybe increase the parameter max_n_classes.".format(n_classes, self._max_n_classes))
        # Check that it is not absurd (ie. y really are labels in [|0, .., n_classes - 1|])
        if not set(y) == set(range(n_classes)):
            raise ValueError("mySVC.fit: incorrect input labels vector y (there is {} classes but y's values are not in [|0, ..., {}|]).".format(n_classes, n_classes))
        # 2. Build n_classes instances of BinarySVC
        # Get the parameters of self (option and parameters given to the mySVC object)
        parameters = {
            'kernel': self._kernel_name,
            'C': self._c,
            'degree': self._degree,
            'gamma': self._gamma,
            'coef0': self._coef0,
            'verbose': self._verbose,
            'cache_size': self._cache_size  # XXX useless
        }
        self._log("  BinarySVC parameters:\n", parameters)
        self._binary_SVCs = [None] * n_classes
        for k in range(n_classes):
            self._binary_SVCs[k] = BinarySVC(**parameters)
        # Initialize the aggregated parameters (not used...)
        self.n_support_ = [None] * n_classes  # np.zeros(n_classes, dtype=int)
        self.b = [None] * n_classes  # np.zeros((n_classes, n_features))
        self.w = [None] * n_classes  # np.zeros(n_classes)
        # 3. Computing the Gram matrix only once
        self._log("  Computing the Gram matrix only once to speed up training time for each BinarySVC.")
        GramK = self._gram_matrix(X)
        # 4. Train each of the BinarySVC model
        # for k in range(n_classes):
        # XXX: run in parallel the training of each BinarySVC !

        def trainOneBinarySVC(log, kth_BinarySVC, X, y, GramK, k):
            """ Training for the k-th BinarySVC. """
            log("  - For the class k = {}:".format(k))
            # yk = 1.0 * ((y == k) * 1)  # XXX Convert from {0,..,N-1} to {0,1}, NOPE
            yk = 1.0 * (((y == k) * 2) - 1)  # FIXED Convert from {0,..,N-1} to {-1,+1}
            log("    There is {} examples in this class (and {} outside).".format(np.sum(yk == 1), np.sum(yk == -1)))
            # FIXED speed up this part by computing the Gram matrix only ONCE and not k times!
            kth_BinarySVC.fit(X, yk, K=GramK)
            return kth_BinarySVC.n_support_, kth_BinarySVC.b, kth_BinarySVC.w

        self._log("  Not using any parallelism for trainOneBinarySVC(k) for k = 0 .. {} ...".format(n_classes-1))
        all_results = [
            trainOneBinarySVC(self._log, self._binary_SVCs[k], X, y, GramK, k)
            for k in range(n_classes)
        ]
        # Unpack and store the results
        for k in range(n_classes):
            # Get the parameters from the k-th binary SVC
            self.n_support_[k], self.b[k], self.w[k] = all_results[k]
        # Done, set the last parameters and go
        self.n_classes_ = n_classes
        return n_classes

    def project(self, X, n_jobs=None):
        """ Computes the SVM projection on the given features X.

        - It tries to use joblib.Parallel to project X with the BinarySVC in parallel (still experimental, it does not speed up the projection yet).
        """
        if n_jobs is None:
            n_jobs = self._n_jobs
        # 4. For each X, project with each of the BinarySVC model
        n_classes = self.n_classes_
        n_samples, _ = np.shape(X)
        self._log("  Projecting on a {}-class SVC for data X of shape {} ...".format(n_classes, np.shape(X)))
        projections = np.zeros((n_classes, n_samples))
        # for k in range(n_classes):

        def projectOneBinarySVC(log, kth_BinarySVC, X, k):
            """ Projecting on the k-th BinarySVC. """
            log("    Projecting on the {}-th BinarySVC ...".format(k))
            projections_k = kth_BinarySVC.project(X)
            # projections[k] = projections_k
            return projections_k

        self._log("  Not using any parallelism for projectOneBinarySVC(k) for k = 0 .. {} ...".format(n_classes-1))
        all_results = [
            projectOneBinarySVC(self._log, self._binary_SVCs[k], X, k)
            for k in range(n_classes)
        ]
        # Unpack and store the results
        for k in range(n_classes):
            projections[k] = all_results[k]
        # 5. Take the most probable label !
        return projections

    def predict(self, X, n_jobs=None):

        if n_jobs is None:
            n_jobs = self._n_jobs
        # n_samples, n_features = np.shape(X)
        projections = self.project(X, n_jobs=n_jobs)
        # If no projections are positive, pick the one that has the smallest margin (in norm), ie the largest values (remember, max(-2,-3)=-2 !)
        # Else, if some projections are positive, pick the one that has the largest positive norm
        predictions = np.array(np.argmax(projections, axis=0))  # , dtype=int)
        self._log("  Predicting on a {}-class SVC for data X of shape {} ...".format(self.n_classes_, np.shape(X)))
        self._log("  Stats about the predictions:\n", list((k, np.sum(predictions == k)) for k in range(self.n_classes_)))
        return predictions



from sklearn.multiclass import OneVsRestClassifier
mySVC2 = OneVsRestClassifier(BinarySVC, n_jobs=(-1))

