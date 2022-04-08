from __future__ import print_function, division  # Python 2 compatibility if needed
import numpy as np
from numpy.linalg import norm

# XXX Using cvxopt as a quadratic program solver
# (install it with 'pip install cvxopt', see http://cvxopt.org/ if needed)
from cvxopt import matrix as cvxMat
from cvxopt.solvers import qp as QPSolver
from cvxopt.solvers import options as QPSolverOptions
QPSolverOptions['show_progress'] = True
QPSolverOptions['maxiters'] = 250
QPSolverOptions['abstol'] = 1e-8
QPSolverOptions['reltol'] = 1e-6
QPSolverOptions['feastol'] = 1e-8


# XXX Using joblib for easy parallel computation (for multi-class SVM)
# (install it with 'pip install joblib', see https://pythonhosted.org/joblib/ if needed)
try:
    from joblib import Parallel #, delayed
    # from joblib import Memory
    # mem = Memory(cachedir='/tmp/joblib/svm.py/')
    # parallel_backend = 'multiprocessing'  # XXX it fails (Can't pickle local object 'mySVC.fit.<locals>.trainOneBinarySVC')
    parallel_backend = 'threading'  # XXX it works, but does NOT speed up the projection/prediction (but does speed up the multi-class QP solver)
    USE_JOBLIB = False
    print("Succesfully imported joblib.Parallel, it will be used to try to run training/projecting of multi-class SVC in parallel ...")
except ImportError:
    parallel_backend = None
    USE_JOBLIB = False
    print("[WARNING] Failed to import joblib.Parallel, it will not be used ...")


# XXX Using numba.jit to speed-up parallel computation (for multi-class SVM)
# (install it with 'pip install numba', see http://numba.pydata.org/#installing if needed)
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


import kernels  # Hand-made module
# Map kernel names to their function.
all_kernels = {'rbf': kernels.rbf}



# ----------------------------------------------------------------------------
# %% Utility function
def classification_score(yc, ytrue):
    """ Compute the classification error of the predicted vector (yc) compared to the known one (ytrue).

    - Uses the counting norm (ie. L0-norm), simply counts the number of mis-classified points.
    - Every mis-classification counts for the same mistake, e.g. classifying a 8 as a 9 or a 7 is the same mistake.
    """
    error = norm(yc - ytrue, 0)  # Use L0-norm
    score = (1 - error/np.size(ytrue))
    return score

# ------------------------------------------------------
# %% The BinarySVC class
# Inspired from http://www.mblondel.org/journal/2010/09/19/support-vector-machines-in-python/

# FIXED all X should NOT be support vectors! Increase this value!
MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-3


class BinarySVC(object):
    """ A home-made binary Support Vector Classifier.

    - WARNING: Binary labels are ALWAYS assumed to be in {-1,+1} and NOT in {0,1}.
    """

    def __init__(self, kernel='rbf', C=1.0,
                 degree=3, gamma='auto', coef0=1.0,
                 threshold=MIN_SUPPORT_VECTOR_MULTIPLIER,
                 use_smo=False, verbose=1, cache_size=200):
        """ A home-made binary Support Vector Classifier.

    Parameters:
      - kernel: can be a string (one from 'linear', 'rbf', 'laplace', 'sigmoid', 'poly'), or a user-defined kernel. Default is 'rbf'
      - C: is the regularization parameter, default is 1. Can be None for *hard*-margin SVM
      - degree: for 'poly' kernel
      - gamma: similarity parameter for 'rbf', 'laplace', 'sigmoid' kernels. Has to be > 0, and 0 or 'auto' use gamma = 1 / n_features
      - coef0: offset for 'poly' and 'sigmoid' kernel
      - threshold: only keep the support vector of Lagrange multiplier > threshold
      - use_smo: True to use the SMO algorithm, False use the cvxopt.qp solver
      - verbose: if > 0, some messages are printed on each method calls (very useful). Set to +1 for some messages, +2 for all messages (LOTS of prints for SMO.smo)
      - cache_size: NOT YET used (for compatibility)

    The SVM object has these main methods:
      - fit(X, y): to train the classifier on training data (X, y) (y are binary, in {-1,+1}, NOT {0, 1})
      - predict(X): predict the labels (0 or 1) for each point in the dataset X
      - score(X, y): give the ratio of well predicted labels (a fraction between 0% -- very bad -- and 100% -- perfect)
      - project(X): projects the points onto the hyperplane (before taking the sign to give the prediction)
      - str(clf): or print(clf) prints the options and learned parameters of the model

    The SVM object has these main attributes:
      - _c, _gamma, _degree, _coef0: parameters of the models
      - _verbose, _cache_size: options (not yet)
      - _kernel, _kernel_name: function and name of the kernel

    When it has been trained, it has:
      - b = intercept_: bias
      - w = coef_: weight in case of linear kernel
      - a: Lagrange multipliers used to compute the projection/prediction (dual optimal variables)
      - n_support_: number of support vector
      - sv: support vectors (shape (n_support_, n_features))
      - sv_y: labels of the support vectors (shape (n_support_, 1))

    WARNING:
      - We use the scikit-learn convention, X is always of shape (n_samples, n_features), and y is of shape (n_samples, ) or (n_samples, 1)
      - The QP solver won't work for too large dataset (> 5000)
        """
        assert (C is None) or (C >= 0), "[ERROR] BinarySVC require a strictly positive value for C (regularization parameter)."
        self._c = C
        if self._c is not None:
            self._c = float(self._c)
        assert (gamma == 'auto') or (gamma >= 0), "[ERROR] BinarySVC require a positive value for gamma or 'auto'."
        self._gamma = gamma
        self._degree = degree
        self._coef0 = coef0
        self._threshold = threshold
        self._use_smo = use_smo
        self._verbose = verbose
        self._cache_size = cache_size
        # Kernel parameter, can be a string or a callable
        if isinstance(kernel, str):
            self._kernel_name = kernel
            k = all_kernels[kernel]
            if kernel in ['rbf', 'laplace']:
                self._kernel = k(gamma=gamma)
            elif kernel in ['poly']:
                self._kernel = k(degree=degree, coef0=coef0)
            elif kernel in ['sigmoid']:
                self._kernel = k(gamma=gamma, coef0=coef0)
            else:  # 'linear'
                self._kernel = k()
        else:
            self._kernel_name = "User defined (%s)" % str(kernel)
            self._kernel = kernel
        self._log("  A new {} object has been created:\n    > {}".format(self.__class__.__name__, self))

    # Print the model if needed:
    def __str__(self):
        """ Print the parameters of the classifier, if possible."""
        return "BinarySVC(kernel={}, C={}, degree={}, gamma={}, coef0={}, threshold={}, use_smo={}, verbose={}, cache_size={})".format(self._kernel_name, self._c, self._degree, self._gamma, self._coef0, self._threshold, self._use_smo, self._verbose, self._cache_size)

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
        """ Given the training features X with labels y, trains the SVM model object (compute Lagrange multipliers, construct a predictor).

        - The number of support vectors is kept in self.n_support_ and returned
        - The Gram matrix K can be given directly, if precomputed
        - The QP problem is solved either with SMO or with cvxopt.qp
        """
        n_samples, n_features = np.shape(X)
        self._log("  Training BinarySVC... on X of shape {} ...".format(np.shape(X)))
        # if set(y) == {-1, +1}:  # In O(n_samples) but not more
        #    print("[WARNING] BinarySVC.fit: y were into {-1, +1} and not {0, 1}, be careful when using the BinarySVC.predict method.")
        #     y = np.array((y + 1) // 2, dtype=int)
        # if set(y) != {0, 1}:  # In O(n_samples) but not more
        #     raise ValueError("BinarySVC.fit: y are not binary, they should belong to {0, 1}.")
        if set(y) == {0, 1}:  # In O(n_samples) but not more
            self._logverbose("[WARNING] BinarySVC.fit: y were into {0, 1} and not {-1, +1}, be careful when using the BinarySVC.predict method.")
            y = np.array((y * 2) - 1, dtype=int)
        if set(y) != {-1, +1}:  # In O(n_samples) but not more
            raise ValueError("BinarySVC.fit: y are not binary, they should belong to {-1, +1}.")
        # XXX if X is pretty large, this will not fit into memory
        if n_samples > 500:
            self._logverbose("[WARNING] BinarySVC.fit: X n_samples is large (> 500), the computation of the Gram matrix K might use a lot of memory.")
        # XXX use a better implementation, like SMO, with a cache_size parameter
        if n_samples > 15000:
            raise ValueError("BinarySVC.fit: X n_samples is very large (> 5000), the computation of the Gram matrix K will fail (out of memory) and the QP program is untractable.\n ==> Use the SMO algorithm!")
        # XXX if X is pretty large, this will not fit into memory
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

        # Find the Lagrange multipliers (solution of dual problem), either by SMO or by the QP solver
        if self._use_smo:
            self._log("  Using the SMO algorithm. WARNING still experimental! ...")
            a, b = self._solve_smo(X, y, K)
        else:
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

        # Intercept/bias value b
        if not self._use_smo:
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

    # Solve the QP problem with the QP solver
    def _solve_qp(self, X, y, K):
        r"""  Train a BinarySVC model with a cvxopt.qp solver.

        Solves this quadratic problem :

            - min 1/2 x^T P x + q^T x
            - s.t.
                G x \coneleq h
                A x = b
        """
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

    # Solve the QP problem with SMO
    def _solve_smo(self, X, y, K):
        """ Train a BinarySVC model with the SMO algorithm and not a QP solver.

        See the file 'SMO.py' and the SMO.smo function for more details.
        """
        alpha, b, passes, total_passes = smo(self, X, y, K)  # , tol=TOLERANCE, max_passes=10)
        self._log("==> Done for SMO:\n - total number of passes = {}\n- final number of passes without changing alpha = {} ...".format(total_passes, passes))
        return alpha, b

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

    # Small function to outsource the computation of the Gram matrix K (cache of the kernel applied to data point)
    # @jit  # FIXME Still highly experimental
    def _gram_matrix(self, X):
        """ Compute the Gram matrix of the samples X.

        - Warning: it takes a time O(n^3) for n ~ n_samples ~ n_features. Will take a lot of time if n is too big!
        - Warning: it requires a memory space of O(n^2). Will not fit if n_samples is too big!
        """
        self._log("  Computing Gram matrix for a BinarySVC for data X of shape {} ...".format(np.shape(X)))
        n_samples, _ = np.shape(X)
        K = np.zeros((n_samples, n_samples))
        # TODO Vectorize or parallelize this slow part !
        # nb_change_on_K = 0
        for i, x_i in enumerate(X):
            # nb_change_on_K += 1
            K[i, i] = self._kernel(x_i, x_i)
            # for j, x_j in enumerate(X):
            for j in range(i+1, n_samples):
                # x_j = X[j]
                # Exploit the symmetry of the kernel, do half the operations
                # if i > j:
                K[i, j] = K[j, i] = self._kernel(x_i, X[j])
                #    nb_change_on_K += 1
                # else:
                #     break
        # assert nb_change_on_K == int(n_samples * (n_samples + 1) / 2), "[ERROR] BinarySVC._gram_matrix : wrong number of computation of K[i, j] ... Check the code!"
        return K

    # @jit  # FIXME Still highly experimental
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
        """ Computes the SVM prediction on the given features X.

        - This work ONLY for BINARY classification, warning! Binary meaning {-1,+1} as ALWAYS.
        - Very efficient, no optimization is done here.
        - Time and memory complexity is is O(n_samples, n_support_vectors).
        """
        self._log("  Predicting on a BinarySVC for data X of shape {} ...".format(np.shape(X)))
        predictions = np.sign(self.project(X))
        self._log("  Stats about the predictions: (0 should never be predicted, labels are in {-1,+1})\n", list((k, np.sum(predictions == k)) for k in [-1, 0, +1]))
        return predictions

    # Just for compatibility with sklearn.svm.SVC

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


# ------------------------------------------------------
# %% The mySVC class

class mySVC(BinarySVC):
    """ A home-made multi-class Support Vector Classifier, implementing the « one-vs-rest » strategy.

    - It will use and train k BinarySVC models if trained with k-class labeled data (Gram matrix is shared, but the number of support vectors will be multiplied by k)
    - 'ovr' is the recommended strategy, see the discussion on http://scikit-learn.org/dev/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    """

    def __init__(self, max_n_classes=10, n_jobs=1, **kwargs):
        """ A home-made binary Support Vector Classifier.

        Parameters:
          - max_n_classes: max number of class it will accept
          - n_jobs: integer, number of parallel job for the training/projection/prediction of the K BinarySVC classifier (default = 1). If -1 all CPUs are used
          - kernel: can be a string (one from 'linear', 'rbf', 'laplace', 'sigmoid', 'poly'), or a user-defined kernel. Default is 'rbf'
          - C: is the regularization parameter, default is 1. Can be None for *hard*-margin SVM
          - degree: for 'poly' kernel
          - gamma: similarity parameter for 'rbf', 'laplace', 'sigmoid' kernels. Has to be > 0, and 0 or 'auto' use gamma = 1 / n_features
          - coef0: offset for 'poly' and 'sigmoid' kernel
          - threshold: only keep the support vector of Lagrange multiplier > threshold
          - use_smo: True to use the SMO algorithm, False use the cvxopt.qp solver
          - verbose: if > 0, some messages are printed on each method calls (very useful). Set to +1 for some messages, +2 for all messages (LOTS of prints for SMO.smo)
          - cache_size: NOT YET used (for compatibility)

        The SVM object has these main methods:
          - fit(X, y): to train the classifier on training data (X, y) (y are binary, 0 or 1, NOT +1 or -1)
          - predict(X): predict the labels (0 or 1) for each point in the dataset X
          - score(X, y): give the ratio of well predicted labels (a fraction between 0% -- very bad -- and 100% -- perfect)
          - project(X): projects the points onto the hyperplane (before taking the sign to give the prediction)
          - str(clf): or print(clf) prints the options and learned parameters of the model

        The SVM object has these main attributes:
          - _c, _gamma, _degree, _coef0: parameters of the models
          - _verbose: control level of verbosity for log messages (integer, 0 = no log, 1 = useful log, 2 = all log)
          - _cache_size: option to increase the cache size for SMO (not yet)
          - _kernel, _kernel_name: function and name of the kernel

        When it has been trained, it has:
          - n_classes_: number of class it has been trained on
          - _binary_SVCs: list of n_classes_ BinarySVC models (one for each class)
          - n_support_: number of support vector for each class (shape (n_classes_, 1))
          - intercept_: bias for each class (shape (n_classes_, 1))
          - coef_: weight in case of linear kernel for each class (shape (n_classes_, n_features) ??)

        WARNING:
          - We use the scikit-learn convention, X is always of shape (n_samples, n_features), and y is of shape (n_samples, ) or (n_samples, 1)
          - The QP solver won't work for too large dataset (n_samples > 5000 or n_features > 1000)

        Remark:
          - [sklearn.multiclass.OneVsRestClassifier](http://scikit-learn.org/dev/modules/generated/sklearn.multiclass.OneVsRestClassifier.html#sklearn.multiclass.OneVsRestClassifier) could have been used instead... But that was NOT the point.
        """
        self._max_n_classes = max_n_classes
        self.n_classes_ = None
        self._n_jobs = n_jobs
        # Black magic to pass the rest of the arguments to the BinarySVC.__init__ method
        super(mySVC, self).__init__(**kwargs)

    # Print the model if needed:
    def __str__(self):
        """ Print the parameters of the classifier, if possible."""
        return "mySVC(kernel={}, C={}, n_classes_={}, max_n_classes={}, degree={}, gamma={}, coef0={}, threshold={}, use_smo={}, verbose={}, n_jobs={}, cache_size={})".format(self._kernel_name, self._c, self.n_classes_, self._max_n_classes, self._degree, self._gamma, self._coef0, self._threshold, self._use_smo, self._verbose, self._n_jobs, self._cache_size)

    __repr__ = __str__

    # Learn the model
    def fit(self, X, y, n_jobs=None):
        """ Given the training features X with labels y, trains the SVM model object (compute Lagrange multipliers, construct a predictor).

        - Will fail if y is not discrete labels in [| 0, ..., n_classes |] (with a ValueError exception),
        - n_classes has to be <= self._max_n_classes,

        - The number of classes is kept in self.n_classes_ and returned.

        - It tries to use joblib.Parallel to train the BinarySVC in parallel (and it works quite fine).
        """
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
            'use_smo': self._use_smo,
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
        # 4'. FIXED Use joblib Parallel to speed up these training steps
        #if USE_JOBLIB:
            #self._log("  Calling a Parallel utility object (from joblib) with n_jobs = {} and trainOneBinarySVC(k) for k = 0 .. {} ...".format(n_jobs, n_classes-1))
            #self._logverbose("  Cf. https://pythonhosted.org/joblib/parallel.html#joblib.Parallel if needed")
            #all_results = Parallel(n_jobs=n_jobs, verbose=20, backend=parallel_backend)(
                #delayed(trainOneBinarySVC, check_pickle=False)(
                    #self._log, self._binary_SVCs[k], X, y, GramK, k)
                #for k in range(n_classes)
            #)
        #else:
            #self._log("  Not using any parallelism for trainOneBinarySVC(k) for k = 0 .. {} ...".format(n_classes-1))
            #all_results = [
                #trainOneBinarySVC(self._log, self._binary_SVCs[k], X, y, GramK, k)
                #for k in range(n_classes)
            #]
        #else:
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
        # 4'. FIXED Use joblib Parallel to speed up these projections
        #if USE_JOBLIB:
            # FIXME it still does not improve the running time, WHY ?
            #self._log("  Calling a Parallel utility object (from joblib) with n_jobs = {} and projectOneBinarySVC(k) for k = 0 .. {} ...".format(n_jobs, n_classes-1))
            #self._logverbose("  Cf. https://pythonhosted.org/joblib/parallel.html#joblib.Parallel if needed")
            #all_results = Parallel(n_jobs=n_jobs, verbose=20, backend=parallel_backend)(
                #delayed(projectOneBinarySVC, check_pickle=False)(
                    #self._log, self._binary_SVCs[k], X, k)
                #for k in range(n_classes)
            #)
        #else:
            #self._log("  Not using any parallelism for projectOneBinarySVC(k) for k = 0 .. {} ...".format(n_classes-1))
            #all_results = [
                #projectOneBinarySVC(self._log, self._binary_SVCs[k], X, k)
                #for k in range(n_classes)
            #]
        #else:
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
        """ Computes the SVM prediction on the given features X.

        - Very efficient, no optimization is done here.
        - Time and memory complexity is is O(n_samples, n_support_vectors).

        - It uses joblib.Parallel to project X with the BinarySVC in parallel (still experimental, it does not speed up the projection yet).
        """
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


# ------------------------------------------------------
# %% XXX If suddenly my implementation of BinarySVC is failing, fix it by uncommenting this line
# from sklearn.svm import SVC as mySVC  # XXX
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html


# ------------------------------------------------------
#%% XXX If suddenly my implementation of mySVC is failing, fix it by uncommenting this line
from sklearn.multiclass import OneVsRestClassifier
mySVC2 = OneVsRestClassifier(BinarySVC, n_jobs=(-1))
# # http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html

# ------------------------------------------------------
# %% XXX To experiment with the « one-vs-one » strategy instead !
# from sklearn.multiclass import OneVsOneClassifier
# mySVC3 = OneVsOneClassifier(BinarySVC, n_jobs=(-1))
# # http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html

# End of svm.py