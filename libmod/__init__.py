"""

Notes
-----
Always include the dependate variable in your imputation model. Whether you should use the imputed values of the dependent variable in your analysis is unclear.

For bootstrap analysis, set the start seed. This will XXX which will decorrelate the replicates and yield lower variance.

Questions
---------
Is smearing necessarying when backtransforming log-transformed data, if the imputed value is drawn from the posterior predictive distribution? For examples see Faria and others, 2014.


"""

import pandas as pd
import numpy as np

from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# TOH testing the following 3 lines are
from sklearn.impute._iterative import * #XXX
from sklearn.impute._base import (_get_mask, MissingIndicator, SimpleImputer, _check_inputs_dtype) #XXX
_ImputerTriplet = namedtuple('_ImputerTriplet', ['feat_idx',
                                                 'neighbor_feat_idx',
                                                 'estimator']) #XXX
from sklearn.exceptions import ConvergenceWarning

from blockbootstrap import BBS

import warnings

class MiceImputer(IterativeImputer):
    def __init__(self, Xt_previous=None, **kwargs):
        params = self._fill_params(**kwargs)
        super().__init__(**params)
        self.Xt_previous = Xt_previous


    def _fill_params(self, **kwargs):
        temp = IterativeImputer(**kwargs)
        params = temp.get_params()
        return params


    def _initial_imputation(self, X):
        """Perform initial imputation for input X.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data, where "n_samples" is the number of samples and
            "n_features" is the number of features.
        Returns
        -------
        Xt : ndarray, shape (n_samples, n_features)
            Input data, where "n_samples" is the number of samples and
            "n_features" is the number of features.
        X_filled : ndarray, shape (n_samples, n_features)
            Input data with the most recent imputations.
        mask_missing_values : ndarray, shape (n_samples, n_features)
            Input data's missing indicator matrix, where "n_samples" is the
            number of samples and "n_features" is the number of features.
        """
        if is_scalar_nan(self.missing_values):
            force_all_finite = "allow-nan"
        else:
            force_all_finite = True

        X = check_array(X, dtype=FLOAT_DTYPES, order="F",
                        force_all_finite=force_all_finite)
        _check_inputs_dtype(X, self.missing_values)

        mask_missing_values = _get_mask(X, self.missing_values)

        if self.initial_imputer_ is None:
            self.initial_imputer_ = SimpleImputer(
                                            missing_values=self.missing_values,
                                            strategy=self.initial_strategy)
            X_filled = self.initial_imputer_.fit_transform(X)
        else:
            X_filled = self.initial_imputer_.transform(X)

        if self.Xt_previous is not None:
            X_filled = self.Xt_previous	
    
        valid_mask = np.flatnonzero(np.logical_not(
            np.isnan(self.initial_imputer_.statistics_)))
        Xt = X[:, valid_mask]
        mask_missing_values = mask_missing_values[:, valid_mask]
        return Xt, X_filled, mask_missing_values
    
class Model:
    """
    Assumes data are Missing at Random (MAR) -- missingness only depends on
    observed variables, not unobserved variables.

    Attributes
    ----------

    """

    def __init__(self, df, target,
                 block_length=100,
                 estimator=BayesianRidge(),
                 missing_values=np.nan,
                 max_iter=10,
                 tol=1e-3,
                 n_jobs=None,
                 seed = 23412,
                 auto_convergance = False
                ):
        """
        Parameters
        ---------
        df : DataFrame

        target :  str

        block_length : int
            Length of block in days to use for block bootstrap resampling.
        """
        self.data = df
        self.target = target
        self.target_col = self.data.columns == self.target

        self.n_jobs = n_jobs

        self._imputer_kwargs = {
            'estimator' : estimator,
            'missing_values' : missing_values,
            'max_iter' : max_iter,
            'tol' : tol,
            #'sample_posterior' : True,
        }

        if auto_convergance:
            self.auto_converge()


        #self.observed_index = ~self.data[self.target].isna()
        #self._observed_data = self.data[self.observed_index]
        #self._unobserved_data = self.data[~self.observed_index]
        #self._bbootstrap = BBS(self.data[self.observed_index],
        #                       block_length=block_length)


    def auto_converge(self):
        """ Tunes the number of iterations until the imputer converges.


        Notes
        -----
        In practice, this number is usually larger than the number of
        iterations necessary to stabilize a Markov Chain imputer. In other
        words, use at least this number of burn-in iterations before attempting
        to use imputed data.
        """
        converged = False

        while not converged:
            imp = IterativeImputer(**self._imputer_kwargs)
            
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=ConvergenceWarning)
                try:
                    imp.fit(self.data)
                    converged = True

                except ConvergenceWarning:
                    self._imputer_kwargs['max_iter'] *= 10

    def burn_in(self):
        imp = IterativeImputer(**self._imputer_kwargs, sample_posterior=False)
        # burn in
        # TODO


    def bbs_replicate(self, seed=None):
        """ Return a block bootstrap sample.

        Parameters
        ----------
        seed : int
            seed for random number generator

        Returns
        -------
            A DataFrame consisting of a block bootstrap sample of the observed
            samples appended to the unobserved samples.

        """
        # get a block bootstrap sample
        boot_sample = self._bbootstrap.sample(seed=seed)

        return boot_sample.append(self._unobserved_data)


    def fit(self, n=0, seed=None):
        """
        Parameters
        ----------
        n : int
            Number of block bootstrap replicates
        """
        imp = IterativeImputer(**self._imputer_kwargs,
                               random_state=None)
        rows, columns = self.data.shape
        #rows = self.data.shape[0]
        temp = imp.fit_transform(self.data)
        self.result = temp[:,self.target_col].ravel()

        if n != 0:
            self.boot_result = np.zeros([rows, n])

        for i in range(n):
            random_state = 1234567 + i

            imp = IterativeImputer(**self._imputer_kwargs,
                                   random_state=random_state)

            # get block bootstrap sample
            boot_sample = self.bbs_replicate(seed=random_state)

            # create temp dataset including sample
            imputed_sample = imp.fit_transform(boot_sample)

            # impute the result
            self.boot_result[:,i] = imputed_sample[:,self.target_col].ravel()


    def predict(self):
        return  0

    def plot(self):
        """
        """
        pass
