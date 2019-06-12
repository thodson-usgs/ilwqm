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

from blockbootstrap import BBS

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
            'sample_posterior' : True,
        }

        self.observed_index = ~self.data[self.target].isna()
        self._observed_data = self.data[self.observed_index]
        self._unobserved_data = self.data[~self.observed_index]
        self._bbootstrap = BBS(self.data[self.observed_index],
                               block_length=block_length)


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
        rows = self.data.shape[0]
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
