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

from pandas import DatetimeIndex

from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LassoCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from blockbootstrap import BBS

class Model:
    """
    Assumes data are Missing at Random (MAR) -- missingness only depends on
    observed variables, not unobserved variables.

    Attributes
    ----------
    bcf : bool
        Whether to apply a bias correction factor to output

    """

    def __init__(self, df, target,
                 block_length=100,
                 missing_values=np.nan,
                 max_iter=10,
                 tol=1e-3,
                 n_jobs=None,
                 seed = 23412,
                 block_length=100,
                 imputer_model=BayesianRidge(),
                 prediction_model=None
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
        self.block_length = block_length
        #self.target_col = self.data.columns == self.target
        #self.bcf = bcf
        self.n_jobs = n_jobs

        if prediction_model is None:
            self._predicter = LassoCV(n_jobs=n_jobs, cv=5, normalize=True)

        if imputer_model is None:
            self._imputer = BayesianRidge()

        if self.n_jobs:
            raise TypeError('n_jobs not implemented')

        self._imputer_kwargs = {
            estimator : estimator,
            missing_values : missing_values,
            max_iter : max_iter,
            tol : tol,
            sample_posterior : True,
        }

        self.observed_index = self.data[self.target].isna()
        #self._observed_data = self.data[self.observed_index]
        #self._unobserved_data = self.data[~self.observed_index]

        #self._bbootstrap = BBS(self.data[self.observed_index],
        #                       block_length=block_length)


    def predict(self, imputer_seed=None, bootstrap_seed=None):
        # impute the data
        imp = IterativeImputer(**self._imputer_kwargs,
                               random_state=imputer_seed)

        data_m = imp.fit_transform(self.transformed_data())

        # resample the data
        samples_m = data_m[self.observed_index]
        bbs = BBS(samples_m, block_length=100, freq='D')
        samples_mb = bbs.sample(seed=bootstrap_seed)

        # fit a model to the resampled data

        # predict results from the fitted model
        observations_m = data_m.loc[:, data_m.columns != self.target]


    def _test_input(self):
        assert isinstance(self.data.index, DatetimeIndex), "Data must have a datetime index"


    def transformed_data(self):
        """ Log transform input data and add seasonal columns
        """
        data = self.data.copy()
        data = np.log(data)
        data['t'] = data.index.to_julian_date() / 365.25 #days per year
        data['cost'] = 2 * np.pi * np.cos(data['t'])
        data['sint'] = 2 * np.pi * np.sin(data['t'])

        return data


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
        if seed is not None:
            # add a large constitent, so incrementing seed +1 will work.
            seed = seed + 1234567:


        imp = IterativeImputer(**self._imputer_kwargs,
                               random_state=seed)
        col = 1
        rows = self.data.shape[0]
        temp = imp(self.data)
        self.result = temp[:,col]

        if n != 0:
            self.boot_result = np.zeros([rows, n])

        for i in range(n):
            random_state = seed + n

            imp = IterativeImputer(**self._imputer_kwargs,
                                   random_state=random_state)

            # get block bootstrap sample
            boot_sample = bbs_replicate(seed=random_state)

            # create temp dataset including sample
            imputed_sample = imp.fit_transform(boot_sample)

            # impute the result
            self.boot_result[:,n] = imputed_sample[:,self.target_col]


    def plot(self):
        """
        """
        pass
