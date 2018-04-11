#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: shirui <shirui816@gmail.com>

import numpy as np


class Evaluation(object):
    r"""Evaluation of model."""

    def __init__(self, model):
        r"""Initialize with model.

        Arguments:
        model: a fit object
        """
        self.model = model

    @classmethod
    def make_sample(cls, n, x, pdf):
        r"""Make random sample taken from x.

        Arguments:
        n: int, sample size
        x: np.ndarray
        pdf: np.ndarray

        Returns:
        sample
        """
        pdf /= np.sum(pdf)
        return np.random.choice(x, size=n, p=pdf)

    # This func is already a sum of functions
    def _log_prob(self, x):
        r"""Calculate log probability.

        Arguments:
        x: samples of (n_samples, n_features)

        Returns:
        probability: np.ndarray
        """
        return np.log(self.model.Function(x, *self.model.Parameters) /
                      self.model.NormalFactor)

    def aic(self, x):
        r"""Calculate AIC.

        Aho, K.; Derryberry, D.; Peterson, T. (2014), "Model selection for
        ecologists: the worldviews of AIC and BIC", Ecology, 95: 631–636,
        doi:10.1890/13-1452.1.

        AIC = 2k - 2\ln{\hat{\mathcal{L}}}, \hat{\mathcal{{L}}} is Likelihood.

        Arguments:
        samples: samples of (n_samples, n_features)

        Returns:
        aic: np.ndarray
        """
        return 2 * self.model.N_ -\
            2 * self.score(x) * x.shape[0]

    def bic(self, x):
        r"""Calculate BIC.

        Schwarz, Gideon E. (1978), "Estimating the dimension of a model",
        Annals of Statistics, 6 (2): 461–464, doi:10.1214/aos/1176344136,
        MR 0468014.

        BIC = \ln{N}k - 2\ln{\hat{\mathcal{L}}}

        Arguments:
        samples: samples of (n_samples, n_features)

        Returns:
        bic: np.ndarray
        """
        return self.model.N_ * np.log(x.shape[0]) -\
            2 * self.score(x) * x.shape[0]

    def aicc(self, x):
        r"""Calculate AICc.

        deLeeuw, J. (1992), "Introduction to Akaike (1973) information theory
        and an extension of the maximum likelihood principle" (PDF),
        in Kotz, S.; Johnson, N.L., Breakthroughs in Statistics I, Springer,
        pp. 599–609.

        AICc = AIC + \frac{2k^2+2k}{N-k-1}

        Arguments:
        samples: samples of (n_samples, n_features)

        Returns:
        bic: np.ndarray
        """
        return 2 * self.model.N_ * np.log(x.shape[0]) -\
            2 * self.score(x) * x.shape[0] +\
            2 * (self.model.N_ ** 2 + self.model.N_) /\
            (x.shape[0] - self.model.N_ - 1)

    def score(self, x):
        r"""Calculate Likelyhood.

        Arguments:
        samples: samples of (n_samples, n_features)

        Returns:
        likelihood: np.ndarray
        """
        return self._log_prob(x).mean()
