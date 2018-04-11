
import warnings
import numpy as np
import pandas as pd
from utils import n_func_maker
from utils import FitLSQ
from utils import Evaluation
from scipy.integrate import trapz

np.set_printoptions(precision=6, suppress=True)


# Define the base function


def _lorentzian(x, a, g, d):
    r"""Lorentzian distribution fucntion with weigth a.

    Arguments:
    x: np.ndarray
    a: weight
    g: gamma of Lorentzian function
    d: \delta_0 of Lorentzian fucnntion

    Returns:
    np.ndarray, weighted Lorentzian fucntion
    """
    return a * 1/(np.pi) * g / ((x - d) ** 2 + g ** 2)


class NMRFitting(object):
    r"""Fitting NMR datas."""

    def __init__(self, files, components_range,
                 n_mc_trials=10, n_samples=3000, shift=0, tol=0.01):
        r"""Initialize.

        Arguments:
        files: a list of files of NMR datas
        components_range: a touple of the range of how many peaks
        n_mc_trials: default is 10. times that finding BIC
        n_samples: default is 3000. samples used to find BIC
        shift: default is 0. Set shift if you want to remove some components.
        tol: Tolerance of ratio of negative areas after shift.
        """
        self._models = [FitLSQ(n_func_maker(_, _lorentzian))
                        for _ in range(*components_range)]
        datas = np.array([pd.read_csv(_, header=None,
                                      squeeze=1, delim_whitespace=True,
                                      comment='#').values
                          for _ in files])
        data = datas.mean(axis=0).T
        x, y = data[0], data[1]
        self.area_1 = trapz(y, x)
        y /= self.area_1
        if shift != 0:
            _std_g = float(input("Enter Gamma of standard sample: "))
            _std_d = float(input("Enter chemical shift of standard sample: "))
            y -= shift * _lorentzian(x, shift, _std_g, _std_d)
            if abs(trapz(y[y < 0], x[y < 0])) > tol:
                warnings.warn("Y has may negative values, "
                              "try a smaller shift factor?", UserWarning)
        y[y < 0] = 0
        self.area_2 = trapz(y, x)
        y /= self.area_2
        self._n_mc_sample = (n_mc_trials, n_samples)
        self._data = data
        self._datas = datas
        self._R = components_range

    def set_p0_bounds(self, p0=(0.5, 0.002, 3.7),
                      bounds=((0, 1e-4, 3.5), (1, 1e-1, 4.1))):
        r"""Set p0 and bounds, defaults are for PEG.

        Arguments:
        p0: 1-d touple or list for area, peak_width and chemical shift
        bounds: 2-d touple or list for the lower/upper value of area,
                peak_width and chemical shift. +/-np.inf for no bounds.

        Returns:
        self
        """
        self._models = [_.set_p0(p0)
                        for __, _ in enumerate(self._models)]
        # bounds are the lower/upper limits of each parmater of base
        # function
        self._models = [_.set_bounds(bounds) for _ in self._models]
        return self

    def fitting(self):
        r"""Fitting method."""
        x, y = self._data[0], self._data[1]
        _models = [_.fit(x, y) for _ in self._models]
        _eva = [Evaluation(_) for _ in _models]
        _pmfs = np.array([_.T[1] for _ in self._datas])
        _xs = np.array([_.T[0] for _ in self._datas])
        _pmfs[_pmfs < 0] = 0
        _n = len(_eva)
        # AIC AICc BIC LIH: 0 1 2 3
        ret = np.zeros((4, _n))
        _n_mc, _n_sample = self._n_mc_sample
        for __ in range(_n_mc):
            samples = np.array([Evaluation.make_sample(_n_sample, _x, _pmf)
                                for _x, _pmf in zip(_xs, _pmfs)])
            sample = samples.flatten()
            ret[0] += np.array([_.aic(sample) for _ in _eva])
            ret[1] += np.array([_.aicc(sample) for _ in _eva])
            ret[2] += np.array([_.bic(sample) for _ in _eva])
            ret[3] += np.array([_.score(sample) for _ in _eva])
        ret /= _n_mc
        _best_aic = _models[np.argmin(ret[0])]
        _best_aicc = _models[np.argmin(ret[1])]
        _best_bic = _models[np.argmin(ret[2])]
        print("Best estamation by AIC is %d\nThe parameters are: %s" %
              (self._R[0] + np.argmin(ret[0]), _best_aic.Parameters))
        print("Best estamation by AICc is %d\nThe parameters are: %s" %
              (self._R[0] + np.argmin(ret[1]), _best_aicc.Parameters))
        print("Best estamation by BIC is %d\nThe parameters are: %s" %
              (self._R[0] + np.argmin(ret[2]), _best_bic.Parameters))
        print("The normalization factor is %.4f, the original is %.4f" %
              (self.area_2, self.area_1))
        np.savetxt('by_aic.txt',
                   np.vstack([x,
                              _best_aic.Function(x, *_best_aic.Parameters)]).T)
        np.savetxt('by_aicc.txt',
                   np.vstack([x, _best_aicc.Function(x,
                             *_best_aicc.Parameters)]).T)
        np.savetxt('by_bic.txt',
                   np.vstack([x,
                              _best_bic.Function(x, *_best_bic.Parameters)]).T)
        o = open('AIC_AICc_BIC_LIH.txt', 'w')
        o.write('#n_components\tAIC\tAICc\tBIC\tLIH\n')
        o.close()
        o = open('AIC_AICc_BIC_LIH.txt', 'a')
        np.savetxt(o, np.hstack([np.arange(*self._R)[:, np.newaxis], ret.T]),
                   fmt=['%d', '%.6f', '%.6f', '%.6f', '%.6f'])
        o.close()
        return ret, _best_aic, _best_aicc, _best_bic
