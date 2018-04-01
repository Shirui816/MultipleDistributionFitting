#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: shirui <shirui816@gmail.com>


from scipy.optimize import curve_fit
import numpy as np
import inspect
import warnings


class FitLSQ(object):
    r"""Least Square fitting model."""

    def __init__(self, func):
        r"""Init function.

        Arguments:
        func: callable, function to be fitted
        """
        self.Parameters = None
        self.Covariances = None
        self.InitParameters = None
        self.Bounds = None
        self.Function = func
        self._N, self._P = self.__get_paramter()

    def __get_paramter(self):
        r"""Get number of parameters.

        Returns: tuple, (n_parameters, parameters)
        """
        args = inspect.getfullargspec(self.Function).args
        # -1 for 'x' is always the 1st value.
        return len(args) - 1, args

    def set_bounds(self, bounds, known=[]):
        r"""Set bounds for target function.

        Arguments:
        bounds: 2d-list for lower and upper bounds (lb, ub) for arguments
                of base function. +/-np.inf for no bounds.
        n: number of parameters ofl BASE functions.
        known: Known parts in functions.
        Returns:
        self
        """
        warnings.warn("Bounds must EXACTLY match the base function!",
                      UserWarning)
        lb, ub = bounds
        # number of bounds must equal to the base function n_parameters' number
        N = self._N + len(known) - known.count(None)
        NBases = N // len(lb)
        Unknow = NBases - len(known) // len(lb)
        assert isinstance(bounds, list), "Bounds must be list object."
        assert len(ub) == len(lb) and N % len(ub) == 0,\
            "Number of bounds must equal to number of BASE's arguments!"
        _lb = [lb[_ % len(lb)]
               for _ in range(len(known)) if known[_] is None] + lb * Unknow
        _ub = [ub[_ % len(ub)]
               for _ in range(len(known)) if known[_] is None] + ub * Unknow
        self.Bounds = [_lb, _ub]
        return self

    def set_p0(self, p0, known=[]):
        r"""Set initial values for fitting.

        Arguments:
        p0: tuple or list for initial parameters.
        known: list for known components.

        Returns:
        self
        """
        warnings.warn("p0 must EXACTLY match the base function!",
                      UserWarning)
        N = self._N + len(known) - known.count(None)
        NBases = N // len(p0)
        Unknow = NBases - len(known) // len(p0)
        assert isinstance(p0, list), "p0 must be list object."
        # number of bounds must equal to the base function n_parameters' number
        assert N % len(p0) == 0,\
            "Number of initials must equal to number of BASE arguments!"
        _p0 = [p0[_ % len(p0)]
               for _ in range(len(known)) if known[_] is None] + p0 * Unknow
        self.InitParameters = _p0
        return self

    def fit(self, x, y, **kwargs):
        r"""Fit the model.

        Arguments:
        x: np.array for x
        y: np.array for y

        Keyword Arguments:
        kwargs that fits scipy.optimize.curve_fit

        Returns:
        self
        """
        p0 = self.InitParameters or None
        bounds = self.Bounds or (-np.inf, np.inf)
        self.Parameters, self.Covariances =\
            curve_fit(self.Function, x, y, p0=p0, bounds=bounds, **kwargs)
        return self
