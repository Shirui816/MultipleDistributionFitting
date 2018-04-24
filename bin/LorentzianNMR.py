from MultiFuncs import n_func_maker
from FitLSQ import FitLSQ
from Evaluate import Evaluation
import numpy as np
from scipy.integrate import simps
from sys import argv
import warnings

# Define the base function


def lorentzian(x, a, g, d):
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


tol, f_2 = 0.02, 1  # Tolerance of negative value and normalization factor
_n = int(input("Enter maximum number of components: "))
# Generate mixtures of base function
_funcs = [lorentzian] + [n_func_maker(_, lorentzian) for _ in range(2, _n + 1)]
_models = [FitLSQ(_) for _ in _funcs]  # Generate fiting models

datas = np.array([np.loadtxt(_) for _ in argv[1:]])
data = datas.mean(axis=0).T  # Readding data
x, y = data[0], data[1]
f_1 = simps(y, x)
y /= f_1

_shift = input("Enter shift factor of intensity (default 0): ") or 0
_shift = float(_shift)
if _shift != 0:
    _std_g = float(input("Enter Gamma of standard sample: "))
    _std_d = float(input("Enter chemical shift of standard sample: "))
    y -= _shift * lorentzian(x, _shift, _std_g, _std_d)
    if abs(simps(y[y < 0], x[y < 0])) > tol:
        warnings.warn("Y has may negative values, try a smaller shift factor?",
                      UserWarning)

y[y < 0] = 0  # Non-negative
f_2 = simps(y, x)
y /= f_2

# p0 is set for the base function
_gamma = float(input("Enter initial guess of GAMMA: "))
_chemical_shift = float(input("Enter initial guess of DELTA: "))
_models = [_.set_p0([1/(__ + 1), _gamma, _chemical_shift])
           for __, _ in enumerate(_models)]
# bounds are the lower/upper limits of each parmater of base function
_gamma_lb = float(input("Enter lower bound of GAMMA: "))
_gamma_hb = float(input("Enter higher bound of GAMMA: "))
_delta_lb = float(input("Enter lower bound of DELTA: "))
_delta_hb = float(input("Enter higher bound of DELTA: "))
_models = [_.set_bounds([[0, _gamma_lb, _delta_lb], [1, _gamma_hb, _delta_hb]])
           for _ in _models]
_models = [_.fit(x, y) for _ in _models]

_eva = [Evaluation(_) for _ in _models]

_m = input("Enter MC trials number (default 10): ") or 10
_o = input("Enter MC sample size (default 3000): ") or 3000
_m, _o = int(_m), int(_o)

_pmfs = np.array([_.T[1] for _ in datas])
_xs = np.array([_.T[0] for _ in datas])
_pmfs[_pmfs < 0] = 0

aic = np.zeros(_n)
bic = np.zeros(_n)
aicc = np.zeros(_n)
lih = np.zeros(_n)
for __ in range(_m):
    samples = np.array([Evaluation.make_sample(_o, _x, _pmf)
                        for _x, _pmf in zip(_xs, _pmfs)])
    sample = samples.flatten()
    bic += np.array([_.bic(sample) for _ in _eva])
    aicc += np.array([_.aicc(sample) for _ in _eva])
    aic += np.array([_.aic(sample) for _ in _eva])
    lih += np.array([_.score(sample) for _ in _eva])

bic /= _m
aicc /= _m
aic /= _m
lih /= _m
_best = _models[np.argmin(bic)]
BIC = ["The BIC of %03d is %.4f" % (_+1, __) for _, __ in enumerate(bic)]
AICc = ["The AICc of %03d is %.4f" % (_+1, __) for _, __ in enumerate(aicc)]
AIC = ["The AIC of %03d is %.4f" % (_+1, __) for _, __ in enumerate(aic)]
LIH = ["The Lih of %03d is %.4f" % (_+1, __) for _, __ in enumerate(lih)]
len_ = max([len(_) for _ in BIC + AICc + LIH + AIC]) + 1
print("\n"*3)
print("="*len_)
print(" "*(len_ // 2 - 3) + "Lih")
print('-'*len_)
[print(_) for _ in LIH]
print("="*len_)
print("\n"*3)
print("="*len_)
print(" "*(len_ // 2 - 3) + "AIC")
print('-'*len_)
[print(_) for _ in AIC]
print("="*len_)
print("\n"*3)
print("="*len_)
print(" "*(len_ // 2 - 4) + "AICc")
print('-'*len_)
[print(_) for _ in AICc]
print("="*len_)
print("\n"*3)
print("="*len_)
print(" "*(len_ // 2 - 3) + "BIC")
print('-'*len_)
[print(_) for _ in BIC]
print("="*len_)
print("\n"*3)
print("n_components with AIC is %d" % (np.argmin(aic) + 1))
print("n_components with AICc is %d" % (np.argmin(aicc) + 1))
print("n_components with BIC is %d" % (np.argmin(bic) + 1))
print("="*len_)
print("The optimized parmaters are: ")
[print("%s:\t%.4f" % (_, __)) for _, __ in zip(_best.P_, _best.Parameters)]
print("="*len_)
if f_2 != 1:
    print("The normalization factor is %.4f, original is %.4f" % (f_2, f_1))
np.savetxt('out.txt', np.vstack([x, _best.Function(x, *_best.Parameters)]).T)
