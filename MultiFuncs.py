import inspect
import numpy as np

# Decorator


def n_func(n, known=[]):
    r"""Decorator for defining functions mixed by base function.

    For scipy.optimize.curv_fit.

    Arguments:
    n: numbers of functions
    Keyword arguments:
    known: optional, known componets of functions,
            n_{knowns} \times n_{function parameters}

    Returns: Function that mixed n base functions with known parameters.
    """
    def _wrapper(func):
        signature_ = inspect.getfullargspec(func).args
        # For scipy fitting
        assert signature_[0] == 'x',\
            "The function must has `x' for its 1st value!"
        # make sure that functions like f(x) appear...
        # what would you want to fit?
        assert len(signature_) > 1, "........Seriously?"
        f_args = signature_[1:]
        Nf, Nk = len(f_args), len(known)
        assert Nk % Nf == 0,\
            "Paramerters of known components must equal to target function!"
        assert len(known) <= n * Nf,\
            "Number of known parts must lesser than number of functions!"
        if not (len(known) != n * Nf or None in known):
            # I am sure that no parameters are put for fitting
            raise ValueError("Seriously?")
        # for f_{1,2,3}(x,a, b), transfrom a -> a0 b0, a1 b1, a2 b2
        nf_args = [f_args[_ % Nf] + str(_ // Nf + Nk // Nf)
                   for _ in range(n * Nf - Nk)]
        # same operation, but for the known part
        nf_args = [f_args[_ % Nf] + str(_ // Nf)
                   for _ in range(Nk) if known[_] is None] + nf_args
        # print(nf_args)
        _signature = [inspect.Parameter(
            'x', kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        _signature = _signature + \
            [inspect.Parameter(
                _, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for _ in nf_args]

        def __wrapper(x, *args):
            assert len(args) == len(
                nf_args), "Number of args must equal to the unknowns!"
            args = list(args)
            # this is amazing
            # [ None, a, b, None ] + [c,d] -> [c,a,b,d], a fancy substitution
            variables = [args.pop(0) if _ is None else _ for _ in known] + args
            return np.sum([func(x, *variables[_:_ + Nf])
                           for _ in range(0, len(variables), Nf)], axis=0)

        __wrapper.__signature__ = inspect.Signature(parameters=_signature)
        return __wrapper
    return _wrapper


# maker
def n_func_maker(n, func, known=[]):
    r"""Maker for defining functions mixed by base function.

    For scipy.optimize.curv_fit.

    Arguments:
    n: numbers of functions
    func: callable, base function
    Keyword arguments:
    known: optional, known componets of functions,
            n_{knowns} \times n_{function parameters}

    Returns: Function that mixed n base functions with known parameters.
    """
    signature_ = inspect.getfullargspec(func).args
    assert signature_[0] == 'x', "The function must has `x' for its 1st value!"
    assert len(signature_) > 1, "........Seriously?"
    f_args = signature_[1:]
    Nf, Nk = len(f_args), len(known)
    assert Nk % Nf == 0,\
        "Paramerters of known components must equal to target function!"
    assert len(known) <= n * Nf,\
        "Number of known parts must lesser than number of functions!"
    nf_args = [f_args[_ % Nf] + str(_ // Nf + Nk // Nf)
               for _ in range(n * Nf - Nk)]
    nf_args = [f_args[_ % Nf] + str(_ // Nf)
               for _ in range(Nk) if known[_] is None] + nf_args
    # print(nf_args)
    _signature = [inspect.Parameter(
        'x', kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    _signature = _signature + \
        [inspect.Parameter(_, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for _ in nf_args]

    def _func(x, *args):
        assert len(args) == len(
            nf_args), "Number of args must equal to the unknowns!"
        args = list(args)
        variables = [args.pop(0) if _ is None else _ for _ in known] + args
        return np.sum([func(x, *variables[_:_ + Nf])
                       for _ in range(0, len(variables), Nf)], axis=0)

    _func.__signature__ = inspect.Signature(parameters=_signature)
    return _func


# addr
def flatten(l):
    r"""Flatten 2d list to 1d.

    Arguments:
    l:  2D list
    Returns: flattened list
    """
    return [item for sublist in l for item in sublist]


def n_func_mix(funcs):
    r"""Mixer for defining functions mixed by base function.

    For scipy.optimize.curv_fit.

    Arguments:
    funcs: A list of callables
    func: callable, base function

    Returns: Function that mixed n base functions.
    """
    signature_ = [inspect.getfullargspec(_).args for _ in funcs]
    lengths = [len(_) - 1 for _ in signature_]
    assert 0 not in lengths, "........Seriously?"
    headers = [_[0] for _ in signature_]
    assert list(set(headers)) == [
        'x'], "All functions must have `x' for their 1st parameters!"
    _p = np.cumsum([0] + lengths).astype(np.int)
    signature_ = [__ + str(_) for _ in range(len(funcs))
                  for __ in signature_[_] if 'x' not in __]
    _signature = [inspect.Parameter(
        'x', kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    _signature = _signature + \
        [inspect.Parameter(_, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for _ in signature_]

    def _func(x, *args):
        assert len(args) == len(_signature) - \
            1, "Number of args must equal to the unknowns!"
        return np.sum([funcs[_](x, *args[_p[_]:_p[_ + 1]])
                       for _ in range(0, len(_p) - 1)], axis=0)

    _func.__signature__ = inspect.Signature(parameters=_signature)
    return _func
