# MultipleDistributionFitting
Finding optimized number of components from mixed distribution data.

Process:

1. Define target function(s): MultiFuncs.py -> n_func, n_func_mix, n_func_maker
2. Create fitting model(s): FitLSQ.py -> FitLSQ
3. Evaluation the model by AIC, AICc, BIC: Evaluation.py -> Evaluation
4. Choose the model that minimizes the BIC, AICc or AIC
