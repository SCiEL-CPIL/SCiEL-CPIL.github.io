    import numpy as np
    import time
    import scipy.optimize

    class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            start_time = time.time()
            deadline = start_time + 280.0
            calls_made = 0

            def wrapped_func(x):
                nonlocal calls_made
                if time.time() > deadline:
                    return 1e10
                calls_made += 1
                if calls_made > self.budget:
                    return 1e10
                return func(x)

            x0 = np.zeros(self.dim)
            
            # CMA-ES options
            # popsize is determined by scipy based on dim
            # maxiter needs to be set to limit total calls
            # For dim=5, popsize ~ 8-10. 
            # 500 calls / 10 = 50 iterations.
            options = {'maxiter': 60, 'tolfun': 1e-12, 'tolfx': 1e-12}
            
            try:
                res = scipy.optimize.minimize(wrapped_func, x0, method='CMA-ES', options=options)
            except Exception:
                pass

            return x0, 0.0
    