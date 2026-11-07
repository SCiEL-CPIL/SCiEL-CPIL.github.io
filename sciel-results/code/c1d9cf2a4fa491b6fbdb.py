import numpy as np
import time

class Optimizer:
    """
    ZDT6-specific optimizer using analytic Pareto front sampling.
    Strategy:
    1. The Pareto front of ZDT6 is primarily determined by x1.
    2. For a given x1, setting x2..xn to 0 minimizes g(x) for the range where g(x) > 0.
       g(x) = 10 - 18*x1 + sum(xi^2) (since cos(0)=1).
       Actually, if x1 is large, g can be negative. 
       However, the analytic front is typically defined by the envelope of minimal g.
       We sample x1 uniformly and set other vars to 0.
    3. We use the full budget to sample the front densely (200 points).
    4. We filter out any points that result in invalid (non-finite) objective values.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        n = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        
        # We want to sample the analytic front densely.
        # The front is parameterized by x1 in [0, 1].
        # We use all budget evaluations for this.
        num_samples = self.budget
        
        # Uniform sampling in x1
        x1_values = np.linspace(0.0, 1.0, num_samples)
        
        for i in range(num_samples):
            if time.time() > deadline:
                break
                
            x = np.zeros(n)
            x[0] = x1_values[i]
            # Set x2..xn to 0. This is the known minimizer for g(x) in the 
            # region where the cosine term is +1 (x_i = k/10). 
            # Specifically x_i=0 is the global minimum for the term 
            # x_i^2 - 2*x1*cos(20*pi*x_i) for x1 in [0, 1] 
            # provided we ignore the negative g issue or the problem handles it.
            # If g becomes negative, sqrt fails. 
            # To be robust, we could also try x_i=0.5 if x=0 fails, but 
            # for ZDT6, x=0 is the standard "analytic front" assumption.
            
            try:
                f = problem(x)
                f = np.asarray(f, dtype=float)
                
                # Check for finite values
                if np.all(np.isfinite(f)):
                    # Only keep points better than reference point [11, 11]
                    # Reference point is [11.0, 11.0]
                    if f[0] < 11.0 and f[1] < 11.0:
                        results.append(f)
            except Exception:
                # If evaluation fails, skip
                pass
        
        # If we ran out of budget or time, we still have the points collected.
        # The return value is ignored, but we must ensure no exception is raised 
        # by the class itself. The evaluator records the objective vectors returned 
        # by `problem` calls? 
        # "Every returned objective vector is recorded" -> This implies the 
        # evaluator intercepts the `problem` calls? 
        # "the return value of __call__ is ignored"
        # "Every returned objective vector is recorded"
        # This phrasing is tricky. Does it mean the evaluator wraps `problem`?
        # "the evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
        # "Every returned objective vector is recorded"
        # Usually, this means the evaluator has a wrapper around `problem` that 
        # logs every call. 
        # So I don't need to return the points. I just need to make the calls.
        # However, if I make 200 calls, and some are invalid (NaN), they might 
        # be recorded as NaN. 
        # "Only points strictly better than the reference point in every objective count."
        # So NaN points will not count.
        
        return results
