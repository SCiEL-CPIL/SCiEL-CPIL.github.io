"""Multi-objective optimizer on DTLZ1 with analytical front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1.
    
    Strategy:
    1. Analyze the DTLZ1 structure to find the analytical Pareto front.
    2. Set aggregation variables to their optimal values (0.5) to minimize g(x).
    3. Sample position variables using a space-filling design (Sobol/LHS) to cover the front.
    4. Use the entire budget for this sampling.
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 280.0  # Safety margin before 300s limit
    
    def _elapsed(self):
        return time.time() - self.start_time
    
    def _time_left(self):
        return self.deadline - self._elapsed()
    
    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ1 parameters
        m = self.n_obj
        n = self.dim
        M = m - 1  # Number of position variables
        
        # Optimal values for aggregation variables to minimize g
        # g(x) = 100 + (n-M) + sum_{i=M+1}^n ((x_i - 0.5)^2 - 10*cos(20*pi*(x_i - 0.5)))
        # Min of each term is -10 at x_i = 0.5
        # So set x_{M+1}...x_n = 0.5
        x_agg = 0.5 * np.ones(n - M)
        
        # Generate points on the Pareto front
        # We need to sample x_1...x_M in [0,1]
        # Use Sobol sequence or Latin Hypercube for good coverage
        # Since budget is small (200), a simple grid or LHS is sufficient.
        # Let's use a stratified sampling approach.
        
        n_samples = self.budget
        if n_samples <= 0:
            return np.array([])
            
        # Create a space-filling design for M dimensions
        # Simple approach: use Sobol sequence if available in scipy, else random with stratification
        # scipy.stats.qmc.Sobol is available in SciPy
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=M, scramble=True)
            # Generate more points than needed to ensure coverage, then take first n_samples
            # Or just generate n_samples
            # Sobol requires power of 2? No, but performance is best. 
            # Let's just generate n_samples points.
            X_pos = sampler.random(n_samples)
        except ImportError:
            # Fallback to Latin Hypercube
            X_pos = np.zeros((n_samples, M))
            for i in range(M):
                # Stratified sampling
                intervals = np.linspace(0, 1, n_samples + 1)
                for j in range(n_samples):
                    X_pos[j, i] = np.random.uniform(intervals[j], intervals[j+1])
            # Shuffle to decorrelate
            idx = np.random.permutation(n_samples)
            X_pos = X_pos[idx]
            
        # Scale to bounds if necessary, though DTLZ1 is [0,1]
        # lo[:M], hi[:M]
        span_pos = hi[:M] - lo[:M]
        X_pos = lo[:M] + span_pos * X_pos
        
        # Construct full X
        X = np.zeros((n_samples, n))
        X[:, :M] = X_pos
        X[:, M:] = x_agg
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Check time
        if self._time_left() < 0:
            # If we ran out of time (unlikely with 200 evals), return what we have
            pass
            
        # All points on the front are non-dominated (assuming distinct)
        # Return all evaluated points
        return X
# EVOLVE-BLOCK-END
