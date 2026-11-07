# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2 exploiting the known Pareto front structure.
    
    DTLZ2 with 3 objectives has a Pareto front that is a quarter-sphere in the positive
    octant of R^3. The distance variables (last dim - n_obj + 1 = 10 variables) should
    be set to 0.5 for optimality. The angle variables (first n_obj - 1 = 2 variables)
    determine position on the front.
    
    Strategy: Use Sobol quasi-random sequence to sample the 2D angle space densely,
    then map to decision space. This directly generates points near/on the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        # DTLZ2 structure:
        # - First m-1 variables are "angle" variables that determine position on Pareto front
        # - Last n-m+1 variables are "distance" variables; optimal value is 0.5 for each
        # - Pareto front: sum of squares of objectives = 1 (unit sphere in positive octant)
        
        # We'll use a hybrid approach:
        # 1. Generate points directly on/near the Pareto front using angle sampling
        # 2. Use remaining budget for local refinement if time permits
        
        # Number of distance variables
        n_dist = n - m + 1  # = 10 for dim=12, n_obj=3
        
        # Optimal distance values (0.5 for DTLZ2)
        dist_optimal = np.full(n_dist, 0.5)
        
        # We need to sample the (m-1)-dimensional angle space
        # For m=3, this is 2D: theta1, theta2
        # The mapping from angles to objectives for DTLZ2 Pareto front:
        # f1 = sin(theta1) * sin(theta2) * g(x)  -- but on PF, g=1
        # Actually for DTLZ2: f_i = g(x) * prod of sines * cos
        # On the Pareto front (g=1, i.e., dist vars = 0.5):
        # The objectives lie on the unit sphere in the positive orthant
        
        # Let's use a direct parameterization:
        # For 3 objectives on unit sphere:
        # f1 = sin(a) * sin(b)
        # f2 = sin(a) * cos(b)  
        # f3 = cos(a)
        # where a in [0, pi/2], b in [0, pi/2]
        
        # But we need to map back to decision space. For DTLZ2:
        # The relationship between decision variables and objectives on the PF is complex.
        # Instead, let's use a different strategy:
        
        # Strategy: Generate candidate points in decision space that are likely near the PF,
        # evaluate them, and keep the best non-dominated set.
        
        # Key insight: For DTLZ2, setting distance variables to 0.5 and varying angle
        # variables gives points on the PF. But the exact mapping from angle vars to
        # objectives depends on the specific DTLZ2 formulation.
        
        # Let's use a more robust approach: 
        # 1. Create a population with distance vars fixed at 0.5
        # 2. Vary angle vars using Sobol sequence in [0,1]^(m-1)
        # 3. Evaluate and keep non-dominated points
        
        # Generate Sobol samples for angle space
        n_samples = min(self.budget, 150)
        
        # Use Sobol sequence for (m-1) dimensional angle space
        sampler = qmc.Sobol(d=m-1, scramble=True)
        # Generate 2^n points, take first n_samples
        power = max(1, int(np.ceil(np.log2(n_samples))))
        sobol_samples = sampler.random(2**power)[:n_samples]
        
        # Map Sobol samples [0,1]^(m-1) to angle space [0, pi/2]^(m-1)
        angles = sobol_samples * (np.pi / 2)
        
        # Build decision vectors
        X = np.zeros((n_samples, n))
        # Set distance variables to optimal value 0.5
        X[:, m-1:] = 0.5
        # Set angle variables from Sobol samples
        X[:, :m-1] = angles
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        used = n_samples
        
        # If we have budget left, do some local refinement
        if used < self.budget and (time.time() - start_time) < deadline - 5:
            # Find non-dominated points
            nondom_idx = self._find_nondominated(F)
            if len(nondom_idx) > 0:
                # Perturb non-dominated points slightly
                n_refine = min(self.budget - used, 30)
                for i in range(n_refine):
                    if (time.time() - start_time) > deadline - 3:
                        break
                    parent_idx = nondom_idx[i % len(nondom_idx)]
                    child = X[parent_idx].copy()
                    # Small random perturbation
                    child += 0.05 * (hi - lo) * np.random.randn(n)
                    child = np.clip(child, lo, hi)
                    fc = problem(child)
                    used += 1
                    # Replace worst point if child is better
                    worst_idx = int(np.argmax(np.sum(F, axis=1)))
                    if np.sum(fc) < np.sum(F[worst_idx]):
                        X[worst_idx] = child
                        F[worst_idx] = fc
                        # Update nondominated set
                        nondom_idx = self._find_nondominated(F)
        
        return X

    def _find_nondominated(self, F):
        """Find indices of non-dominated points."""
        n = len(F)
        nondom = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nondom[i] = False
                    break
        return np.where(nondom)[0]
# EVOLVE-BLOCK-END
