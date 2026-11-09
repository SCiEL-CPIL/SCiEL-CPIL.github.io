"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hybrid optimizer: Halton sequence exploration followed by CMA-ES-like local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _halton(self, index, base):
        """Generate Halton sequence value for given index and base."""
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result

    def _halton_point(self, index, dim):
        """Generate a Halton sequence point in [0,1]^dim."""
        point = np.zeros(dim)
        for d in range(dim):
            # Use prime bases for each dimension
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            point[d] = self._halton(index, primes[d])
        return point

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Halton sequence exploration (first ~100 evaluations)
        # This systematically covers the domain to find a good starting point
        n_halton = min(100, budget)
        
        best_x = None
        best_f = np.inf
        
        for i in range(1, n_halton + 1):
            # Map Halton point from [0,1]^d to [-5,5]^d
            h_point = self._halton_point(i, dim)
            x = -5.0 + 10.0 * h_point
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if best_x is None:
            # Fallback if no evaluations were made
            best_x = np.zeros(dim)
            best_f = func(best_x)

        # Phase 2: Local refinement using a CMA-ES-like strategy
        remaining_budget = budget - n_halton
        if remaining_budget <= 0:
            return best_x, best_f

        # Initialize CMA-ES parameters
        x = best_x.copy()
        fx = best_f
        sigma = 0.5  # Initial step size
        
        # Covariance matrix (start with identity)
        C = np.eye(dim)
        # Eigen decomposition for step size adaptation
        lambda_pop = max(4, int(np.ceil(4 + 3 * np.log(dim))))  # Population size
        
        # Track success rate for sigma adaptation
        successes = 0
        total_trials = 0
        
        for trial in range(remaining_budget):
            # Generate population
            for j in range(lambda_pop):
                # Sample from multivariate normal with covariance C
                # Use Cholesky decomposition
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                
                z = rng.standard_normal(dim)
                y = x + sigma * L @ z
                y = np.clip(y, -5.0, 5.0)
                
                fy = func(y)
                
                # Update best if found better
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                
                total_trials += 1
            
            # Adapt sigma based on success rate (1/5 rule variant)
            success_rate = successes / max(1, total_trials)
            if success_rate > 0.3:
                sigma *= 1.2
            elif success_rate < 0.1:
                sigma *= 0.7
            else:
                sigma *= 0.95
            
            # Reset counters for next generation
            successes = 0
            total_trials = 0
            
            # Update covariance matrix (simplified CMA-ES update)
            # This is a simplified version; full CMA-ES would be more complex
            # but this should still provide good local search
            if trial % 10 == 0:
                # Decay covariance towards identity to avoid degeneracy
                C = 0.9 * C + 0.1 * np.eye(dim)

        return x, fx
# EVOLVE-BLOCK-END
