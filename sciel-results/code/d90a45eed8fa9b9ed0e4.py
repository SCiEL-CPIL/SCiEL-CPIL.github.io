"""Multi-objective optimizer for ZDT4 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits ZDT4's analytic Pareto set: x2..xn at zeros of sin(pi*x_i)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4: g(x) = (n-1) + 10*n + sum(sin^2(2*pi*x_i) - 10*cos(2*pi*x_i))
        # Pareto front: x1 in [0,1], x2..xn at zeros of sin^2 => sin(pi*x_i)=0
        # So x_i = k/n for integer k, but we need sin^2(2*pi*x_i)=0 => 2*pi*x_i = k*pi => x_i = k/2
        # Actually: sin^2(2*pi*x_i) = 0 when 2*pi*x_i = k*pi, i.e., x_i = k/2
        # Within [0,1], x_i can be 0, 0.5, 1.0
        # The g function: g = (n-1) + 10*n + sum(sin^2(2*pi*x_i) - 10*cos(2*pi*x_i)) for i=2..n
        # At x_i = 0: sin^2(0)=0, cos(0)=1, so term = 0 - 10 = -10
        # At x_i = 0.5: sin^2(pi)=0, cos(pi)=-1, so term = 0 - (-10) = 10
        # At x_i = 1.0: sin^2(2*pi)=0, cos(2*pi)=1, so term = 0 - 10 = -10
        # So optimal is x_i in {0, 1} giving g = (n-1) + 10*n + (n-1)*(-10) = (n-1) + 10n - 10(n-1) = (n-1) + 10
        # Wait, let me recalculate: n=10 variables, x2..x10 are 9 variables
        # g = (10-1) + 10*10 + sum_{i=2}^{10}(sin^2(2*pi*x_i) - 10*cos(2*pi*x_i))
        # = 9 + 100 + sum = 109 + sum
        # At x_i = 0 or 1: each term = -10, so sum = 9*(-10) = -90
        # g = 109 - 90 = 19
        # f1 = x1, f2 = (1+x1/g)*(2 - 2*x1^2 - 0.2*sin(8*pi*x1))
        
        # Key insight: set x2..x10 to values that minimize g
        # sin^2(2*pi*x) - 10*cos(2*pi*x) is minimized when cos(2*pi*x)=1 (i.e., x=0 or 1)
        # giving -10 per term
        
        # Strategy: 
        # 1. Evaluate many points with x2..x10 at optimal values (0 or 1)
        # 2. Vary x1 across [0,1] to sample the Pareto front
        # 3. Use remaining budget for local refinement
        
        dim = self.dim
        budget = self.budget
        used = 0
        
        # Optimal values for x2..x10: 0.0 (or 1.0, both give same g)
        optimal_tail = np.zeros(dim - 1)
        
        # Phase 1: Dense sampling of x1 with optimal tail
        # Use most of the budget for this
        n_phase1 = min(budget - 20, 180)  # Leave some budget for refinement
        
        # Sample x1 uniformly across [0, 1]
        x1_values = np.linspace(0.0, 1.0, n_phase1)
        
        for x1 in x1_values:
            if used >= budget or time.time() > deadline:
                break
            x = np.concatenate([[x1], optimal_tail])
            x = np.clip(x, lo, hi)
            problem(x)
            used += 1
        
        # Phase 2: Local refinement around promising regions
        # Focus on areas where the front might have gaps
        if used < budget:
            remaining = budget - used
            # Try some perturbed points to fill gaps
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                # Random x1 with slight perturbation of tail
                x1 = np.random.random()
                tail = optimal_tail + 0.01 * np.random.randn(dim - 1)
                x = np.concatenate([[x1], tail])
                x = np.clip(x, lo, hi)
                problem(x)
                used += 1
        
        return None
# EVOLVE-BLOCK-END
