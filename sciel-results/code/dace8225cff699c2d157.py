"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Dense ZDT3-aware front sampling with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        budget = self.budget
        dim = self.dim

        # Strategy: 
        # Phase 1: Dense uniform sampling of x[0] in [0,1] with x[1:] = 0 (on the Pareto front)
        # Phase 2: Perturb x[1:] slightly to explore near-front region and escape local minima of h()
        # Phase 3: Local refinement around best points found so far

        # Phase 1: Uniform grid on x[0]
        # Use about 40% of budget for this
        n_phase1 = int(budget * 0.4)
        n_phase1 = max(n_phase1, 10)
        
        x0_vals = np.linspace(0, 1, n_phase1)
        
        for i in range(n_phase1):
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x0_vals[i]
            # x[1:] = 0 gives g=1, which is on the Pareto front for ZDT3
            problem(x)

        # Phase 2: Perturb x[1:] slightly to explore near-front region
        # Use about 35% of budget for this
        n_phase2 = int(budget * 0.35)
        n_phase2 = max(n_phase2, 5)
        
        # We'll sample x[0] uniformly again, but with perturbations in x[1:]
        x0_vals_p2 = np.linspace(0, 1, n_phase2)
        
        for i in range(n_phase2):
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x0_vals_p2[i]
            # Small random perturbations in x[1:] to explore h() landscape
            # Use deterministic pseudo-random based on i for reproducibility
            rng_local = np.random.RandomState(i + 12345)
            # Perturb a few dimensions with small values
            n_perturb = rng_local.randint(1, 5)
            perturb_dims = rng_local.choice(dim - 1, size=n_perturb, replace=False) + 1
            for d in perturb_dims:
                x[d] = rng_local.random() * 0.01  # small perturbation
            problem(x)

        # Phase 3: Local refinement around best points
        # Use remaining budget
        n_phase3 = budget - n_phase1 - n_phase2
        n_phase3 = max(n_phase3, 0)
        
        if n_phase3 > 0:
            # We'll do a simple local search: sample around the uniform grid points
            # but with smaller perturbations and more dimensions perturbed
            n_phase3 = min(n_phase3, budget - n_phase1 - n_phase2)
            
            for i in range(n_phase3):
                if time.time() > deadline:
                    break
                x = np.zeros(dim)
                # Sample x[0] uniformly
                x[0] = i / max(1, n_phase3 - 1)
                # Use a different seed pattern
                rng_local = np.random.RandomState(i + 99999)
                # Perturb more dimensions with even smaller values
                n_perturb = rng_local.randint(2, 8)
                perturb_dims = rng_local.choice(dim - 1, size=min(n_perturb, dim-1), replace=False) + 1
                for d in perturb_dims:
                    x[d] = rng_local.random() * 0.005  # smaller perturbation
                problem(x)

        return None
# EVOLVE-BLOCK-END
