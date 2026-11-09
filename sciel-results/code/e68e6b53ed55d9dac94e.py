"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Adaptive Simulated Annealing (ASA) for BBOB f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # We'll do multiple restarts with ASA
        # Strategy: divide budget among several restarts
        # For dim=5, Griewank-Rosenbrock is multimodal but not too harsh
        # Use 3-5 restarts

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Determine number of restarts based on budget
        n_restarts = 4
        budget_per_restart = budget // n_restarts

        for restart in range(n_restarts):
            # Allocate remaining budget to last restart
            if restart == n_restarts - 1:
                b = budget - calls_used
            else:
                b = budget_per_restart

            if b < 10:
                break

            # Initialize starting point
            # For first restart, use a somewhat structured initialization
            if restart == 0:
                # Start near origin or use a few candidate points
                x0 = np.zeros(dim)
            else:
                # Random restarts, biased toward center
                x0 = rng.uniform(-3, 3, dim)
            
            # Evaluate initial point
            x = np.clip(x0, -5, 5)
            fx = func(x)
            calls_used += 1
            best_local_x = x.copy()
            best_local_f = fx

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # ASA parameters
            # Temperature schedule: T = T0 * alpha^t or similar
            # For ASA, we use: x_new = x + sigma * (random_direction * |random_scalar|)
            # where sigma is controlled by temperature
            
            T0 = 3.0  # initial "temperature" (step size scale)
            T_end = 0.01
            alpha = (T_end / T0) ** (1.0 / max(b - 1, 1))
            T = T0

            for t in range(b - 1):
                if calls_used >= budget:
                    break

                # Generate ASA perturbation
                # Standard ASA: x_new = x + T * (u * |v|) where u is uniform direction, v is normal
                # Or simpler: x_new = x + T * normal_vector, with T decreasing
                
                # Use a mix: primarily normal perturbation scaled by T
                perturbation = rng.standard_normal(dim)
                x_new = np.clip(x + T * perturbation, -5, 5)
                
                # ASA-specific: sometimes use a larger exploratory step
                # The classic ASA uses: x_{k+1} = x_k + T_k * (random_unit_vector * |N(0,1)|)
                # Let's use a simpler but effective variant
                
                fx_new = func(x_new)
                calls_used += 1

                # Metropolis acceptance criterion
                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                    if fx < best_local_f:
                        best_local_f = fx
                        best_local_x = x.copy()
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                else:
                    # Accept with probability exp(-delta_f / T)
                    delta_f = fx_new - fx
                    if delta_f > 0:
                        # Avoid overflow
                        if delta_f / T < 50:
                            prob = np.exp(-delta_f / T)
                        else:
                            prob = 0.0
                        if rng.random() < prob:
                            x = x_new
                            fx = fx_new
                            if fx < best_local_f:
                                best_local_f = fx
                                best_local_x = x.copy()
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()

                # Cool down
                T *= alpha
                T = max(T, T_end * 0.5)

        # If we have budget left, do a final local refinement around best_x
        remaining = budget - calls_used
        if remaining > 5 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            T = 0.1
            alpha = (0.005 / T) ** (1.0 / max(remaining - 1, 1))
            for t in range(remaining - 1):
                if calls_used >= budget:
                    break
                perturbation = rng.standard_normal(dim)
                x_new = np.clip(x + T * perturbation, -5, 5)
                fx_new = func(x_new)
                calls_used += 1
                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    delta_f = fx_new - fx
                    if delta_f > 0 and delta_f / T < 50:
                        prob = np.exp(-delta_f / T)
                        if rng.random() < prob:
                            x = x_new
                            fx = fx_new
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                T *= alpha
                T = max(T, 0.003)

        return best_x, best_f
# EVOLVE-BLOCK-END
