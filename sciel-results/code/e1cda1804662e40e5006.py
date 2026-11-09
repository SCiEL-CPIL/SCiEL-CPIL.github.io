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

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Phase 1: Broad exploration with multiple restarts
        # Use 10 restarts to ensure good coverage
        n_restarts = 10
        # Budget allocation: more balanced, slightly heavier on early exploration
        fractions = [0.10, 0.10, 0.09, 0.09, 0.09, 0.09, 0.08, 0.08, 0.08, 0.10]
        budget_alloc = [int(budget * f) for f in fractions]
        diff = budget - sum(budget_alloc)
        budget_alloc[-1] += diff

        for restart in range(n_restarts):
            b = budget_alloc[restart]
            if b < 5 or calls_used >= budget:
                break

            # Initialize starting point with diverse strategies
            # Use a more uniform spread for the first few restarts to cover the domain
            if restart == 0:
                x0 = np.zeros(dim)
            elif restart == 1:
                x0 = rng.uniform(-4.0, 4.0, dim)
            elif restart == 2:
                x0 = rng.uniform(-4.0, 4.0, dim)
            elif restart == 3:
                x0 = rng.uniform(-4.0, 4.0, dim)
            elif restart == 4:
                x0 = rng.uniform(-3.0, 3.0, dim)
            elif restart == 5:
                x0 = rng.uniform(-3.0, 3.0, dim)
            elif restart == 6:
                x0 = rng.uniform(-2.0, 2.0, dim)
            elif restart == 7:
                x0 = rng.uniform(-2.0, 2.0, dim)
            elif restart == 8:
                x0 = rng.uniform(-1.0, 1.0, dim)
            else:
                x0 = rng.uniform(-1.0, 1.0, dim)

            x = np.clip(x0, -5, 5)
            fx = func(x)
            calls_used += 1
            best_local_x = x.copy()
            best_local_f = fx

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # ASA with adaptive step scaling
            # Start with larger steps for exploration, cool down
            T0 = 2.5 if restart < 5 else 1.5
            T_end = 0.005
            alpha = (T_end / T0) ** (1.0 / max(b - 1, 1))
            T = T0

            # Track recent improvements to adapt temperature
            no_improve_count = 0

            for t in range(b - 1):
                if calls_used >= budget:
                    break

                # Classic ASA direction: u * |v| where u is random unit vector
                u = rng.standard_normal(dim)
                norm_u = np.linalg.norm(u)
                if norm_u > 1e-10:
                    u = u / norm_u
                v = abs(rng.standard_normal(dim))
                perturbation = u * v
                x_new = np.clip(x + T * perturbation, -5, 5)

                fx_new = func(x_new)
                calls_used += 1

                # Metropolis acceptance with adaptive temperature
                if fx_new < fx:
                    improvement = fx - fx_new
                    x = x_new
                    fx = fx_new
                    no_improve_count = 0
                    if fx < best_local_f:
                        best_local_f = fx
                        best_local_x = x.copy()
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    # If improving, keep temperature or increase slightly
                    # This helps escape shallow local minima
                    if improvement > T * 0.1:
                        T = min(T * 1.05, T0)
                else:
                    delta_f = fx_new - fx
                    if delta_f / T < 50:
                        prob = np.exp(-delta_f / T)
                        if rng.random() < prob:
                            x = x_new
                            fx = fx_new
                            no_improve_count += 1
                            if fx < best_local_f:
                                best_local_f = fx
                                best_local_x = x.copy()
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                        else:
                            no_improve_count += 1
                    else:
                        no_improve_count += 1

                # Cool temperature
                T *= alpha
                T = max(T, T_end * 0.5)

                # If no improvement in last 30 steps, boost temperature significantly
                if no_improve_count >= 30:
                    T = min(T * 3.0, T0)
                    no_improve_count = 0

        # Phase 2: Local refinement around best solution
        remaining = budget - calls_used
        if remaining > 3 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            # Start refinement with moderate step
            T = 0.05
            T_end = 0.001
            alpha = (T_end / T) ** (1.0 / max(remaining - 1, 1))
            
            # Use coordinate-wise perturbations mixed with random directions
            for t in range(remaining - 1):
                if calls_used >= budget:
                    break
                
                # Mix of random direction and coordinate perturbations
                if rng.random() < 0.4:
                    # Coordinate-wise perturbation
                    coord = rng.integers(0, dim)
                    perturbation = np.zeros(dim)
                    perturbation[coord] = rng.standard_normal()
                    step_size = T
                else:
                    # Full random direction
                    u = rng.standard_normal(dim)
                    norm_u = np.linalg.norm(u)
                    if norm_u > 1e-10:
                        u = u / norm_u
                    v = abs(rng.standard_normal(dim))
                    perturbation = u * v
                    step_size = T

                x_new = np.clip(x + step_size * perturbation, -5, 5)
                fx_new = func(x_new)
                calls_used += 1
                
                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Keep temperature when improving
                else:
                    delta_f = fx_new - fx
                    if delta_f / T < 50:
                        prob = np.exp(-delta_f / T)
                        if rng.random() < prob:
                            x = x_new
                            fx = fx_new
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()

                T *= alpha
                T = max(T, T_end * 0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
