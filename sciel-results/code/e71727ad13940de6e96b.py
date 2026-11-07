"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0 # 20s safety margin before 300s limit
        
        best_x = None
        best_f = float('inf')
        call_count = 0
        rng = np.random.default_rng(42)

        # Helper to evaluate and update best
        def evaluate(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget:
                return float('inf')
            x = np.clip(x, -5, 5)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Nelder-Mead implementation
        def nelder_mead(start_x, max_iter=50):
            nonlocal call_count, deadline
            dim = self.dim
            # Initial simplex
            simplex = [start_x.copy()]
            f_simplex = [evaluate(start_x)]
            
            # Generate other vertices
            for i in range(dim):
                x_i = start_x.copy()
                # Small perturbation
                if start_x[i] > 4.0:
                    x_i[i] -= 0.5
                else:
                    x_i[i] += 0.5
                simplex.append(x_i)
                f_simplex.append(evaluate(x_i))
            
            # Sort initial simplex
            order = np.argsort(f_simplex)
            simplex = [simplex[i] for i in order]
            f_simplex = [f_simplex[i] for i in order]
            
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            it = 0
            while it < max_iter:
                if time.time() > deadline or call_count >= budget:
                    break
                
                # 1. Reflection
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                
                centroid = np.sum(simplex[:-1], axis=0) / dim
                x_r = centroid + alpha * (centroid - x_worst)
                f_r = evaluate(x_r)
                
                if f_r < f_simplex[0]:
                    # 2. Expansion
                    x_e = centroid + gamma * (centroid - x_worst)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # 3. Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        f_c = evaluate(x_c)
                        if f_c < f_r:
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                f_simplex[i] = evaluate(simplex[i])
                            # Re-sort
                            order = np.argsort(f_simplex)
                            simplex = [simplex[i] for i in order]
                            f_simplex = [f_simplex[i] for i in order]
                            break # Shrink often indicates convergence or trouble, restart logic handles it
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - x_worst)
                        f_c = evaluate(x_c)
                        if f_c < f_worst:
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                f_simplex[i] = evaluate(simplex[i])
                            # Re-sort
                            order = np.argsort(f_simplex)
                            simplex = [simplex[i] for i in order]
                            f_simplex = [f_simplex[i] for i in order]
                            break

                # Sort simplex
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                # Convergence check
                if abs(f_simplex[0] - f_simplex[-1]) < 1e-8:
                    break
                it += 1

            return simplex[0], f_simplex[0]

        # Multi-start strategy
        # We allocate budget to multiple starts.
        # 10 starts, roughly 50 calls each for NM (simplex size 6 + iterations)
        num_starts = 10
        calls_per_start = budget // num_starts
        
        # First start at 0
        start_points = [np.zeros(dim)]
        # Add random starts
        for _ in range(num_starts - 1):
            start_points.append(rng.uniform(-4, 4, dim))
            
        # Shuffle starts to avoid bias if time is tight
        rng.shuffle(start_points)
        
        for sp in start_points:
            if time.time() > deadline or call_count >= budget:
                break
            # Run NM
            # Limit iterations to ensure we don't blow budget in one start
            # NM uses dim+1 initial calls + iterations * 1-2 calls
            # With budget 500 and 10 starts, ~50 calls/start. 
            # Initial 6. Remaining 44. Max iter ~20 is safe.
            nelder_mead(sp, max_iter=20)
            
        return best_x, best_f

# EVOLVE-BLOCK-END
