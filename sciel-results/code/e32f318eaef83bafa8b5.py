"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Memetic Differential Evolution with Hypervolume-based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _dominates(self, a, b):
        """Check if vector a dominates vector b."""
        return np.all(a <= b) and np.any(a < b)

    def _non_dominated_filter(self, points):
        """Filter a list of points to keep only non-dominated ones."""
        if not points:
            return []
        nd_points = []
        for p in points:
            is_dominated = False
            for q in nd_points:
                if self._dominates(q, p):
                    is_dominated = True
                    break
            if not is_dominated:
                # Check if p dominates any existing point, remove them
                nd_points = [q for q in nd_points if not self._dominates(p, q)]
                nd_points.append(p)
        return nd_points

    def _hypervolume_contribution(self, point, front, ref):
        """
        Approximate the hypervolume contribution of a point to the front.
        For 3 objectives, we can calculate the exact volume added by this point
        relative to the reference point, assuming the front is sorted or small.
        This is a heuristic proxy: volume of the box defined by point and ref,
        minus volumes of boxes dominated by other points in the front.
        """
        # Simple proxy: Product of (ref - point)
        # This is not exact hypervolume contribution but correlates well
        # for small fronts and uniform distribution.
        return np.prod(ref - point)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim, n_obj = self.dim, self.n_obj
        ref = np.array([1.1, 1.3, 1.1])

        # Population size
        pop_size = 50
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # DE Parameters
        F_de = 0.8  # Differential weight
        CR = 0.9    # Crossover rate

        while used < self.budget:
            if time.time() > deadline:
                break

            # Select a random individual to mutate
            i = rng.integers(pop_size)
            x_i = X[i]
            f_i = F[i]

            # DE/rand/1: x_trial = x_i + F_de * (x_a - x_b)
            a, b = rng.choice(pop_size, 2, replace=False)
            while a == i or b == i:
                a, b = rng.choice(pop_size, 2, replace=False)
            
            v = x_i + F_de * (X[a] - X[b])
            
            # Crossover
            j_rand = rng.integers(dim)
            trial = np.empty(dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = x_i[j]
            
            # Clip to bounds
            trial = np.clip(trial, lo, hi)
            
            f_trial = problem(trial)
            used += 1

            # Selection: Compare trial with individual i
            # If trial dominates i, replace i
            if self._dominates(f_trial, f_i):
                X[i] = trial
                F[i] = f_trial
            # If i dominates trial, keep i
            elif self._dominates(f_i, f_trial):
                pass
            # If they are non-dominated, use hypervolume contribution to decide
            else:
                # Calculate approximate hypervolume contribution of trial and i
                # We need to check against the current non-dominated set or just the neighbor
                # For efficiency, we compare against the current individual and a few neighbors
                # A more robust approach: maintain an archive of non-dominated points
                
                # Let's maintain a simple archive of non-dominated points
                # To keep it simple and fast, we just replace if the trial has a better
                # "proxy" hypervolume contribution than the current individual,
                # considering the local context.
                
                # Better approach: If non-dominated, replace if trial is "more diverse"
                # or has better hypervolume proxy.
                # Let's use the product of (ref - f) as a proxy for potential HV contribution.
                # Points closer to the ideal point (0,0,0) but still dominated by others 
                # might have lower proxy but higher actual contribution if they fill gaps.
                
                # Simple heuristic: Replace if the trial has a lower sum of objectives
                # (closer to ideal) OR if it expands the front.
                # Since we don't have the full front here, let's just keep the one with
                # lower sum of objectives as a tie-breaker, or random.
                # Actually, a better tie-breaker for HV is to keep the one that is
                # further from the other non-dominated points.
                
                # Let's just keep the current one to avoid premature convergence
                # unless the trial is significantly better in some objective.
                # Or, we can do a simple local search around the better of the two.
                
                # For now, let's replace with probability 0.5 to maintain diversity
                if rng.random() < 0.5:
                    X[i] = trial
                    F[i] = f_trial

            # Periodic random restart to escape local optima
            if used % 50 == 0 and used < self.budget:
                rand_idx = rng.integers(pop_size)
                X[rand_idx] = lo + (hi - lo) * rng.random(dim)
                F[rand_idx] = problem(X[rand_idx])
                used += 1

        return X
# EVOLVE-BLOCK-END
