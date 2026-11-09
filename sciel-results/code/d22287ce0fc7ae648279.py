"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with LHS initialization and periodic restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _latin_hypercube(self, rng, n_samples, dim):
        """Generate n_samples points via Latin Hypercube Sampling in [-5, 5]^dim."""
        points = np.zeros((n_samples, dim))
        for d in range(dim):
            perm = rng.permutation(n_samples)
            points[:, d] = (perm + 0.5) / n_samples * 10.0 - 5.0
        return points

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290.0

        # Phase 1: Latin Hypercube sampling for fast early progress
        n_init = 20
        lhd = self._latin_hypercube(rng, n_init, dim)
        
        best_x = lhd[0].copy()
        best_f = func(lhd[0])
        calls = 1

        for i in range(1, n_init):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            fi = func(lhd[i])
            calls += 1
            if fi < best_f:
                best_f = fi
                best_x = lhd[i].copy()

        # Phase 2: (1+1)-ES with periodic restarts
        # Katsuura has many local minima; periodic restarts with larger sigma
        # help escape local basins while (1+1)-ES efficiently refines.
        
        sigma = 1.5
        no_improve = 0
        restart_interval = 30  # restart every 30 calls if no improvement
        current_x = best_x.copy()
        current_f = best_f

        while calls < budget:
            if time.time() > deadline:
                break

            y = np.clip(current_x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < current_f:
                current_x, current_f = y, fy
                sigma *= 1.1
                no_improve = 0
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            else:
                sigma *= 0.95
                no_improve += 1

            # Periodic restart from global best with larger sigma
            if no_improve >= restart_interval:
                # Restart from global best with larger sigma to escape local minima
                current_x = best_x.copy()
                current_f = best_f
                sigma = max(sigma * 2.0, 1.0)
                no_improve = 0
                # Occasionally try a completely new random point
                if rng.random() < 0.3:
                    new_x = rng.uniform(-5, 5, dim)
                    new_f = func(new_x)
                    calls += 1
                    if new_f < best_f:
                        best_f = new_f
                        best_x = new_x.copy()
                    current_x = new_x.copy()
                    current_f = new_f

            # Safety check
            if calls >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
