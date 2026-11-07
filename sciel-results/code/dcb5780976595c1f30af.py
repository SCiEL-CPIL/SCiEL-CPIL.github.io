"""Black-box continuous optimizer on BBOB f12 (Bent cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-strategy optimizer: CMA-ES-like with coordinate descent polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # 300s limit, 10s safety margin

        # State
        best_x = None
        best_f = np.inf
        calls_made = 0

        def eval_point(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: Multi-restart CMA-ES-like evolution strategy ---
        # Use a population-based ES with adaptive step size
        rng = np.random.default_rng(42)

        # Number of restarts based on budget
        # Each restart uses a fraction of the budget
        num_restarts = 3
        budget_per_restart = budget // num_restarts

        for restart in range(num_restarts):
            if calls_made >= budget or time.time() > deadline:
                break

            # Initialize
            if restart == 0:
                x = rng.uniform(-5, 5, dim)
            else:
                # Perturb best found so far
                if best_x is not None:
                    x = np.clip(best_x + rng.standard_normal(dim) * 0.5, -5, 5)
                else:
                    x = rng.uniform(-5, 5, dim)

            fx = eval_point(x)
            if best_x is None:
                best_x = x.copy()
                best_f = fx

            sigma = 3.0
            # Population size for CMA-ES-like
            lam = 4 + int(3 * np.log(dim))  # ~10 for dim=5
            # Success rate tracking
            successes = 0
            trials = 0
            target_success = 1 / (lam + 1)

            remaining_in_restart = budget_per_restart - (calls_made - (budget - budget_per_restart * (num_restarts - restart)))
            # Simpler: just track calls used in this restart
            calls_in_restart = 0
            max_calls_in_restart = budget_per_restart

            while calls_in_restart < max_calls_in_restart and calls_made < budget:
                if time.time() > deadline:
                    break

                # Generate offspring
                for i in range(lam):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    calls_in_restart += 1

                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = eval_point(y)

                    # Update if better than current x
                    if fy < fx:
                        x = y.copy()
                        fx = fy
                        successes += 1
                    trials += 1

                    # Adaptive sigma
                    if trials > 0:
                        success_rate = successes / trials
                        if success_rate > target_success:
                            sigma *= 1.05
                        elif success_rate < target_success:
                            sigma *= 0.95
                        sigma = np.clip(sigma, 1e-6, 10.0)

                # Reset success tracking periodically
                if trials > 20:
                    successes = 0
                    trials = 0

                # Check for stagnation
                if sigma < 1e-5:
                    break

        # --- Phase 2: Coordinate descent polish ---
        if best_x is not None and calls_made < budget and time.time() < deadline:
            # Use remaining budget for coordinate descent
            remaining = budget - calls_made
            # Allocate some for coordinate descent, some for local random search
            cd_budget = min(remaining, 200)
            remaining -= cd_budget

            # Coordinate descent with golden section search
            x = best_x.copy()
            fx = best_f

            # Determine step size from recent progress or use adaptive
            step = 0.5

            for _ in range(cd_budget):
                if calls_made >= budget or time.time() > deadline:
                    break

                improved = False
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        break

                    # Golden section search along dimension d
                    # Try a few steps in both directions
                    for direction in [1, -1]:
                        for mult in [1.0, 0.5, 0.25, 0.1, 0.05]:
                            if calls_made >= budget or time.time() > deadline:
                                break
                            trial = x.copy()
                            trial[d] = np.clip(trial[d] + direction * step * mult, -5, 5)
                            f_trial = eval_point(trial)
                            if f_trial < fx:
                                x = trial.copy()
                                fx = f_trial
                                improved = True
                                step *= 0.5
                                break  # Found improvement, shrink step
                        if improved:
                            break

                if not improved:
                    # Try larger steps
                    step *= 2.0
                    if step > 5.0:
                        step = 5.0
                    # Try random perturbations in all dims
                    for _ in range(5):
                        if calls_made >= budget or time.time() > deadline:
                            break
                        trial = np.clip(x + rng.standard_normal(dim) * step, -5, 5)
                        f_trial = eval_point(trial)
                        if f_trial < fx:
                            x = trial.copy()
                            fx = f_trial
                            step *= 0.5
                            break

            best_x = x.copy()
            best_f = fx

        # --- Phase 3: Final local random search with very small steps ---
        if best_x is not None and calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            x = best_x.copy()
            fx = best_f
            sigma_local = 0.01

            for _ in range(remaining):
                if calls_made >= budget or time.time() > deadline:
                    break
                trial = np.clip(x + sigma_local * rng.standard_normal(dim), -5, 5)
                f_trial = eval_point(trial)
                if f_trial < fx:
                    x = trial.copy()
                    fx = f_trial
                    sigma_local *= 1.1
                else:
                    sigma_local *= 0.9
                sigma_local = max(sigma_local, 1e-8)

            best_x = x.copy()
            best_f = fx

        return best_x, best_f

# EVOLVE-BLOCK-END
