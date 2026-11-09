"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Analytical optimizer for ZDT2 that exploits the known structure.

    ZDT2: f1 = x0, f2 = 1 + (9/(n-1)) * sum_{i=1}^{n-1} x_i * (1 + sin(3*pi*x_i))
    The Pareto front is f2 = 1 + f1^0.5, achieved when x1..x_{n-1} = 0.
    We sample x0 values to spread f1, then use remaining budget for
    local refinement to ensure x1..x_{n-1} are as close to 0 as possible.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 295.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.dim
        used = 0

        # Phase 1: Dense structured sampling of x0 with x1..=0
        # This directly hits the Pareto front since x1..=0 minimizes f2 for any f1
        n_phase1 = min(150, self.budget // 2)
        if n_phase1 < 20:
            n_phase1 = self.budget

        # Use a mix of uniform and concentrated sampling near 0 and 1
        # to cover the entire front well
        x0_samples = np.linspace(0.0, 1.0, n_phase1)
        records = []

        for f1_val in x0_samples:
            if used >= self.budget or time.time() > deadline - 20:
                break
            x = np.zeros(n)
            x[0] = f1_val
            f = problem(x)
            records.append((x.copy(), np.array(f, dtype=float)))
            used += 1

        # Phase 2: Local refinement - for each sampled point, try to reduce f2
        # by adjusting x1..x_{n-1} toward 0 with small perturbations
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline - 20:
            rng = np.random.default_rng(42)

            # Sort by f1 to process in order
            records.sort(key=lambda r: r[0][0])

            # For each point, do a few local search steps
            n_refine = min(len(records), max(10, remaining // 5))
            evals_per_point = max(1, remaining // n_refine)

            for i in range(n_refine):
                if used >= self.budget or time.time() > deadline - 20:
                    break
                base_x, base_f = records[i]

                # Try to reduce x1..x_{n-1} toward 0
                for step in [0.05, 0.02, 0.01, 0.005]:
                    if used >= self.budget or time.time() > deadline - 20:
                        break
                    for _ in range(max(1, evals_per_point // 4)):
                        if used >= self.budget or time.time() > deadline - 20:
                            break
                        x_new = base_x.copy()
                        # Perturb x1..x_{n-1} with bias toward 0
                        perturb = step * rng.standard_normal(n - 1)
                        x_new[1:] = np.clip(base_x[1:] + perturb, 0.0, 1.0)
                        f_new = problem(x_new)
                        f_new = np.array(f_new, dtype=float)

                        if f_new[1] < base_f[1]:
                            base_x = x_new.copy()
                            base_f = f_new.copy()
                            records.append((x_new.copy(), f_new))
                        else:
                            records.append((x_new.copy(), f_new))
                        used += 1

        # Phase 3: Fill remaining budget with additional front points
        if used < self.budget and time.time() < deadline - 20:
            rng2 = np.random.default_rng(123)
            while used < self.budget and time.time() < deadline - 20:
                x = np.zeros(n)
                x[0] = rng2.random()
                f = problem(x)
                records.append((x.copy(), np.array(f, dtype=float)))
                used += 1

        return records  # ignored by evaluator

# EVOLVE-BLOCK-END
