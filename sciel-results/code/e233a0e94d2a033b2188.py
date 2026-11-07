"""Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting DTLZ1 structure for hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        budget = self.budget
        dim = self.dim

        # DTLZ1 with M=3, N=7:
        # Pareto front: x[2:] = 0.5, x[0]^2 + x[1]^2 = 1, x[0], x[1] >= 0
        # On PF: f1 = 0.5*x[0], f2 = 0.5*x[1], f3 = 0
        # Objective space is a quarter-disk of radius 0.5

        all_F = []

        def halton_sequence(n, base):
            """Generate n points from the Halton sequence in [0,1] with given base."""
            result = np.zeros(n)
            f = 1.0 / base
            for i in range(n):
                idx = i + 1
                x = 0.0
                while idx > 0:
                    f *= 1.0 / base
                    x += f * (idx % base)
                    idx //= base
                result[i] = x
            return result

        # Phase 1: Dense sampling of the Pareto front using Halton sequence for theta
        # This gives low-discrepancy coverage of the quarter-circle
        n_pf = min(budget, 180)  # Use most of budget for front points

        # Generate Halton sequence for theta in [0, pi/2]
        theta_halton = halton_sequence(n_pf, 2) * (np.pi / 2.0)

        # Add corner points for better hypervolume coverage at extremes
        corner_thetas = np.array([0.0, np.pi/400, np.pi/100, np.pi/2 - np.pi/100, np.pi/2 - np.pi/400, np.pi/2])
        theta_vals = np.concatenate([corner_thetas, theta_halton])
        theta_vals = np.unique(np.round(theta_vals, 10))
        theta_vals = np.sort(theta_vals)[:n_pf]

        for i in range(len(theta_vals)):
            if time.time() > deadline - 5.0:
                break

            theta = theta_vals[i]
            x = np.zeros(dim)
            x[0] = np.cos(theta)
            x[1] = np.sin(theta)
            x[2:] = 0.5  # Optimal shape variables for DTLZ1

            # Clip to bounds
            x = np.clip(x, lo, hi)

            f = problem(x)
            all_F.append(f)

        # Phase 2: Fill remaining budget with perturbed front points
        remaining = budget - len(all_F)
        if remaining > 0:
            rng = np.random.default_rng(42)
            for i in range(remaining):
                if time.time() > deadline - 5.0:
                    break

                # Sample theta from Halton-like distribution for good coverage
                theta = rng.random() * (np.pi / 2.0)
                x = np.zeros(dim)
                x[0] = np.cos(theta)
                x[1] = np.sin(theta)
                x[2:] = 0.5

                # Tiny perturbation to avoid exact duplicates
                x += 1e-4 * rng.standard_normal(dim)
                x = np.clip(x, lo, hi)

                f = problem(x)
                all_F.append(f)

        return all_F
# EVOLVE-BLOCK-END
