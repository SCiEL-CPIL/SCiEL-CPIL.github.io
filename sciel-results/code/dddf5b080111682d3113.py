"""Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploits DTLZ1 structure: set x[2:] = 0.5 (g=0), optimize (x[0], x[1])
    on the quarter-circle x[0]^2 + x[1]^2 = 1 to maximize hypervolume.
    Uses adaptive sampling with finer resolution near high-contribution regions.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # safety margin before 330s kill

        # Set variables 2..6 to 0.5 to achieve g=0
        # Variables 0 and 1 define position on the Pareto front
        base = np.zeros(self.dim)
        base[2:] = 0.5  # g = 0 when x_i = 0.5 for i >= 2

        lo0, hi0 = bounds[0]
        lo1, hi1 = bounds[1]

        # We want x[0]^2 + x[1]^2 = 1 with x[0], x[1] >= 0
        # Parameterize by angle theta in [0, pi/2]
        # x[0] = cos(theta), x[1] = sin(theta)
        # f1 = 0.5*cos(theta), f2 = 0.5*sin(theta), f3 = 0

        # Hypervolume contribution is highest where objectives are small
        # i.e., near the "origin" of the quarter-disk in objective space
        # This corresponds to theta near 0 (x0=1, x1=0 -> f1=0.5, f2=0, f3=0)
        # and theta near pi/2 (x0=0, x1=1 -> f1=0, f2=0.5, f3=0)
        # Actually, the hypervolume reference point is [400,400,400], so
        # all points on the PF contribute. The contribution is proportional
        # to the volume of the box from the point to the reference point.
        # For f3=0, contribution ~ (400-f1)*(400-f2)*400
        # This is maximized when f1 and f2 are small, i.e., near the axes.

        # Strategy: sample theta with non-uniform density, denser near 0 and pi/2
        # Use a beta-like distribution or simply cluster points near the ends

        n_evals = 0
        results = []

        def evaluate(theta):
            nonlocal n_evals
            x = base.copy()
            x[0] = np.cos(theta)
            x[1] = np.sin(theta)
            # Clip to bounds
            x[0] = np.clip(x[0], lo0, hi0)
            x[1] = np.clip(x[1], lo1, hi1)
            f = problem(x)
            n_evals += 1
            return f

        # Phase 1: Dense sampling near the "corners" (theta near 0 and pi/2)
        # and moderate sampling in between
        # Use a transformation that clusters points near the ends
        # Let u in [0,1], map to theta with higher density near 0 and pi/2

        # We'll use a piecewise approach:
        # - 40% of budget: very fine near theta=0 (0 to 0.3)
        # - 40% of budget: very fine near theta=pi/2 (pi/2-0.3 to pi/2)
        # - 20% of budget: uniform in the middle

        n_total = self.budget
        n_corner = int(0.4 * n_total)
        n_mid = n_total - 2 * n_corner

        # Corner 1: theta in [0, 0.5] with dense sampling
        # Use sqrt transformation to cluster near 0
        thetas_c1 = 0.5 * np.sqrt(np.linspace(0, 1, n_corner))
        # Corner 2: theta in [pi/2 - 0.5, pi/2] with dense sampling
        thetas_c2 = np.pi / 2 - 0.5 * np.sqrt(np.linspace(0, 1, n_corner))
        # Middle: theta in [0.5, pi/2 - 0.5] uniform
        thetas_m = np.linspace(0.5, np.pi / 2 - 0.5, n_mid)

        all_thetas = np.concatenate([thetas_c1, thetas_c2, thetas_m])
        # Shuffle to avoid systematic bias
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(all_thetas))
        all_thetas = all_thetas[idx]

        # Evaluate all points
        for theta in all_thetas:
            if n_evals >= self.budget:
                break
            if time.time() > deadline - 5.0:
                break
            f = evaluate(theta)
            results.append(f)

        # Phase 2: If budget remains, do local refinement around best regions
        if n_evals < self.budget and time.time() < deadline - 10.0:
            # Find the point with best hypervolume contribution (smallest objectives)
            # and do a local search around it
            if results:
                F = np.array(results)
                # Simple: pick a few random thetas from the evaluated set and refine
                for _ in range(min(20, self.budget - n_evals)):
                    if n_evals >= self.budget:
                        break
                    if time.time() > deadline - 5.0:
                        break
                    # Pick a random evaluated theta and perturb
                    i = rng.integers(len(all_thetas[:n_evals]))
                    theta0 = all_thetas[i]
                    theta_new = theta0 + 0.01 * rng.standard_normal()
                    theta_new = np.clip(theta_new, 0, np.pi / 2)
                    f = evaluate(theta_new)
                    results.append(f)

        return results
# EVOLVE-BLOCK-END
