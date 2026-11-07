"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize


class Optimizer:
    """
    Multi-objective optimizer for WFG4 (12 dims, 3 objectives).
    Uses a hybrid approach:
      1. Generate diverse initial points using deterministic + random sampling.
      2. Use scalarized sub-problems (Tchebycheff / weighted sum) with local
         coordinate-descent / Nelder-Mead refinement to explore the Pareto front.
      3. Maintain a non-dominated set and use hypervolume contribution as a
         guide for selecting which points to refine further.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # Reference point for hypervolume (from task)
        ref = np.array([3.0, 5.0, 7.0])

        # We will record all evaluated objective vectors
        all_F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.asarray(problem(x), dtype=float)
            eval_count += 1
            all_F.append(f)
            return f

        # --- Step 1: Initial diverse sampling ---
        # Use a mix of random points and a small grid / Latin hypercube
        n_init = min(40, self.budget // 5)
        init_X = []
        # Random
        n_rand = n_init // 2
        for _ in range(n_rand):
            init_X.append(lo + span * rng.random(self.dim))
        # Quasi-random (Halton-like via simple deterministic spread)
        n_det = n_init - n_rand
        for i in range(n_det):
            # Spread along a diagonal pattern
            t = (i + 1) / (n_det + 1)
            x = np.zeros(self.dim)
            for d in range(self.dim):
                # Different phase per dimension
                x[d] = lo[d] + span[d] * ((t * (d + 1) + 0.5 / (d + 1)) % 1.0)
            init_X.append(x)

        for x in init_X:
            evaluate(x)

        # --- Step 2: Scalarized optimization with local refinement ---
        # Use Tchebycheff weights to target different parts of the Pareto front
        # For 3 objectives, use a set of weight vectors
        n_subproblems = 12  # number of scalarized sub-problems
        # Generate weight vectors on a simplex
        weights = []
        # Simple grid on simplex for 3 objectives
        for w1 in np.linspace(0.05, 0.9, 4):
            for w2 in np.linspace(0.05, 0.9 - w1, 4):
                w3 = 1.0 - w1 - w2
                if w3 >= 0.05:
                    weights.append([w1, w2, w3])
        # Shuffle and take n_subproblems
        rng.shuffle(weights)
        weights = weights[:n_subproblems]

        # For each weight vector, define a scalarized objective
        # Tchebycheff: max_i (w_i * (f_i - z_i^min))
        # First estimate z^min from initial evaluations
        F_arr = np.array(all_F)
        z_min = F_arr.min(axis=0)

        # Local refinement via coordinate descent with ternary search
        def scalarized_obj(x, w):
            f = evaluate(x)
            # Tchebycheff scalarization
            return np.max(w * (f - z_min))

        # We'll use a simple local search: for each dimension, do a small
        # perturbation search (not full ternary to save evaluations)
        def local_refine(x0, w, max_evals=10):
            """Refine x0 using coordinate-wise perturbation search."""
            x = x0.copy()
            f0 = evaluate(x)
            scalar0 = np.max(w * (f0 - z_min))
            remaining = max_evals
            for _ in range(remaining // self.dim + 1):
                if remaining <= 0:
                    break
                best_x = x.copy()
                best_scalar = scalar0
                for d in range(self.dim):
                    if remaining <= 0:
                        break
                    # Try a few perturbations in dimension d
                    for delta in [0.05 * span[d], -0.05 * span[d],
                                  0.02 * span[d], -0.02 * span[d]]:
                        if remaining <= 0:
                            break
                        x_trial = best_x.copy()
                        x_trial[d] += delta
                        x_trial = np.clip(x_trial, lo, hi)
                        f_trial = evaluate(x_trial)
                        remaining -= 1
                        scalar_trial = np.max(w * (f_trial - z_min))
                        if scalar_trial < best_scalar:
                            best_scalar = scalar_trial
                            best_x = x_trial.copy()
                x = best_x
            return x

        # Start each subproblem from a random or initial point
        for w in weights:
            if eval_count >= self.budget:
                break
            # Start from a random point
            x0 = lo + span * rng.random(self.dim)
            f0 = evaluate(x0)
            # Refine
            x_refined = local_refine(x0, w, max_evals=8)

        # --- Step 3: Final phase - use remaining budget for global exploration ---
        # If we have budget left, do more random sampling + local refinement
        while eval_count < self.budget:
            # Pick a random starting point
            x0 = lo + span * rng.random(self.dim)
            f0 = evaluate(x0)
            if eval_count >= self.budget:
                break
            # Quick local refinement with a random weight
            w = rng.random(self.n_obj)
            w /= np.sum(w)
            x_refined = local_refine(x0, w, max_evals=5)

        return all_F
# EVOLVE-BLOCK-END
