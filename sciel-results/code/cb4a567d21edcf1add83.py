# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using grid-based decomposition with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 260  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        evaluated_points = []
        evaluated_f = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget or time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            evaluated_points.append(x.copy())
            evaluated_f.append(f.copy())
            eval_count += 1
            return f

        # Phase 1: Coarse grid sampling for systematic coverage
        # Use 3 points per dimension for a 3^4 = 81 point grid
        n_per_dim = 3
        grid_points = []
        for i in range(n_per_dim ** self.dim):
            coords = []
            temp = i
            for d in range(self.dim):
                level = temp % n_per_dim
                temp //= n_per_dim
                if level == 0:
                    coords.append(lo[d])
                elif level == 1:
                    coords.append((lo[d] + hi[d]) / 2.0)
                else:
                    coords.append(hi[d])
            grid_points.append(np.array(coords))

        # Evaluate grid points
        for x in grid_points:
            result = evaluate(x)
            if result is None:
                break

        if eval_count >= self.budget or time.time() > deadline:
            return np.array(evaluated_points)

        # Build array of evaluated points
        F_arr = np.array(evaluated_f)
        X_arr = np.array(evaluated_points)

        # Phase 2: Identify promising regions and refine with Nelder-Mead
        def is_pareto_dominated(f, F_set):
            """Check if f is dominated by any point in F_set."""
            for f_other in F_set:
                if np.all(f_other <= f) and np.any(f_other < f):
                    return True
            return False

        # Find non-dominated points from current evaluations
        pareto_points = []
        for i, f in enumerate(F_arr):
            if not is_pareto_dominated(f, np.delete(F_arr, i, axis=0)):
                pareto_points.append(i)

        # Select refinement targets: best per objective + some pareto points
        refinement_targets = []

        # Best point for each objective
        for obj_idx in range(self.n_obj):
            best_idx = np.argmin(F_arr[:, obj_idx])
            if best_idx not in refinement_targets:
                refinement_targets.append(best_idx)

        # Add some non-dominated points for diversity
        for idx in pareto_points:
            if idx not in refinement_targets and len(refinement_targets) < 12:
                refinement_targets.append(idx)

        # Shuffle to add diversity
        np.random.seed(42)
        np.random.shuffle(refinement_targets)

        # Phase 3: Local refinement using Nelder-Mead with weighted sums
        # Generate weight vectors for decomposition - more weights for better coverage
        rng = np.random.default_rng(123)
        n_weights = 12
        weights = []
        for _ in range(n_weights):
            w = rng.dirichlet(np.ones(self.n_obj) * 0.6)
            weights.append(w)
        # Add extreme weights for each objective
        for i in range(self.n_obj):
            w = np.ones(self.n_obj) * 0.05
            w[i] = 1.0 - 0.05 * (self.n_obj - 1)
            weights.append(w)
        # Add balanced weights
        weights.append(np.ones(self.n_obj) / self.n_obj)

        weights = np.array(weights)

        # For each target and weight, run local optimization
        for target_idx in refinement_targets:
            if eval_count >= self.budget or time.time() > deadline:
                break

            x0 = X_arr[target_idx].copy()

            for w in weights:
                if eval_count >= self.budget or time.time() > deadline:
                    break

                # Define scalarized objective
                def make_scalar_obj(w_vec):
                    def scalar_obj(x):
                        if time.time() > deadline:
                            return 1e10
                        x = np.clip(x, lo, hi)
                        f = np.array(problem(x), dtype=float)
                        evaluated_points.append(x.copy())
                        evaluated_f.append(f.copy())
                        eval_count += 1
                        return float(f @ w_vec)
                    return scalar_obj

                scalar_obj = make_scalar_obj(w)

                # Run Nelder-Mead with tight tolerances for precision
                try:
                    result = minimize(scalar_obj, x0, method='Nelder-Mead',
                                      options={'maxiter': 25, 'xatol': 1e-5, 'fatol': 1e-5,
                                               'initial_simplex_size': 0.15})
                except Exception:
                    pass

        # Phase 4: Final polish pass - refine the best points found so far
        if eval_count < self.budget and time.time() < deadline:
            F_final = np.array(evaluated_f)
            # Find non-dominated points in final set
            final_pareto = []
            for i, f in enumerate(F_final):
                if not is_pareto_dominated(f, np.delete(F_final, i, axis=0)):
                    final_pareto.append(i)

            # Select top points for final polish
            polish_targets = final_pareto[:min(6, len(final_pareto))]

            # Use a few more weight vectors for final refinement
            polish_weights = []
            for _ in range(4):
                w = rng.dirichlet(np.ones(self.n_obj) * 0.5)
                polish_weights.append(w)
            polish_weights.append(np.ones(self.n_obj) / self.n_obj)
            polish_weights = np.array(polish_weights)

            for target_idx in polish_targets:
                if eval_count >= self.budget or time.time() > deadline:
                    break

                x0 = evaluated_points[target_idx].copy()

                for w in polish_weights:
                    if eval_count >= self.budget or time.time() > deadline:
                        break

                    def make_polish_obj(w_vec):
                        def polish_obj(x):
                            if time.time() > deadline:
                                return 1e10
                            x = np.clip(x, lo, hi)
                            f = np.array(problem(x), dtype=float)
                            evaluated_points.append(x.copy())
                            evaluated_f.append(f.copy())
                            eval_count += 1
                            return float(f @ w_vec)
                        return polish_obj

                    polish_obj = make_polish_obj(w)

                    try:
                        result = minimize(polish_obj, x0, method='Nelder-Mead',
                                          options={'maxiter': 30, 'xatol': 1e-6, 'fatol': 1e-6,
                                                   'initial_simplex_size': 0.05})
                    except Exception:
                        pass

        return np.array(evaluated_points)
# EVOLVE-BLOCK-END
