"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with diverse initializations for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 325.0  # safety margin before 330s kill

        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_point(x):
            nonlocal call_count
            if time.time() > deadline:
                return None
            if call_count >= budget:
                return None
            call_count += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        # Generate diverse initial points
        rng = np.random.default_rng(42)
        candidates = []

        # 1. Origin
        candidates.append(np.zeros(dim))

        # 2. Center of box
        candidates.append(np.zeros(dim))  # same as origin for [-5,5]

        # 3. Uniform random points
        for _ in range(8):
            candidates.append(rng.uniform(-5, 5, dim))

        # 4. Grid-like points at specific coordinates
        for coord in [-4, -2, 0, 2, 4]:
            pt = np.zeros(dim)
            pt[0] = coord
            candidates.append(pt)

        # 5. Diagonal points
        for scale in [1, 2, 3, 4]:
            candidates.append(np.full(dim, scale))
            candidates.append(np.full(dim, -scale))

        # 6. Axis-aligned points
        for i in range(dim):
            for val in [-4, -2, 2, 4]:
                pt = np.zeros(dim)
                pt[i] = val
                candidates.append(pt)

        # 7. More uniform random points
        for _ in range(10):
            candidates.append(rng.uniform(-5, 5, dim))

        # 8. Perturbations of zero
        for _ in range(5):
            candidates.append(rng.normal(0, 1.5, dim))

        # Evaluate all candidates first
        candidate_evals = []
        for c in candidates:
            c_clipped = np.clip(c, -5, 5)
            fc = eval_point(c_clipped)
            if fc is None:
                break
            candidate_evals.append((fc, c_clipped))
            if fc < best_f:
                best_f, best_x = fc, c_clipped.copy()

        # Sort by fitness
        candidate_evals.sort(key=lambda x: x[0])

        # Take top candidates for L-BFGS-B refinement
        # Use remaining budget for local optimization
        # Each L-BFGS-B call uses some evaluations; we need to be careful
        # Let's try up to 10 local optimizations, each with a small budget

        top_k = min(10, len(candidate_evals))
        used_evals = call_count
        remaining_budget = budget - call_count

        # For each top candidate, run L-BFGS-B with limited maxiter
        # L-BFGS-B with numerical gradients uses ~dim+1 evals per iteration
        # With dim=5, that's ~6 evals per iteration
        # Let's allow about 5 iterations per local opt (30 evals)
        # But we need to track calls carefully

        for k in range(top_k):
            if call_count >= budget - 10:  # reserve some budget
                break
            if time.time() > deadline:
                break

            fc, xc = candidate_evals[k]
            if fc >= best_f:
                continue

            # Wrap function to track calls and stop early
            local_calls = [0]
            max_local_calls = min(30, budget - call_count - 5)

            def local_func(x, _calls=local_calls):
                if _calls[0] >= max_local_calls:
                    return best_f  # return best known to stop
                _calls[0] += 1
                x_clipped = np.clip(x, -5, 5)
                fx = eval_point(x_clipped)
                if fx is None:
                    return best_f
                if fx < best_f:
                    best_f = fx
                    best_x = x_clipped.copy()
                return fx

            # Use L-BFGS-B
            try:
                res = minimize(local_func, xc, method='L-BFGS-B',
                               options={'maxiter': 5, 'maxls': 10})
            except Exception:
                pass

            # Check if we improved
            if res.success or res.fun < fc:
                x_new = np.clip(res.x, -5, 5)
                f_new = eval_point(x_new)
                if f_new is not None and f_new < best_f:
                    best_f, best_x = f_new, x_new.copy()

        # Final polish: fine local search around best
        if best_x is not None and call_count < budget - 3 and time.time() < deadline:
            sigma_polish = 0.1
            for _ in range(5):
                y = np.clip(best_x + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = eval_point(y)
                if fy is None:
                    break
                if fy < best_f:
                    best_f, best_x = fy, y.copy()
                    sigma_polish *= 0.7
                else:
                    sigma_polish *= 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
