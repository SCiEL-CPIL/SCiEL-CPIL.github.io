"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using adaptive Tchebycheff with boundary-initiated local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        eval_count = 0
        all_X = []
        all_F = []

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), lo, hi)
            f = problem(x)
            f = np.asarray(f, dtype=np.float64)
            eval_count += 1
            all_X.append(x.copy())
            all_F.append(f.copy())
            return f

        # Check remaining time
        def time_ok():
            return (time.time() < deadline) and (eval_count < budget)

        # Non-dominated mask
        def non_dominated_mask(F):
            n = len(F)
            if n == 0:
                return np.array([], dtype=bool)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] = False
                        break
            return nd

        # HV contribution for a single point given a set of points (approximate via sorting)
        def hv_contributions(F, ref):
            n = len(F)
            contribs = np.zeros(n)
            for i in range(n):
                # Hypervolume of point i against ref, considering all other points
                # Simple: compute HV of {i} minus HV of {i} dominated by others
                # For 3D, we can compute exactly
                fi = F[i]
                if np.any(fi >= ref):
                    contribs[i] = 0
                    continue
                # Volume of box from fi to ref
                vol = 1.0
                for d in range(n_obj):
                    if fi[d] < ref[d]:
                        vol *= (ref[d] - fi[d])
                    else:
                        vol = 0
                        break
                # Subtract volume dominated by other points that also dominate fi
                # A point j dominates fi if F[j] <= fi element-wise and not equal
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= fi) and np.any(F[j] < fi):
                        # j dominates i, so i's contribution is 0
                        vol = 0
                        break
                contribs[i] = vol
            return contribs

        # Initialize with random points
        rng = np.random.default_rng(12345)
        n_init = 20
        for _ in range(min(n_init, budget)):
            if not time_ok():
                break
            x = lo + (hi - lo) * rng.random(dim)
            evaluate(x)

        # Sort initial archive
        F_arr = np.array(all_F) if all_F else np.zeros((0, n_obj))
        X_arr = np.array(all_X) if all_X else np.zeros((0, dim))

        # Generate adaptive weight vectors
        def generate_weights(n_w, F_current):
            weights = []
            # Axis weights
            for i in range(n_obj):
                w = np.zeros(n_obj)
                w[i] = 1.0
                weights.append(w)
            # If we have current front, generate weights that emphasize gaps
            if len(F_current) >= 2:
                # Use points on the front to guide weight selection
                nd_mask = non_dominated_mask(F_current)
                nd_pts = F_current[nd_mask]
                if len(nd_pts) > 0:
                    # For each pair of adjacent non-dominated points, create a weight
                    # that emphasizes the direction between them
                    for i in range(len(nd_pts)):
                        for j in range(i + 1, len(nd_pts)):
                            if len(weights) >= n_w:
                                break
                            # Direction from point i to j in objective space
                            diff = nd_pts[j] - nd_pts[i]
                            if np.all(diff == 0):
                                continue
                            # Weight inversely proportional to distance in each dim
                            # where one point is better
                            w = np.ones(n_obj)
                            for d in range(n_obj):
                                if diff[d] > 0:
                                    w[d] = 1.0 / (1.0 + diff[d])
                                elif diff[d] < 0:
                                    w[d] = 1.0 / (1.0 + abs(diff[d]))
                            w = w / w.sum()
                            weights.append(w)
                        if len(weights) >= n_w:
                            break
            # Fill with random weights
            while len(weights) < n_w:
                w = rng.random(n_obj)
                w = w / w.sum()
                weights.append(w)
            return weights[:n_w]

        # Tchebycheff local search with coordinate descent fallback
        def tchebycheff_local_search(x0, w, zmin, zmax, max_iter=30):
            """Local search using Tchebycheff scalarization with coordinate descent."""
            if not time_ok():
                return None

            def tcheb_obj(x):
                if eval_count >= budget:
                    return 1e10, None
                f = evaluate(x)
                if f is None:
                    return 1e10, None
                zrange = zmax - zmin
                zrange[zrange < 1e-10] = 1.0
                norm = (f - zmin) / zrange
                return np.max(w * norm), f

            x_cur = np.clip(x0.copy(), lo, hi)
            _, f_cur = tcheb_obj(x_cur)
            if f_cur is None:
                return None
            best_val = np.max(w * ((f_cur - zmin) / np.maximum(zmax - zmin, 1e-10)))
            best_x = x_cur.copy()
            best_f = f_cur.copy()

            step_size = 0.15 * (hi - lo)

            for _ in range(max_iter):
                if not time_ok():
                    break
                improved = False
                for d in range(dim):
                    # Try positive step
                    x_try = best_x.copy()
                    x_try[d] = min(x_try[d] + step_size, hi[d])
                    val, f_try = tcheb_obj(x_try)
                    if f_try is not None and val < best_val - 1e-12:
                        best_val = val
                        best_x = x_try.copy()
                        best_f = f_try.copy()
                        improved = True
                    # Try negative step
                    x_try = best_x.copy()
                    x_try[d] = max(x_try[d] - step_size, lo[d])
                    val, f_try = tcheb_obj(x_try)
                    if f_try is not None and val < best_val - 1e-12:
                        best_val = val
                        best_x = x_try.copy()
                        best_f = f_try.copy()
                        improved = True
                if not improved:
                    # Reduce step size
                    step_size *= 0.5
                    if step_size < 1e-6 * (hi - lo).max():
                        break
            return best_x, best_f

        # Main loop: adaptive Tchebycheff with boundary initialization
        F_arr = np.array(all_F) if all_F else np.zeros((0, n_obj))
        if len(F_arr) > 0:
            zmin = np.min(F_arr, axis=0)
            zmax = np.max(F_arr, axis=0)
        else:
            zmin = np.zeros(n_obj)
            zmax = np.ones(n_obj)

        weights = generate_weights(15, F_arr)
        weight_idx = 0
        stagnation_count = 0
        best_hv_proxy = 0

        while time_ok():
            w = weights[weight_idx % len(weights)]
            weight_idx += 1

            # Find best starting point from archive for this weight
            if len(F_arr) > 0:
                zrange = zmax - zmin
                zrange[zrange < 1e-10] = 1.0
                norm = (F_arr - zmin) / zrange
                dists = np.max(w * norm, axis=1)
                # Pick from a few best points for diversity
                n_candidates = min(5, len(F_arr))
                top_idx = np.argsort(dists)[:n_candidates]
                start_idx = top_idx[rng.randint(len(top_idx))]
                x0 = all_X[start_idx].copy()
            else:
                x0 = (lo + hi) / 2.0

            # With some probability, initialize at a boundary
            if rng.random() < 0.25:
                # Pick a random boundary
                d = rng.randint(dim)
                if rng.random() < 0.5:
                    x0 = x0.copy()
                    x0[d] = lo[d]
                else:
                    x0 = x0.copy()
                    x0[d] = hi[d]
            elif rng.random() < 0.2:
                # Random start
                x0 = lo + (hi - lo) * rng.random(dim)

            # Run local search
            result = tchebycheff_local_search(x0, w, zmin, zmax, max_iter=25)

            if result is not None:
                x_res, f_res = result
                # Check if this is a new non-dominated point
                if len(F_arr) > 0:
                    is_dom = False
                    for af in F_arr:
                        if np.all(af <= f_res) and np.any(af < f_res):
                            is_dom = True
                            break
                    if not is_dom:
                        # Remove dominated points
                        new_F = []
                        new_X = []
                        for i in range(len(F_arr)):
                            if not (np.all(f_res <= F_arr[i]) and np.any(f_res < F_arr[i])):
                                new_F.append(F_arr[i])
                                new_X.append(all_X[i])
                        new_F.append(f_res)
                        new_X.append(x_res)
                        F_arr = np.array(new_F) if new_F else np.zeros((0, n_obj))
                        all_F = [f.copy() for f in F_arr]
                        all_X = new_X
                        zmin = np.min(F_arr, axis=0) if len(F_arr) > 0 else np.zeros(n_obj)
                        zmax = np.max(F_arr, axis=0) if len(F_arr) > 0 else np.ones(n_obj)
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                else:
                    F_arr = f_res.reshape(1, -1)
                    all_F = [f_res.copy()]
                    all_X = [x_res.copy()]
                    zmin = f_res.copy()
                    zmax = f_res.copy()
                    stagnation_count = 0

            # Regenerate weights periodically
            if weight_idx % 20 == 0 and len(F_arr) > 2:
                weights = generate_weights(15, F_arr)
                weight_idx = 0

            # If stagnating, do a few random restarts
            if stagnation_count >= 10:
                for _ in range(3):
                    if not time_ok():
                        break
                    x_rand = lo + (hi - lo) * rng.random(dim)
                    evaluate(x_rand)
                    if eval_count < budget and len(all_F) > 0:
                        F_arr = np.array(all_F)
                        zmin = np.min(F_arr, axis=0)
                        zmax = np.max(F_arr, axis=0)
                stagnation_count = 0
                weights = generate_weights(15, F_arr)
                weight_idx = 0

        return all_X

# EVOLVE-BLOCK-END
