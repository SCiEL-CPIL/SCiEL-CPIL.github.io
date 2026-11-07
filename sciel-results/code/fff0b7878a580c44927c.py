"""Distance-based space-filling optimization for RE37 multi-objective task."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Distance-based space-filling in objective space:
    1) Deterministic space-filling initial design (Halton + special points).
    2) Iteratively select next point by maximizing minimum distance in objective space
       to all previously evaluated points, ensuring uniform Pareto front coverage.
    3) No surrogate models or weighted-sum scalarization.
    4) Adaptive candidate generation around promising regions.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 310.0
        self.eval_count = 0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj

        rng = np.random.default_rng(0)

        def eval_real(x):
            if self.eval_count >= self.budget:
                raise RuntimeError("budget exceeded")
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != dim:
                x = np.resize(x, dim)
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float).reshape(-1)
            self.eval_count += 1
            return f

        # --- Halton sequence for initial space-filling design ---
        def halton_sequence(n, dim):
            def halton_one(i, base):
                f = 1.0
                r = 0.0
                while i > 0:
                    f /= base
                    r += f * (i % base)
                    i //= base
                return r
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2
                for i in range(1, n + 1):
                    points[i-1, d] = halton_one(i, base)
            return points

        # --- Initial space-filling design ---
        n_halton = 25
        halton_pts = halton_sequence(n_halton, dim)

        specials = [lo, hi, (lo+hi)/2.0,
                    np.array([0,0,1,1]), np.array([0,1,0,1]),
                    np.array([0,1,1,0]), np.array([1,0,0,1]),
                    np.array([1,0,1,0]), np.array([1,1,0,0]),
                    np.array([0.5,0.5,0.5,0.5]),
                    np.array([0.25,0.25,0.25,0.25]),
                    np.array([0.75,0.75,0.75,0.75])]
        specials = np.array([np.clip(s, lo, hi) for s in specials], dtype=float)

        grid = np.vstack([specials, halton_pts])
        rng.shuffle(grid)

        # Track evaluated points
        model_X = []
        model_Y = []

        # Evaluate initial design
        for x in grid:
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            try:
                f = eval_real(x)
                model_X.append(np.asarray(x, dtype=float).copy())
                model_Y.append(np.asarray(f, dtype=float).copy())
            except Exception:
                break

        if len(model_X) == 0:
            return None

        # --- Distance-based point selection in objective space ---
        def min_distance_to_set(f_new, F_set):
            """Compute minimum Euclidean distance from f_new to all points in F_set."""
            if len(F_set) == 0:
                return np.inf
            dists = np.sqrt(np.sum((F_set - f_new)**2, axis=1))
            return np.min(dists)

        def generate_candidates_around(X_set, Y_set, n_cand=50):
            """Generate candidate points around existing evaluated points."""
            candidates = []
            n_pts = len(X_set)
            if n_pts == 0:
                return np.array([])

            # Sample around each existing point with adaptive radius
            for i in range(n_pts):
                if time.time() > deadline:
                    break
                # Radius based on local density
                radius = 0.15 + 0.1 * (i / max(n_pts, 1))
                for _ in range(3):
                    xc = X_set[i] + rng.normal(0, radius, size=dim)
                    xc = np.clip(xc, lo, hi)
                    candidates.append(xc)

            # Also add some random points
            for _ in range(10):
                xc = rng.uniform(lo, hi, size=dim)
                candidates.append(xc)

            return np.array(candidates)

        def select_best_candidate(candidates, Y_set, n_select=5):
            """Select candidates that maximize minimum distance in objective space.
            Since we don't know objective values yet, we use a heuristic:
            prefer candidates far from existing X points AND in under-explored regions.
            """
            if len(candidates) == 0:
                return []

            # Compute distance from each candidate to all existing X points
            X_set = np.array(model_X)
            if len(X_set) > 0:
                # For each candidate, find min distance to any existing point
                cand_dists = np.zeros(len(candidates))
                for i, c in enumerate(candidates):
                    dists = np.sqrt(np.sum((X_set - c)**2, axis=1))
                    cand_dists[i] = np.min(dists)
            else:
                cand_dists = np.ones(len(candidates))

            # Select top candidates by distance (most distant from existing)
            top_idx = np.argsort(cand_dists)[::-1][:n_select]
            return candidates[top_idx]

        # --- Main loop: iterative distance-based selection ---
        max_iterations = self.budget - len(model_X)
        for iteration in range(max_iterations):
            if self.eval_count >= self.budget or time.time() > deadline:
                break

            Y_set = np.array(model_Y)

            # Generate candidates
            candidates = generate_candidates_around(model_X, model_Y, n_cand=60)

            if len(candidates) == 0:
                break

            # Select top candidates
            top_cands = select_best_candidate(candidates, Y_set, n_select=5)

            # Evaluate selected candidates and keep those that improve coverage
            best_x = None
            best_f = None
            best_min_dist = -1

            for xc in top_cands:
                if self.eval_count >= self.budget or time.time() > deadline:
                    break
                try:
                    fc = eval_real(xc)
                except Exception:
                    continue

                # Check if this point improves coverage in objective space
                min_dist = min_distance_to_set(fc, Y_set)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_x = xc.copy()
                    best_f = fc.copy()

            # Add the best candidate (if any)
            if best_x is not None:
                model_X.append(best_x)
                model_Y.append(best_f)
                Y_set = np.array(model_Y)
            else:
                # If no improvement, try a random point
                xc = rng.uniform(lo, hi, size=dim)
                try:
                    fc = eval_real(xc)
                    model_X.append(xc.copy())
                    model_Y.append(fc.copy())
                except Exception:
                    break

        # --- Final exploitation: local refinement around best Pareto points ---
        def local_refine(x0, max_evals=3):
            """Refine around a point using small perturbations."""
            best_x = np.asarray(x0, dtype=float).copy()
            try:
                best_f = eval_real(best_x)
            except Exception:
                return None, None

            steps = 0.03
            for _ in range(max_evals):
                if self.eval_count >= self.budget or time.time() > deadline:
                    break
                improved = False
                for d in range(dim):
                    if self.eval_count >= self.budget or time.time() > deadline:
                        break
                    for sign in [-1, 1]:
                        xc = best_x.copy()
                        xc[d] += sign * steps
                        xc = np.clip(xc, lo, hi)
                        try:
                            fc = eval_real(xc)
                        except Exception:
                            continue
                        # Check if this improves coverage
                        Y_arr = np.array(model_Y)
                        min_dist_old = min_distance_to_set(best_f, Y_arr)
                        min_dist_new = min_distance_to_set(fc, Y_arr)
                        if min_dist_new > min_dist_old:
                            best_x = xc
                            best_f = fc
                            model_X.append(best_x.copy())
                            model_Y.append(best_f.copy())
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    steps *= 0.5
                    if steps < 0.01:
                        break
            return best_x, best_f

        # Refine around a few promising points
        if len(model_Y) > 0:
            Y_arr = np.array(model_Y)
            # Find points with good coverage (high min distance to others)
            n_pts = len(model_Y)
            coverage_scores = np.zeros(n_pts)
            for i in range(n_pts):
                others = np.delete(Y_arr, i, axis=0)
                if len(others) > 0:
                    coverage_scores[i] = min_distance_to_set(Y_arr[i], others)
                else:
                    coverage_scores[i] = 0

            # Refine top 3 points by coverage
            top_idx = np.argsort(coverage_scores)[::-1][:3]
            for idx in top_idx:
                if self.eval_count >= self.budget or time.time() > deadline:
                    break
                x0 = model_X[idx]
                local_refine(x0, max_evals=2)

        return None
# EVOLVE-BLOCK-END
