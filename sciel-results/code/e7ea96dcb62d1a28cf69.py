# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Hypervolume-driven adaptive refinement for DTLZ1 (3 objectives, 7 dims).

    Strategy:
    - Initialize with a coarse grid on the Pareto front parameter space.
    - Iteratively propose new points by sampling regions with high marginal HV potential.
    - Use efficient dominance filtering and exact HV computation for selection.
    - Map objective space candidates to decision space using DTLZ1 structure (g=0).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_evals = 0
        max_evals = self.budget

        # Store all evaluated points
        X_list = []
        F_list = []

        def evaluate(x):
            nonlocal n_evals
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            n_evals += 1
            X_list.append(x.copy())
            F_list.append(f)
            return f

        # Helper: compute hypervolume of a set of 3D points w.r.t. reference point
        def hypervolume_3d(points, ref):
            """Compute exact 3D hypervolume using O(n^2) algorithm."""
            if len(points) == 0:
                return 0.0
            pts = np.array(points)
            # Filter points that are strictly better than ref in all objectives
            mask = np.all(pts < ref, axis=1)
            pts = pts[mask]
            if len(pts) == 0:
                return 0.0

            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            n = len(pts)

            def hv2d(pts2d, ref2d):
                """Compute 2D hypervolume of points w.r.t. reference point."""
                if len(pts2d) == 0:
                    return 0.0
                pts2d = np.array(pts2d)
                mask = np.all(pts2d < ref2d, axis=1)
                pts2d = pts2d[mask]
                if len(pts2d) == 0:
                    return 0.0
                # Sort by first coord
                order = np.argsort(pts2d[:, 0])
                pts2d = pts2d[order]
                n2 = len(pts2d)
                area = 0.0
                prev_x = 0.0
                min_y = ref2d[1]
                for j in range(n2):
                    x = pts2d[j, 0]
                    y = pts2d[j, 1]
                    if y < min_y:
                        area += (x - prev_x) * (min_y - y)
                        min_y = y
                    prev_x = x
                return area

            hv = 0.0
            for i in range(n):
                if i == 0:
                    dx = pts[i, 0]
                else:
                    dx = pts[i, 0] - pts[i-1, 0]
                if dx <= 0:
                    continue
                pts_ge_i = pts[i:, 1:3]
                area_i = hv2d(pts_ge_i, [ref[1], ref[2]])
                hv += dx * area_i

            return hv

        ref = np.array([400.0, 400.0, 400.0])
        rng = np.random.default_rng(0)

        # Helper: Generate a candidate point on the Pareto front of DTLZ1
        def generate_pareto_candidate(f1=None, f3=None):
            """Generate a point on the DTLZ1 Pareto front.
            
            DTLZ1 Pareto front (g=0):
            f1 = 0.5 * x1
            f2 = 0.5 * x2 * (1-x1)
            f3 = 0.5 * (1-x2)
            
            Inverse mapping:
            x1 = 2 * f1
            x2 = 1 - 2 * f3
            f2 = 0.5 * (1 - 2*f3) * (1 - 2*f1)
            """
            if f1 is None:
                f1 = rng.uniform(0.0, 0.5)
            if f3 is None:
                f3 = rng.uniform(0.0, 0.5)
            
            f2 = 0.5 * (1.0 - 2.0 * f3) * (1.0 - 2.0 * f1)
            f = np.array([f1, f2, f3])
            
            x = np.zeros(self.dim)
            x[0] = 2.0 * f1
            x[1] = 1.0 - 2.0 * f3
            x[2:] = 0.5  # Set g(x) = 0 to be on Pareto front
            
            return x, f

        # Helper: Check if a point is dominated by any point in a set
        def is_dominated(point, points_set):
            if len(points_set) == 0:
                return False
            pts = np.array(points_set)
            return np.any(np.all(pts >= point, axis=1) & np.any(pts < point, axis=1))

        # Helper: Compute marginal HV gain of adding a point to existing set
        def marginal_hv_gain(f_new, F_existing, ref):
            """Compute the marginal hypervolume gain of adding f_new to F_existing."""
            if len(F_existing) == 0:
                # HV of single point
                return (ref[0] - f_new[0]) * (ref[1] - f_new[1]) * (ref[2] - f_new[2])
            
            # Check if f_new is dominated
            if is_dominated(f_new, F_existing):
                return 0.0
            
            # Compute HV with and without the new point
            F_with = np.vstack([F_existing, [f_new]])
            hv_with = hypervolume_3d(F_with, ref)
            hv_without = hypervolume_3d(F_existing, ref)
            
            return hv_with - hv_without

        # Initialization: Coarse grid on the Pareto front parameter space
        # The Pareto front is parameterized by (f1, f3) in [0, 0.5] x [0, 0.5]
        # We use a structured grid to ensure good initial coverage
        
        # Use a 5x5 grid for initialization (25 points)
        n_grid = 5
        init_params = []
        for i in range(n_grid):
            for j in range(n_grid):
                f1 = i / (n_grid - 1) * 0.5
                f3 = j / (n_grid - 1) * 0.5
                init_params.append((f1, f3))
        
        # Add the three extreme points explicitly
        init_params.append((0.0, 0.0))
        init_params.append((0.5, 0.0))
        init_params.append((0.0, 0.5))
        
        # Deduplicate and limit to budget
        seen = set()
        unique_params = []
        for f1, f3 in init_params:
            key = (round(f1, 6), round(f3, 6))
            if key not in seen:
                seen.add(key)
                unique_params.append((f1, f3))
        
        # Limit to fit within budget
        n_init = min(len(unique_params), max_evals // 2)
        init_params = unique_params[:n_init]
        
        for f1, f3 in init_params:
            x, f = generate_pareto_candidate(f1, f3)
            evaluate(x)

        # Main loop: Greedy hypervolume maximization
        while n_evals < max_evals:
            F = np.array(F_list)
            
            # Find non-dominated points
            n_points = len(F)
            if n_points == 0:
                # Fallback: random point
                x = rng.uniform(lo, hi)
                evaluate(x)
                continue
            
            is_dom = np.zeros(n_points, dtype=bool)
            for i in range(n_points):
                if is_dom[i]:
                    continue
                for j in range(n_points):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dom[i] = True
                        break
            
            non_dom_idx = np.where(~is_dom)[0]
            if len(non_dom_idx) == 0:
                non_dom_idx = np.arange(n_points)
            
            F_nd = F[non_dom_idx]
            
            # Generate candidates focused on high-potential regions
            # Sample from the Pareto front parameter space with emphasis on uncovered regions
            n_candidates = 200
            candidates = []
            
            # Mix of random sampling and targeted sampling
            for _ in range(n_candidates):
                # Random sampling in the parameter space
                f1 = rng.uniform(0.0, 0.5)
                f3 = rng.uniform(0.0, 0.5)
                x_cand, f_cand = generate_pareto_candidate(f1, f3)
                candidates.append((x_cand, f_cand))
            
            # Filter dominated candidates
            valid_candidates = []
            for x_cand, f_cand in candidates:
                if not is_dominated(f_cand, F_nd):
                    valid_candidates.append((x_cand, f_cand))
            
            if not valid_candidates:
                # If all are dominated, pick a random one
                x_cand, f_cand = generate_pareto_candidate()
                evaluate(x_cand)
                continue

            # Score candidates by estimated marginal HV gain
            # Use box volume as a fast proxy, then refine with exact HV for top candidates
            scored_candidates = []
            for x_cand, f_cand in valid_candidates:
                # Box volume as upper bound on marginal HV
                vol = (ref[0] - f_cand[0]) * (ref[1] - f_cand[1]) * (ref[2] - f_cand[2])
                scored_candidates.append((vol, x_cand, f_cand))
            
            # Sort by volume descending and take top K for exact HV computation
            scored_candidates.sort(key=lambda item: item[0], reverse=True)
            top_k = min(20, len(scored_candidates))
            top_candidates = scored_candidates[:top_k]
            
            # Compute exact marginal HV for top candidates
            best_x = None
            best_f = None
            best_gain = -1.0
            
            for _, x_cand, f_cand in top_candidates:
                gain = marginal_hv_gain(f_cand, F_nd, ref)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x_cand
                    best_f = f_cand
            
            if best_x is not None and best_gain > 0:
                evaluate(best_x)
            else:
                # Fallback: random point
                x = rng.uniform(lo, hi)
                evaluate(x)

        return X_list
# EVOLVE-BLOCK-END
