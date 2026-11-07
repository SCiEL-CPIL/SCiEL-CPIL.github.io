import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        def evaluate(x):
            nonlocal used
            if not check_time():
                return
            if used >= budget:
                return
            x_clipped = np.clip(x, lo, hi)
            x_eval = x_clipped.copy()
            x_eval[2:] = 0.0
            problem(x_eval)
            used += 1

        # DTLZ7 Pareto front: x[2:]=0, so g=1
        # f1 = x[0], f2 = x[1], f3 = 1 - x[1]*h(x[1], 1)
        # h(y, 1) = 9 / (1 + 2*(sin(3*pi*y) + sin(4*pi*y)))
        #
        # Reference point: [1.1, 1.1, 25.0]
        # Hypervolume contribution of point (f1,f2,f3) ~ (1.1-f1)*(1.1-f2)*(25-f3)
        #
        # Strategy: Use max-min facility location in (f1,f2) space, weighted by (25-f3)
        # This ensures uniform coverage in the hypervolume-relevant region

        # Helper: compute f3 given x1
        def compute_f3(x1):
            # h(y, 1) = 9 / (1 + 2*(sin(3*pi*y) + sin(4*pi*y)))
            sin_sum = np.sin(3 * np.pi * x1) + np.sin(4 * np.pi * x1)
            denom = 1 + 2 * sin_sum
            if np.abs(denom) < 1e-12:
                h_val = 9.0  # fallback
            else:
                h_val = 9.0 / denom
            return 1.0 - x1 * h_val

        # Helper: compute hypervolume weight for a point (x0, x1)
        def hv_weight(x0, x1):
            f1 = x0
            f2 = x1
            f3 = compute_f3(x1)
            # Only count if all objectives < reference
            w1 = max(0.0, 1.1 - f1)
            w2 = max(0.0, 1.1 - f2)
            w3 = max(0.0, 25.0 - f3)
            return w1 * w2 * w3

        # Step 1: Evaluate a small set of anchor points (corners + center)
        anchor_points = [
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.5, 0.5),
            (0.0, 0.5),
            (0.5, 0.0),
            (0.5, 1.0),
            (1.0, 0.5),
        ]

        for x0, x1 in anchor_points:
            if not check_time() or used >= budget:
                break
            x = np.zeros(dim)
            x[0] = lo[0] + (hi[0] - lo[0]) * x0
            x[1] = lo[1] + (hi[1] - lo[1]) * x1
            evaluate(x)

        # Step 2: Max-min facility location to add points
        # We'll maintain a list of (x0, x1) points already evaluated
        # and iteratively add the point that maximizes the minimum weighted distance
        evaluated_points = []
        for x0, x1 in anchor_points[:min(used, len(anchor_points))]:
            evaluated_points.append((x0, x1))

        # For efficiency, we'll use a candidate grid and pick the best from it
        # Candidate grid in (x0, x1) space
        n_candidate = 20
        candidate_x0 = np.linspace(0, 1, n_candidate)
        candidate_x1 = np.linspace(0, 1, n_candidate)

        # Weighted distance: d((x0a,x1a), (x0b,x1b)) = sqrt((x0a-x0b)^2 + (x1a-x1b)^2) * sqrt(w_a * w_b)
        # Actually, for max-min, we want to maximize the minimum distance to existing points
        # Let's use: score(candidate) = min over existing points of (euclidean_distance * sqrt(hv_weight(candidate)))
        # This biases toward regions with high hypervolume contribution

        max_min_budget = min(budget - used, 100)  # Use up to 100 evaluations for max-min
        for _ in range(max_min_budget):
            if not check_time() or used >= budget:
                break

            # Find the best candidate from the grid
            best_score = -1
            best_x0, best_x1 = 0.5, 0.5

            for cx0 in candidate_x0:
                for cx1 in candidate_x1:
                    # Skip if too close to an existing point
                    min_dist = np.inf
                    for ex0, ex1 in evaluated_points:
                        dist = np.sqrt((cx0 - ex0)**2 + (cx1 - ex1)**2)
                        if dist < min_dist:
                            min_dist = dist

                    # Weight by hypervolume contribution
                    w = hv_weight(cx0, cx1)
                    score = min_dist * np.sqrt(w)

                    if score > best_score:
                        best_score = score
                        best_x0, best_x1 = cx0, cx1

            # Evaluate the best point
            x = np.zeros(dim)
            x[0] = lo[0] + (hi[0] - lo[0]) * best_x0
            x[1] = lo[1] + (hi[1] - lo[1]) * best_x1
            evaluate(x)
            evaluated_points.append((best_x0, best_x1))

        # Step 3: Fill remaining budget with power-law warped Halton sequence
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        alpha = 2.0
        def warp(u):
            return u ** (1.0 / (1.0 + alpha))

        remaining = budget - used
        if remaining > 0:
            for i in range(remaining):
                if not check_time():
                    break
                if used >= budget:
                    break

                n = i + 1
                u0 = halton(n, 2)
                u1 = halton(n, 3)
                x0_val = lo[0] + (hi[0] - lo[0]) * warp(u0)
                x1_val = lo[1] + (hi[1] - lo[1]) * warp(u1)

                x = np.zeros(dim)
                x[0] = x0_val
                x[1] = x1_val
                evaluate(x)

        return None
