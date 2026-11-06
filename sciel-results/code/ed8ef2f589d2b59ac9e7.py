"""Rewritten program: max-min fairness via smooth log-sum-exp on dense grid."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Cover the unit sphere with 50 equal caps.
    Strategy:
    1. High-quality initial guess via Fibonacci spiral + random perturbation.
    2. Fast vectorized Lloyd's relaxation to get a reasonable tiling.
    3. Smooth max-min optimization: use log-sum-exp over a dense test grid
       to approximate the covering radius with a differentiable surrogate,
       then perform gradient ascent on the minimum distance (equivalently
       gradient descent on the max distance).
    4. Multiple restarts with different seeds and perturbation levels.
    5. Explicit deadline guard for safe return.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin

    def fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def normalize(pts):
        n = np.linalg.norm(pts, axis=1, keepdims=True)
        n = np.where(n < 1e-14, 1.0, n)
        return pts / n

    def lloyd_iteration(points, test_points, step_size=1.0):
        dots = np.clip(test_points @ points.T, -1.0, 1.0)
        nearest = np.argmax(dots, axis=1)
        n_pts = len(points)
        sums = np.zeros((n_pts, 3))
        np.add.at(sums, nearest, test_points)
        counts = np.bincount(nearest, minlength=n_pts)
        counts_safe = np.where(counts == 0, 1, counts)
        centroids = sums / counts_safe[:, np.newaxis]
        centroids = normalize(centroids)
        empty = counts == 0
        if np.any(empty):
            centroids[empty] = points[empty]
        result = (1 - step_size) * points + step_size * centroids
        return normalize(result)

    def compute_covering_radius_precise(points):
        """Exact covering radius via convex hull facet circumradii (in degrees)."""
        try:
            hull = ConvexHull(points)
        except Exception:
            test = fibonacci_sphere(200000)
            dots = np.clip(test @ points.T, -1.0, 1.0)
            min_dot = np.min(dots, axis=1)
            return np.degrees(np.arccos(np.max(min_dot)))

        max_rad = 0.0
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            e1, e2 = v2 - v1, v3 - v1
            A = np.array([[2 * np.dot(e1, e1), 2 * np.dot(e1, e2)],
                          [2 * np.dot(e1, e2), 2 * np.dot(e2, e2)]])
            b = np.array([np.dot(e1, e1), np.dot(e2, e2)])
            try:
                params = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue
            cc = v1 + params[0] * e1 + params[1] * e2
            ncc = np.linalg.norm(cc)
            if ncc < 1e-12:
                continue
            cc = cc / ncc
            d = np.clip(np.dot(cc, v1), -1.0, 1.0)
            ang = np.arccos(d)
            if ang > max_rad:
                max_rad = ang
        return np.degrees(max_rad)

    def separate_close_points(points, min_cos=0.9995):
        """Push apart points that are nearly coincident."""
        pts = points.copy()
        for _ in range(3):
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.dot(pts[i], pts[j])
                    if d > min_cos:
                        diff = pts[i] - pts[j]
                        nd = np.linalg.norm(diff)
                        if nd > 1e-10:
                            diff = diff / nd
                            push = 0.002 * (1.0 - d) / (1.0 - min_cos + 1e-9)
                            pts[i] = pts[i] + push * diff
                            pts[j] = pts[j] - push * diff
                            pts[i] = pts[i] / np.linalg.norm(pts[i])
                            pts[j] = pts[j] / np.linalg.norm(pts[j])
        return pts

    def smooth_maxmin_refine(points, test_grid, num_rounds=200, beta_start=50.0, beta_end=500.0, base_lr=0.02):
        """
        Smooth max-min optimization using log-sum-exp.
        
        We want to minimize max_y arccos(min_i dot(y, p_i)).
        Equivalent to maximizing min_i dot(y, p_i) for the worst y.
        
        Surrogate: f(beta) = (1/beta) * log(sum_y exp(beta * (min_i dot(y, p_i))))
        This is a smooth approximation of max_y min_i dot(y, p_i).
        
        Gradient w.r.t. p_i: for each y, the min_i dot is achieved by some i*.
        With log-sum-exp, the gradient is a weighted average.
        
        Actually, let's think more carefully:
        Let g(y) = min_i dot(y, p_i). We want to maximize min_y g(y) = min_y min_i dot(y, p_i).
        Equivalently, minimize max_y max_i (-dot(y, p_i)).
        
        Let h(y, i) = -dot(y, p_i). We want to minimize max_{y,i} h(y, i).
        Log-sum-exp surrogate: F = (1/beta) log(sum_{y,i} exp(beta * h(y, i)))
        
        Gradient w.r.t. p_j:
        dF/dp_j = (1/beta) * sum_{y,i} exp(beta*h(y,i)) * dh(y,i)/dp_j / sum_{y,i} exp(beta*h(y,i))
        
        dh(y,i)/dp_j = -y * delta(i,j)
        
        So dF/dp_j = -(1/beta) * sum_y exp(beta * h(y,j)) * y / sum_{y,i} exp(beta*h(y,i))
        
        This is a weighted average of -y, where the weight for (y,j) is proportional to exp(beta * (-dot(y, p_j))).
        
        We want to minimize F, so we move p_j in the direction -dF/dp_j = +weighted_avg(y).
        But we need to project to the tangent plane.
        
        Let's implement this.
        """
        pts = points.copy()
        n_test = len(test_grid)
        
        for rnd in range(num_rounds):
            if time.time() > deadline:
                break
            
            # Anneal beta
            t = rnd / max(1, num_rounds - 1)
            beta = beta_start + (beta_end - beta_start) * t
            
            # Compute dots: (n_test, N)
            dots = test_grid @ pts.T  # (n_test, N)
            neg_dots = -dots  # (n_test, N)
            
            # For numerical stability, subtract max per row? No, we need global max for log-sum-exp
            # Actually, for each (y, i), we have h(y,i) = -dot(y, p_i)
            # Let's compute the log-sum-exp properly
            max_neg = np.max(neg_dots)
            shifted = neg_dots - max_neg
            exp_vals = np.exp(beta * shifted)
            log_sum = np.log(np.sum(exp_vals)) + beta * max_neg
            F = log_sum / beta
            
            # Gradient: dF/dp_j = -(1/beta) * sum_y exp(beta * neg_dots[y,j]) * test_grid[y] / sum_{y,i} exp(beta * neg_dots[y,i])
            # The denominator is sum_{y,i} exp(beta * neg_dots[y,i]) = exp(beta * max_neg) * sum(exp_vals)
            denom = np.sum(exp_vals)
            
            # For each j, grad[j] = -(1/beta) * sum_y exp_vals[:, j] * test_grid[y] / denom
            # = -(1/(beta * denom)) * (exp_vals.T @ test_grid)
            grad = -(exp_vals.T @ test_grid) / (beta * denom)
            
            # Project gradient to tangent plane: for each point p_j, the tangent space is {v : dot(v, p_j) = 0}
            # Projected grad: grad_j - dot(grad_j, p_j) * p_j
            for j in range(N):
                radial = np.dot(grad[j], pts[j])
                grad[j] -= radial * pts[j]
            
            # Normalize gradient for stable step size
            gn = np.linalg.norm(grad)
            if gn > 1e-14:
                lr = base_lr * (1.0 - 0.7 * t)  # decay learning rate
                pts = pts + lr * (grad / gn)
                pts = normalize(pts)
        
        return pts

    # Pre-compute Lloyd test grid
    lloyd_test = fibonacci_sphere(100000)
    # Dense test grid for smooth optimization
    smooth_test = fibonacci_sphere(50000)

    best_points = None
    best_score = float('inf')

    rng = np.random.RandomState(42)

    max_restarts = 15
    restart = 0

    while time.time() < deadline and restart < max_restarts:
        restart += 1

        # Initialization
        if restart == 1:
            points = fibonacci_sphere(N)
        elif restart == 2:
            points = fibonacci_sphere(N) + rng.standard_normal((N, 3)) * 0.02
            points = normalize(points)
        elif restart <= 5:
            points = fibonacci_sphere(N) + rng.standard_normal((N, 3)) * (0.03 + 0.02 * (restart - 3))
            points = normalize(points)
        elif best_points is not None and rng.random() < 0.7:
            points = best_points + rng.standard_normal((N, 3)) * 0.03
            points = normalize(points)
        else:
            points = rng.standard_normal((N, 3))
            points = normalize(points)

        # Phase 1: Lloyd's relaxation
        step_size = 0.5
        for it in range(60):
            if time.time() > deadline - 25:
                break
            points = lloyd_iteration(points, lloyd_test, step_size)
            step_size *= 0.95
            if step_size < 0.02:
                step_size = 0.02

        # Phase 2: Separate close points
        points = separate_close_points(points)

        # Phase 3: Smooth max-min refinement
        points = smooth_maxmin_refine(points, smooth_test, num_rounds=150, beta_start=30.0, beta_end=200.0, base_lr=0.015)
        points = separate_close_points(points, min_cos=0.9998)

        # Evaluate
        score = compute_covering_radius_precise(points)

        if score < best_score:
            best_score = score
            best_points = points.copy()

        # Phase 4: Fine-tune if we're getting close to known best
        if best_score < 18.5 and time.time() < deadline - 35:
            points = smooth_maxmin_refine(points, smooth_test, num_rounds=100, beta_start=100.0, beta_end=500.0, base_lr=0.005)
            points = separate_close_points(points, min_cos=0.9999)
            score2 = compute_covering_radius_precise(points)
            if score2 < best_score:
                best_score = score2
                best_points = points.copy()

    # Final normalization
    if best_points is not None:
        best_points = normalize(best_points)
    else:
        best_points = fibonacci_sphere(N)

    return best_points
# EVOLVE-BLOCK-END
