import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 8
    num_pairs = n * (n - 1) // 2
    
    def compute_distances(pts):
        """Compute pairwise distances matrix."""
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        return dists

    def log_score(pts):
        """Compute log of the product of all unordered pairwise distances."""
        dists = compute_distances(pts)
        # Avoid log(0) by setting diagonal to 1 and ignoring it
        np.fill_diagonal(dists, 1.0)
        iu = np.triu_indices(n, k=1)
        log_delta = np.sum(np.log(dists[iu]))
        return log_delta

    def diameter(pts):
        dists = compute_distances(pts)
        return np.max(dists)

    def objective(x, penalty_weight=100.0):
        pts = x.reshape(n, 2)
        ls = log_score(pts)
        d = diameter(pts)
        penalty = 0.0
        if d > 2.0:
            penalty = penalty_weight * (d - 2.0) ** 2
        # We want to maximize ls, so minimize -ls + penalty
        return -ls + penalty

    def gradient(x, penalty_weight=100.0):
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(x)
        f0 = objective(x, penalty_weight)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (objective(x_plus, penalty_weight) - f0) / eps
        return grad

    best_score = -np.inf
    best_points = None

    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        # Scale to diameter 2
        d = diameter(pts)
        if d > 1e-6:
            pts = pts * (2.0 / d)
        else:
            return
        # Check distinctness
        dists = compute_distances(pts)
        if np.min(dists) < 1e-5:
            return
        score = log_score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # 1. Regular Octagon
    theta = 2 * np.pi * np.arange(n) / n
    pts_oct = np.column_stack([np.cos(theta), np.sin(theta)])
    update_best(pts_oct)

    # 2. Concentric Squares
    # Square 1: angles 0, pi/2, pi, 3pi/2
    # Square 2: angles pi/4, 3pi/4, 5pi/4, 7pi/4
    def make_concentric_squares(r1, r2):
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        angles2 = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        return np.vstack([pts1, pts2])

    # Try a range of ratios r2/r1
    # The diameter is determined by the max distance.
    # Let's just try a few fixed ratios and scale to diameter 2.
    ratios = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
    for ratio in ratios:
        r1 = 1.0
        r2 = ratio
        pts = make_concentric_squares(r1, r2)
        update_best(pts)

    # 3. Differential Evolution for Global Search
    # Bounds: Points should be within [-2, 2] roughly, since diameter is 2.
    bounds = [(-2.0, 2.0)] * (2 * n)
    
    if time.time() < deadline - 60:
        try:
            # DE is slow, so limit iterations
            de_result = differential_evolution(
                lambda x: objective(x, 100.0),
                bounds,
                maxiter=100,
                popsize=20,
                seed=42,
                tol=1e-4,
                polish=False,
                update_interval=10,
                callback=lambda xk, convergence: time.time() < deadline - 60
            )
            
            if de_result.x is not None:
                update_best(de_result.x.reshape(n, 2))
        except Exception as e:
            pass

    # 4. Local Optimization (L-BFGS-B) from multiple starts
    rng = np.random.RandomState(123)
    
    # Starts: Best found so far, Octagon, a few random, and specific cage structures
    starts = []
    if best_points is not None:
        starts.append(best_points.flatten())
    starts.append(pts_oct.flatten())
    
    # Add some random starts
    for _ in range(5):
        random_pts = rng.uniform(-1.5, 1.5, (n, 2))
        starts.append(random_pts.flatten())
        
    # Add more structured starts: Concentric squares with random radii
    for _ in range(5):
        r1 = rng.uniform(0.5, 1.5)
        r2 = rng.uniform(0.5, 1.5)
        pts = make_concentric_squares(r1, r2)
        starts.append(pts.flatten())

    for x0 in starts:
        if time.time() > deadline - 20:
            break
        
        # Multi-stage penalty
        for pw in [10.0, 100.0, 1000.0, 10000.0]:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(
                    lambda x: objective(x, pw),
                    x0,
                    jac=lambda x: gradient(x, pw),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.success:
                    x0 = res.x
                    update_best(x0.reshape(n, 2))
            except Exception:
                continue

    # Final Polish on the best point found so far
    if time.time() < deadline - 5 and best_points is not None:
        x0 = best_points.flatten()
        try:
            res = minimize(
                lambda x: objective(x, 10000.0),
                x0,
                jac=lambda x: gradient(x, 10000.0),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.success:
                update_best(res.x.reshape(n, 2))
        except Exception:
            pass

    if best_points is None:
        # Fallback to octagon
        best_points = pts_oct

    # Ensure diameter is exactly 2 and points are distinct
    d = diameter(best_points)
    if d > 0:
        best_points = best_points * (2.0 / d)
        
    # Return as list of lists
    return [[float(p[0]), float(p[1])] for p in best_points]
