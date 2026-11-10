import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    t0 = time.time()
    DEADLINE = t0 + 110.0  # Safety margin

    def elapsed():
        return time.time() - t0

    def score(points):
        """
        Calculate the maximum radius for the given centers.
        points: (N, 2) array.
        """
        if points.shape[0] < 2:
            return 0.0
        
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        
        # Set diagonal to infinity to ignore self-distance
        np.fill_diagonal(dist, np.inf)
        
        min_dist = np.min(dist)
        
        # Calculate maximum distance from origin
        norms = np.linalg.norm(points, axis=1)
        max_norm = np.max(norms)
        
        # Radius is limited by both pair distances and boundary
        r_pair = 0.5 * min_dist
        r_bound = 1.0 - max_norm
        
        return min(r_pair, r_bound)

    def gen_hex_seed():
        """
        Generate a hexagonal lattice seed.
        """
        # Estimate initial spacing
        # r ~ sqrt(0.9 / N)
        r_est = np.sqrt(0.9 / N)
        spacing = 2.0 * r_est
        
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        
        # If we have more than N points, select the ones closest to the center
        if len(pts) > N:
            radii = np.linalg.norm(pts, axis=1)
            idx = np.argsort(radii)
            pts = pts[idx[:N]]
        elif len(pts) < N:
            # Shrink spacing and retry
            while len(pts) < N:
                spacing *= 0.95
                k = int(1.0 / spacing) + 3
                pts = []
                for j in range(-k, k + 1):
                    for i in range(-k, k + 1):
                        x = (i + 0.5 * (j % 2)) * spacing
                        y = j * spacing * np.sqrt(3.0) / 2.0
                        if x*x + y*y <= 1.0:
                            pts.append([x, y])
                pts = np.array(pts)
                if len(pts) > N:
                    radii = np.linalg.norm(pts, axis=1)
                    idx = np.argsort(radii)
                    pts = pts[idx[:N]]
                elif len(pts) == N:
                    break
        
        return pts

    def gen_spiral_seed():
        """
        Generate a Vogel spiral seed.
        """
        golden = (1.0 + np.sqrt(5.0)) / 2.0
        ang = np.arange(N) * golden * 2.0 * np.pi
        r = np.sqrt(np.arange(N) / float(N)) * 0.999
        return np.column_stack((r * np.cos(ang), r * np.sin(ang)))

    def objective_and_grad(x_flat):
        """
        Objective function for L-BFGS-B.
        Maximizes the minimum distance (pair or boundary).
        Returns negative of the min distance and its gradient.
        """
        x = x_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist, np.inf)
        
        # Boundary distances
        radii = np.linalg.norm(x, axis=1)
        bound_dist = 2 * (1 - radii) # Distance to boundary is (1-r), but we want to maximize min(0.5*d_pair, 1-r_bound)
        # Wait, the score is min(0.5 * min_pair, min_bound).
        # Let's maximize D = min(min_pair, 2*min_bound).
        # Then r = D/2.
        
        min_pair_dist = np.min(dist)
        min_bound_dist = np.min(bound_dist)
        
        min_val = min(min_pair_dist, min_bound_dist)
        
        # Gradient
        grad_flat = np.zeros_like(x_flat)
        x_grad = np.zeros_like(x)
        
        if min_pair_dist < min_bound_dist:
            # Find the pair that is closest
            idx = np.unravel_index(np.argmin(dist), dist.shape)
            i, j = idx
            d = dist[i, j]
            if d < 1e-10:
                # Degenerate, add small noise to break symmetry
                return -min_val, grad_flat
            v = (x[i] - x[j]) / d
            # We want to increase distance between i and j
            # Gradient of d is (x[i]-x[j])/d for x[i] and -(x[i]-x[j])/d for x[j]
            # We are minimizing -d, so gradient is -v for x[i] and v for x[j]
            x_grad[i] -= v
            x_grad[j] += v
        else:
            # Find the point closest to boundary
            i = np.argmin(bound_dist)
            r = radii[i]
            if r < 1e-10:
                # At center, gradient undefined, push out
                x_grad[i] = np.array([1.0, 0.0]) 
            else:
                # bound_dist = 2 - 2r. Gradient wrt x is -2 * (x/r)
                # We are minimizing -bound_dist = 2r - 2. Gradient is 2 * (x/r)
                # Wait. Objective is -min_val.
                # If min_val = bound_dist, objective = -bound_dist.
                # grad(bound_dist) = -2 * x/r.
                # grad(-bound_dist) = 2 * x/r.
                x_grad[i] = 2 * x[i] / r
        
        grad_flat = x_grad.flatten()
        return -min_val, grad_flat

    def run_optimization(initial_points, max_iter=500):
        """
        Run L-BFGS-B optimization on a set of initial points.
        """
        x_flat = initial_points.flatten()
        
        def wrapper(x):
            val, grad = objective_and_grad(x)
            return val, grad
            
        try:
            res = minimize(wrapper, x_flat, method='L-BFGS-B', jac=True, 
                           options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
            return res.x.reshape(-1, 2)
        except Exception:
            return initial_points

    # Initial Seeds
    best_pts = gen_hex_seed()
    best_score = score(best_pts)
    
    seeds = [
        best_pts,
        gen_spiral_seed()
    ]
    
    # Add a few random seeds if time permits
    if elapsed() < DEADLINE - 30:
        rng = np.random.default_rng(42)
        r = np.sqrt(rng.random(N))
        theta = rng.random(N) * 2 * np.pi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        seeds.append(np.column_stack((x, y)))

    # Optimize each seed
    for i, seed_pts in enumerate(seeds):
        if elapsed() > DEADLINE - 10:
            break
            
        # Time budget per optimization: 30 seconds
        # We don't have a direct timeout in minimize, so we rely on maxiter.
        # L-BFGS-B on 600D is fast. 500 iterations should be well under 10s.
        opt_pts = run_optimization(seed_pts, max_iter=500)
        
        # Clip points to unit circle to ensure validity
        norms = np.linalg.norm(opt_pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            opt_pts[mask] = opt_pts[mask] / norms[mask, None]
        
        opt_score = score(opt_pts)
        
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts

    # Final Polish
    # The previous step already does L-BFGS-B.
    # We can do a few more iterations on the best result found.
    if elapsed() < DEADLINE - 5:
        final_pts = run_optimization(best_pts, max_iter=100)
        norms = np.linalg.norm(final_pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            final_pts[mask] = final_pts[mask] / norms[mask, None]
        
        final_score = score(final_pts)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts

    # Final validity check
    if not np.all(np.isfinite(best_pts)):
        best_pts = gen_hex_seed()
        
    rr = np.linalg.norm(best_pts, axis=1)
    m = rr > 1.0
    if np.any(m):
        best_pts[m] = best_pts[m] / rr[m, None]

    return best_pts
