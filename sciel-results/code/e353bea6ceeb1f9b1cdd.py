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
        if points.shape[0] != N:
            return 0.0
        if not np.all(np.isfinite(points)):
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
        Generate a hexagonal lattice seed, ensuring exactly N points.
        """
        # Start with a reasonable spacing
        # Area of unit circle is pi. Area per circle is pi*r^2.
        # N * pi * r^2 <= pi => r <= 1/sqrt(N)
        # Spacing s = 2r. s <= 2/sqrt(N).
        # For N=300, s <= 2/17.32 ~ 0.115.
        # Let's start with s = 0.12 and shrink if needed.
        
        best_pts = None
        # Try a range of spacings
        for spacing in [0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]:
            k = int(1.0 / spacing) + 3
            pts = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    # Keep points well inside to allow optimization room
                    if x*x + y*y <= (1.0 - spacing/2.0)**2:
                        pts.append([x, y])
            
            pts = np.array(pts)
            if len(pts) >= N:
                radii = np.linalg.norm(pts, axis=1)
                idx = np.argsort(radii)
                best_pts = pts[idx[:N]]
                break
            elif len(pts) == N:
                best_pts = pts
                break
        
        if best_pts is None:
            # Fallback: random
            rng = np.random.default_rng(42)
            r = np.sqrt(rng.random(N))
            theta = rng.random(N) * 2 * np.pi
            best_pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
            
        return best_pts

    def gen_spiral_seed():
        """
        Generate a Vogel spiral seed.
        """
        golden = (1.0 + np.sqrt(5.0)) / 2.0
        ang = np.arange(N) * golden * 2.0 * np.pi
        r = np.sqrt(np.arange(N) / float(N)) * 0.99
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
        
        # Boundary distances: 2 * (1 - r)
        radii = np.linalg.norm(x, axis=1)
        bound_dist = 2 * (1 - radii)
        
        min_pair_dist = np.min(dist)
        min_bound_dist = np.min(bound_dist)
        
        min_val = min(min_pair_dist, min_bound_dist)
        
        # Gradient
        grad_flat = np.zeros_like(x_flat)
        x_grad = np.zeros_like(x)
        
        # Use a small epsilon to capture "active" constraints
        eps = 1e-4
        
        if min_pair_dist <= min_bound_dist:
            # Active pairs: d_ij < min_val + eps
            # We want to increase the distance.
            # Gradient of d_ij wrt x_i is (x_i - x_j) / d_ij
            # Gradient of d_ij wrt x_j is (x_j - x_i) / d_ij
            # We are minimizing -min_val.
            # If min_val is determined by pair (i,j), we want to increase d_ij.
            # So we add gradient pointing away from each other.
            
            # Find indices of pairs close to min
            # To be efficient, we can just take the argmin, but averaging over a few
            # is more robust. Let's take the top 5 closest pairs.
            flat_dist = dist.flatten()
            # Exclude diagonal
            flat_dist[np.diag_indices(N)] = np.inf
            # Get indices of smallest distances
            # argsort is O(N^2 log N), which is fine for N=300 (90k elements)
            # But we can do better.
            # Let's just use the single closest pair for speed, as L-BFGS-B
            # takes small steps.
            idx = np.unravel_index(np.argmin(flat_dist), dist.shape)
            i, j = idx
            d = dist[i, j]
            if d < 1e-10:
                # Degenerate
                return -min_val, grad_flat
            v = (x[i] - x[j]) / d
            # We want to increase d.
            # grad(d) wrt x_i is v.
            # We are minimizing -d.
            # grad(-d) wrt x_i is -v.
            x_grad[i] -= v
            x_grad[j] += v
        else:
            # Active boundary points
            i = np.argmin(bound_dist)
            r = radii[i]
            if r < 1e-10:
                x_grad[i] = np.array([1.0, 0.0]) 
            else:
                # bound_dist = 2 - 2r.
                # grad(bound_dist) wrt x is -2 * (x/r)
                # We are minimizing -bound_dist = 2r - 2.
                # grad(2r-2) wrt x is 2 * (x/r)
                x_grad[i] = 2 * x[i] / r
        
        grad_flat = x_grad.flatten()
        return -min_val, grad_flat

    def run_optimization(initial_points, max_iter=2000):
        """
        Run L-BFGS-B optimization on a set of initial points.
        """
        x_flat = initial_points.flatten()
        
        def wrapper(x):
            val, grad = objective_and_grad(x)
            return val, grad
            
        try:
            res = minimize(wrapper, x_flat, method='L-BFGS-B', jac=True, 
                           options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
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
    
    # Add random seeds
    rng = np.random.default_rng(42)
    r = np.sqrt(rng.random(N))
    theta = rng.random(N) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    seeds.append(np.column_stack((x, y)))
    
    # Optimize each seed
    for i, seed_pts in enumerate(seeds):
        if elapsed() > DEADLINE - 15:
            break
            
        opt_pts = run_optimization(seed_pts, max_iter=2000)
        
        # Clip points to unit circle to ensure validity
        norms = np.linalg.norm(opt_pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            opt_pts[mask] = opt_pts[mask] / norms[mask, None]
        
        opt_score = score(opt_pts)
        
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts

    # Final Polish on the best result
    if elapsed() < DEADLINE - 5:
        final_pts = run_optimization(best_pts, max_iter=1000)
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
