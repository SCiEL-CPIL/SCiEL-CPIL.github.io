"""Physics-based circle packing: repulsive particles + soft wall, gradient descent."""
N = 50
import numpy as np
import time

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s limit

    # Initialize with a hexagonal lattice pattern (better starting point than random)
    def hex_init():
        pts = []
        # Try a few spacings to get ~N points
        for sp in np.linspace(0.12, 0.25, 40):
            k = int(1.0 / sp) + 2
            rows = [((i + 0.5 * (j % 2)) * sp, j * sp * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            arr = np.array(rows)
            r = np.hypot(arr[:, 0], arr[:, 1])
            mask = r <= 1.0 - sp / 2.0
            pts = arr[mask]
            if len(pts) >= N:
                # Sort by radius, take innermost N
                idx = np.argsort(r[mask])[:N]
                return pts[idx].copy()
        # Fallback: random
        return np.random.randn(N, 2) * 0.5

    def energy_and_grad(pts):
        """Compute energy and gradient.
        Energy = sum of repulsive pair potentials + wall potential.
        Repulsive: for pairs closer than d_min, strong repulsion.
        Wall: for points outside radius (1 - margin), push inward.
        """
        n = len(pts)
        E = 0.0
        grad = np.zeros_like(pts)
        
        # Pairwise repulsion: use a smooth potential
        # V(r) = k / r^p for r < r0, else 0
        # We want to maximize min distance, so minimize overlap
        # Use: for each pair, if dist < target_dist, add penalty
        # Better: use a global repulsive potential that encourages spacing
        
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Repulsive potential: V = k * (1/d - 1/d0)^2 for d < d0
        # This pushes points apart when they're closer than d0
        d0 = 0.26  # target minimum distance (twice the target radius ~0.13)
        k_rep = 100.0
        mask = dist < d0
        if np.any(mask):
            d = dist[mask]
            # V = k_rep * (1/d - 1/d0)^2
            # dV/dd = k_rep * 2 * (1/d - 1/d0) * (-1/d^2)
            # Gradient wrt position: dV/dd * (diff/dist)
            v = k_rep * (1.0 / d - 1.0 / d0) ** 2
            E += np.sum(v)
            
            # Gradient for each pair
            dv_dd = k_rep * 2.0 * (1.0 / d - 1.0 / d0) * (-1.0 / d ** 2)
            # grad_i += dv_dd * (diff_ij / dist_ij) * (-1)  [force on i from j]
            # Actually: dV/dx_i = dV/dd * d(d)/d(x_i) = dv_dd * (x_i - x_j)/d
            # For pair (i,j): grad_i += dv_dd * (pts[i] - pts[j]) / d
            #               grad_j += dv_dd * (pts[j] - pts[i]) / d
            
            # Use matrix operations
            # For each masked pair (i,j), add to grad
            ii, jj = np.where(mask)
            for a, b in zip(ii, jj):
                d_ab = dist[a, b]
                if d_ab < 1e-12:
                    continue
                dv = dv_dd[a]  # scalar for this pair
                # grad_a += dv * (pts[a] - pts[b]) / d_ab
                # grad_b += dv * (pts[b] - pts[a]) / d_ab
                vec = (pts[a] - pts[b]) / d_ab
                grad[a] += dv * vec
                grad[b] += dv * (-vec)
        
        # Wall potential: for points with |p| > r_wall, push inward
        r_wall = 1.0
        k_wall = 500.0
        r_pts = np.sqrt(np.sum(pts ** 2, axis=1))
        outside = r_pts > r_wall
        if np.any(outside):
            r_out = r_pts[outside]
            # V = k_wall * (r - r_wall)^2 for r > r_wall
            v_wall = k_wall * (r_out - r_wall) ** 2
            E += np.sum(v_wall)
            # dV/dr = 2 * k_wall * (r - r_wall)
            # grad = dV/dr * (p / r)
            dv_dr = 2.0 * k_wall * (r_out - r_wall)
            for idx in np.where(outside)[0]:
                r = r_pts[idx]
                if r < 1e-12:
                    continue
                grad[idx] += dv_dr[idx] * pts[idx] / r
        
        return E, grad

    def compute_score(pts):
        """Compute the packing score: min(min_pair_dist/2, min(1-|c_i|))."""
        n = len(pts)
        if n != N:
            return 0.0
        # Check all points are finite
        if not np.all(np.isfinite(pts)):
            return 0.0
        # Check all inside unit circle
        r_pts = np.sqrt(np.sum(pts ** 2, axis=1))
        if np.any(r_pts > 1.0 + 1e-10):
            return 0.0
        # Min distance between pairs
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        min_radius = min_pair_dist / 2.0
        min_wall = np.min(1.0 - r_pts)
        score = min(min_radius, min_wall)
        return max(0.0, score)

    # Start with hex initialization
    pts = hex_init()
    if len(pts) != N:
        # Pad or trim
        if len(pts) < N:
            extra = np.random.randn(N - len(pts), 2) * 0.3
            pts = np.vstack([pts, extra])
        else:
            pts = pts[:N]
    
    best_pts = pts.copy()
    best_score = compute_score(pts)
    
    # Gradient descent with adaptive step size
    lr = 0.001
    n_iter_max = 20000
    
    for it in range(n_iter_max):
        if time.time() > deadline:
            break
        
        E, grad = energy_and_grad(pts)
        
        # Adaptive step: reduce if energy increases
        pts_new = pts - lr * grad
        
        E_new, _ = energy_and_grad(pts_new)
        
        if E_new < E:
            pts = pts_new
            # Occasionally increase learning rate if improving
            if it % 100 == 0:
                lr = min(lr * 1.05, 0.01)
        else:
            # Reduce learning rate
            lr *= 0.7
        
        # Update best
        score = compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
        
        # Convergence check
        if it > 100 and np.max(np.abs(grad)) < 1e-6:
            break
    
    # Final refinement: local optimization to push out wall-constrained points
    # and increase minimum distance
    for refine_iter in range(500):
        if time.time() > deadline:
            break
        
        # Compute current score components
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        r_pts = np.sqrt(np.sum(pts ** 2, axis=1))
        min_wall = np.min(1.0 - r_pts)
        
        current_score = min(min_pair_dist / 2.0, min_wall)
        
        # Try to improve by moving points that are limiting
        # Find the bottleneck: is it pair distance or wall?
        if min_pair_dist / 2.0 < min_wall:
            # Pair distance is limiting: push apart the closest pair
            ii, jj = np.unravel_index(np.argmin(dist), dist.shape)
            if ii != jj:
                d = dist[ii, jj]
                if d < 1e-12:
                    continue
                vec = (pts[ii] - pts[jj]) / d
                step = 0.001
                pts[ii] += vec * step
                pts[jj] -= vec * step
        else:
            # Wall is limiting: push inward the point closest to wall
            idx = np.argmin(1.0 - r_pts)
            r = r_pts[idx]
            if r < 1e-12:
                continue
            vec = pts[idx] / r
            step = 0.001
            pts[idx] -= vec * step
        
        # Check validity
        if not np.all(np.isfinite(pts)):
            pts = best_pts.copy()
            break
        
        score = compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Return the best valid configuration
    final = best_pts
    # Ensure all points are strictly inside
    r_final = np.sqrt(np.sum(final ** 2, axis=1))
    mask_out = r_final >= 1.0
    if np.any(mask_out):
        # Scale inward
        final[mask_out] *= (1.0 - 1e-6) / r_final[mask_out]
    
    return final
