import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    def objective(pts):
        """
        Calculate the squared ratio of max to min pairwise distance.
        """
        # pts: (N, 2)
        # Compute all pairwise squared distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Mask diagonal
        iu = np.triu_indices(N, 1)
        d_sq = dists_sq[iu]
        
        # Check for coincident points
        min_d_sq = np.min(d_sq)
        if min_d_sq < 1e-12:
            return 1e10
            
        max_d_sq = np.max(d_sq)
        
        return max_d_sq / min_d_sq

    def make_hexagonal_subset(n):
        # Generate triangular lattice points and pick n closest to origin
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.5, np.sqrt(3) / 2])
        pts = []
        for i in range(-10, 11):
            for j in range(-10, 11):
                p = i * v1 + j * v2
                pts.append(p)
        pts = np.array(pts)
        pts -= pts.mean(axis=0)
        dists = np.linalg.norm(pts, axis=1)
        order = np.argsort(dists)
        return pts[order[:n]]

    def make_rings(radii_counts):
        # radii_counts: list of tuples (radius, count)
        pts = []
        for radius, count in radii_counts:
            if count > 0:
                angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
                # Add a small random rotation to break symmetry if needed
                angles += np.random.uniform(0, 2*np.pi/count)
                for a in angles:
                    pts.append([radius * np.cos(a), radius * np.sin(a)])
        return np.array(pts[:n], dtype=float) if len(pts) >= n else np.array(pts, dtype=float)

    def make_random_disk(n):
        # Random points in unit disk
        r = np.sqrt(np.random.uniform(0, 1, n))
        theta = np.random.uniform(0, 2 * np.pi, n)
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    # Generate diverse initial candidates
    candidates = []
    
    # 1. Hexagonal lattice subset
    candidates.append(make_hexagonal_subset(N))
    
    # 2. Ring configurations
    candidates.append(make_rings([(0, 1), (1.0, 6), (1.8, 9)]))
    candidates.append(make_rings([(0, 1), (1.0, 6), (1.5, 9)]))
    candidates.append(make_rings([(1.0, 12), (0.5, 4)]))
    candidates.append(make_rings([(1.0, 16)])) # All on boundary
    
    # 3. Random starts
    for _ in range(10):
        candidates.append(make_random_disk(N))

    best_val = np.inf
    best_pts = candidates[0].copy()

    # Local optimization using Nelder-Mead
    # Nelder-Mead is robust for this non-smooth problem
    for i, init_pts in enumerate(candidates):
        if time.time() > deadline:
            break
            
        try:
            result = minimize(
                objective,
                init_pts,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_pts = result.x.copy()
        except Exception:
            pass

    # If time permits, run Differential Evolution on the best found configuration
    # DE is good for escaping local minima but slow.
    if time.time() < deadline - 20:
        # Center the best points
        center = best_pts.mean(axis=0)
        best_pts_centered = best_pts - center
        
        # Determine bounds
        # The scale is arbitrary, but we can bound the coordinates to a reasonable range
        # based on the current best extent
        extents = best_pts_centered.max(axis=0) - best_pts_centered.min(axis=0)
        margin = 1.0
        lower = (best_pts_centered.min(axis=0) - margin).flatten()
        upper = (best_pts_centered.max(axis=0) + margin).flatten()
        
        bounds = list(zip(lower, upper))
        
        try:
            de_result = differential_evolution(
                lambda x: objective(x.reshape(N, D)),
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=True
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_pts = de_result.x.reshape(N, D) + center
        except Exception:
            pass

    # Final check
    # Ensure points are distinct (objective handles this, but just in case)
    # Return the best points found
    return best_pts
