"""Pack 100 equal spheres in a unit cube; maximize the common radius."""
import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 100
CUBE_HALF = 0.5


def _compute_radius(centers):
    """Compute the maximum common radius for a set of centers."""
    if len(centers) < 2:
        return CUBE_HALF - np.max(np.abs(centers))
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    # Wall distances
    wall = CUBE_HALF - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall)
    return min(min_pair, min_wall)


def _build_bcc_init():
    """Build a BCC-inspired initial configuration for N spheres."""
    # Use a BCC lattice in a cube. For N=100, we need a 5x5x4 grid of BCC cells
    # BCC has 2 points per cell: corner + body center
    # Let's use a 5x5x4 arrangement of BCC unit cells
    # That gives 5*5*4*2 = 200 points, take first 100
    # Actually let's think differently.
    
    # A simple approach: use a 5x5x4 grid where each cell has a center point
    # and offset points. But let's just use a dense grid.
    
    # Better: create a BCC lattice
    # BCC points: (i,j,k) and (i+0.5, j+0.5, k+0.5) for i,j,k in grid
    # In a cube of side 1, if we have n divisions per axis, spacing = 1/n
    # BCC: points at (i/n, j/n, k/n) and ((i+0.5)/n, (j+0.5)/n, (k+0.5)/n)
    # for i,j,k = 0..n-1, but we need to keep them in [0,1]
    
    # For N=100, let's try n=5: 5^3 = 125 corner points, plus 125 body centers = 250 total
    # Take 100 of them.
    
    n = 5
    points = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # Corner point (shifted to be inside cube)
                x = (i + 0.5) / n
                y = (j + 0.5) / n
                z = (k + 0.5) / n
                points.append([x, y, z])
                # Body center
                if i < n - 1 and j < n - 1 and k < n - 1:
                    bx = (i + 1.0) / n
                    by = (j + 1.0) / n
                    bz = (k + 1.0) / n
                    points.append([bx, by, bz])
    
    points = np.array(points)
    # Shift to center at origin: subtract 0.5
    points = points - 0.5
    
    # We have more than 100 points, take the first 100
    # But let's be smarter: take a subset that is well-distributed
    # Actually, let's just take the first 100
    if len(points) >= N:
        return points[:N].copy()
    else:
        # Pad with random if needed (shouldn't happen)
        remaining = N - len(points)
        rand_pts = (np.random.rand(remaining, 3) - 0.5) * 0.9
        return np.vstack([points, rand_pts])


def _compute_score(centers):
    """Compute the radius (score) for given centers."""
    return _compute_radius(centers)


def construct():
    """Construct the optimal packing of 100 equal spheres in a unit cube."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Initialize with BCC-inspired configuration
    centers = _build_bcc_init()
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    # Use scipy's L-BFGS-B to maximize the minimum distance
    # We minimize the negative of the radius
    def objective(x):
        c = x.reshape(N, 3)
        # Keep centers within bounds
        c = np.clip(c, -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
        return -_compute_radius(c)
    
    # Try multiple optimization passes with different initializations
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        if attempts == 1:
            x0 = best_centers.flatten()
        else:
            # Perturb the best solution
            x0 = (best_centers + 0.01 * np.random.randn(N, 3)).flatten()
            x0 = np.clip(x0, -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'maxfun': 50000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            new_centers = result.x.reshape(N, 3)
            new_centers = np.clip(new_centers, -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
            new_score = _compute_score(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
        except Exception:
            pass
        
        # Check if we've reached a good solution
        if best_score > 0.1113:
            break
    
    return best_centers


# Ensure the function is accessible
if __name__ == "__main__":
    centers = construct()
    r = _compute_radius(centers)
    print(f"Radius: {r:.10f}")
