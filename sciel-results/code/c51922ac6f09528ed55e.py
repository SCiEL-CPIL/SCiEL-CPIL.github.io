"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import time

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # 5s safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check centers are inside the square [-0.5, 0.5]^2
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    
    # Distance to walls
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    tree = cKDTree(centers)
    # Query each point's nearest neighbor (excluding self)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    
    r = min(min_wall, min_pair / 2.0)
    return r

def _generate_hex_grid():
    """Generate a hexagonal lattice packing as initial configuration."""
    best_centers = None
    best_r = 0.0
    
    # Try different row counts
    for n_rows in range(10, 25):
        n_cols = int(np.ceil(N / n_rows))
        
        # Hexagonal spacing: horizontal = s, vertical = s * sqrt(3)/2
        # With alternating offset of s/2
        # We need to fit in [-0.5, 0.5]^2
        
        # Let's compute the required spacing
        # Width: (n_cols - 1) * s + 2*r <= 1  =>  s + 2*r <= 1/(n_cols-1) if n_cols > 1
        # Actually, for hex: the width is (n_cols - 1) * s + 2*r
        # The height is (n_rows - 1) * s * sqrt(3)/2 + 2*r
        
        # We want to maximize r subject to:
        # (n_cols - 1) * s + 2*r <= 1
        # (n_rows - 1) * s * sqrt(3)/2 + 2*r <= 1
        # s >= 2*r (non-overlapping in hex: nearest neighbor distance = s)
        
        # From hex geometry, nearest neighbor distance = s
        # So we need s >= 2*r
        
        # Let's solve: maximize r
        # s = 2*r (tight packing)
        # (n_cols - 1) * 2*r + 2*r <= 1  =>  2*r * n_cols <= 1  =>  r <= 1/(2*n_cols)
        # (n_rows - 1) * 2*r * sqrt(3)/2 + 2*r <= 1  =>  r * ((n_rows-1)*sqrt(3) + 2) <= 1  =>  r <= 1/((n_rows-1)*sqrt(3) + 2)
        
        r1 = 1.0 / (2.0 * n_cols)
        r2 = 1.0 / ((n_rows - 1) * np.sqrt(3) + 2)
        r = min(r1, r2)
        s = 2 * r
        
        # Build the grid
        centers = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(centers) >= N:
                    break
                x = -0.5 + r + j * s + (i % 2) * s / 2
                y = -0.5 + r + i * s * np.sqrt(3) / 2
                centers.append([x, y])
            if len(centers) >= N:
                break
        
        centers = np.array(centers[:N])
        
        # Check if all are in bounds
        if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
            # Adjust: try to center the grid
            # Recompute with centering
            xs = centers[:, 0]
            ys = centers[:, 1]
            cx = (xs.max() + xs.min()) / 2
            cy = (ys.max() + ys.min()) / 2
            centers[:, 0] -= cx
            centers[:, 1] -= cy
            
            # Now check if it fits
            if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 - 1e-9):
                continue
        
        r_actual = _compute_radius(centers)
        if r_actual > best_r:
            best_r = r_actual
            best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers

def _pde_membrane_relax(centers, max_iter=20000, h=0.02, stiffness=1.0):
    """
    PDE-driven relaxation using a finite-difference Laplacian on a membrane.
    This models the packing as a soap film with pinholes, where the equilibrium
    shape minimizes surface energy, naturally enforcing uniform spacing.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    # Create a finite-difference grid
    grid_size = int(np.ceil(1.0 / h))
    x_grid = np.linspace(-0.5, 0.5, grid_size)
    y_grid = np.linspace(-0.5, 0.5, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    # Map particle positions to grid indices
    def pos_to_idx(x, y):
        ix = int((x + 0.5) / h)
        iy = int((y + 0.5) / h)
        ix = np.clip(ix, 0, grid_size - 1)
        iy = np.clip(iy, 0, grid_size - 1)
        return ix, iy
    
    # Initialize membrane height field
    U = np.zeros((grid_size, grid_size))
    
    for it in range(max_iter):
        if _time_left() < 10.0:
            break
        
        # Map particles to grid
        particle_idxs = []
        for i in range(N):
            ix, iy = pos_to_idx(centers[i, 0], centers[i, 1])
            particle_idxs.append((ix, iy))
        
        # Set pinholes (particles) as fixed boundaries
        U_new = U.copy()
        for ix, iy in particle_idxs:
            U_new[ix, iy] = 0.0  # Pin to zero
        
        # Compute Laplacian: U_xx + U_yy
        # Using central differences
        U_lap = np.zeros_like(U)
        U_lap[1:-1, 1:-1] = (
            U[2:, 1:-1] + U[:-2, 1:-1] + U[1:-1, 2:] + U[1:-1, :-2] - 4 * U[1:-1, 1:-1]
        ) / (h * h)
        
        # Boundary conditions: zero at edges (fixed walls)
        U_lap[0, :] = 0
        U_lap[-1, :] = 0
        U_lap[:, 0] = 0
        U_lap[:, -1] = 0
        
        # Relax: U += dt * Laplacian(U)
        dt = 0.25 * h * h  # Stability condition for explicit scheme
        U = U_new + dt * U_lap
        
        # Extract forces from membrane curvature
        # Force on particle i is proportional to local curvature
        forces = np.zeros((N, 2))
        for i in range(N):
            ix, iy = particle_idxs[i]
            # Curvature in x and y directions
            if 0 < ix < grid_size - 1:
                fx = (U[ix + 1, iy] - 2 * U[ix, iy] + U[ix - 1, iy]) / (h * h)
            else:
                fx = 0
            if 0 < iy < grid_size - 1:
                fy = (U[ix, iy + 1] - 2 * U[ix, iy] + U[ix, iy - 1]) / (h * h)
            else:
                fy = 0
            
            # Add repulsive force from nearby particles (short-range)
            for j in range(N):
                if i == j:
                    continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d_sq = dx * dx + dy * dy
                if d_sq < (0.15) ** 2:  # Only consider nearby particles
                    d = np.sqrt(d_sq)
                    if d > 1e-9:
                        # Repulsive force: k / d^2
                        f_rep = stiffness * 0.01 / (d_sq + 1e-6)
                        forces[i, 0] += f_rep * dx / d
                        forces[i, 1] += f_rep * dy / d
            
            # Wall forces
            for k in range(2):
                pos = centers[i, k]
                if pos > 0.4:
                    forces[i, k] -= 10.0 * (pos - 0.4)
                elif pos < -0.4:
                    forces[i, k] += 10.0 * (-0.4 - pos)
            
            # Combine membrane and particle forces
            forces[i, 0] += 0.5 * fx
            forces[i, 1] += 0.5 * fy
        
        # Update positions
        centers += 0.01 * forces
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        # Check improvement
        r_trial = _compute_radius(centers)
        if r_trial > best_r:
            best_r = r_trial
            best_centers = centers.copy()
    
    return best_centers, best_r

def _coordinate_ascent_refine(centers, max_iter=5000):
    """
    High-precision coordinate ascent refinement.
    For each coordinate, find the optimal value that maximizes the minimum radius.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_r = _compute_radius(centers)
    
    tree = cKDTree(centers)
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        improved = False
        
        for i in range(N):
            for k in range(2):
                # Current value
                x0 = centers[i, k]
                
                # Find nearest neighbor in this coordinate
                # We want to maximize min(distance to walls, half distance to nearest neighbor)
                
                # Distance to walls
                wall_left = x0 + 0.5
                wall_right = 0.5 - x0
                
                # Find nearest neighbor in this coordinate
                # We can use the tree to find nearest neighbors
                dists, idxs = tree.query(centers[i], k=5)
                min_nn_dist = np.inf
                for d, j in zip(dists, idxs):
                    if j == i:
                        continue
                    # Distance in this coordinate
                    dd = abs(centers[i, k] - centers[j, k])
                    if dd < min_nn_dist:
                        min_nn_dist = dd
                
                # The optimal x is the midpoint between the closest constraints
                # Constraints: x >= -0.5, x <= 0.5, x >= center_j - d/2, x <= center_j + d/2
                # For simplicity, use a local search
                
                # Try a few candidate positions
                candidates = [x0]
                if min_nn_dist < np.inf:
                    candidates.append(x0 + min_nn_dist / 4)
                    candidates.append(x0 - min_nn_dist / 4)
                    candidates.append(x0 + min_nn_dist / 2)
                    candidates.append(x0 - min_nn_dist / 2)
                
                # Add wall candidates
                candidates.append(np.clip(x0, -0.5 + 1e-6, 0.5 - 1e-6))
                
                best_x = x0
                best_local_r = _compute_radius(centers)
                
                for x_cand in candidates:
                    x_cand = np.clip(x_cand, -0.5, 0.5)
                    centers[i, k] = x_cand
                    r_cand = _compute_radius(centers)
                    if r_cand > best_local_r:
                        best_local_r = r_cand
                        best_x = x_cand
                
                centers[i, k] = best_x
                if best_local_r > best_r:
                    best_r = best_local_r
                    best_centers = centers.copy()
                    improved = True
        
        if not improved and it > 100:
            break
        
        # Rebuild tree periodically
        if it % 50 == 0:
            tree = cKDTree(centers)
    
    return best_centers, best_r

def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing the minimum radius."""
    # 1. Start with a good hexagonal grid
    centers = _generate_hex_grid()
    r = _compute_radius(centers)
    
    # 2. PDE-driven membrane relaxation
    if _time_left() > 50.0:
        centers, r = _pde_membrane_relax(centers, max_iter=15000)
    
    # 3. Coordinate ascent refinement
    if _time_left() > 10.0:
        centers, r = _coordinate_ascent_refine(centers, max_iter=5000)
    
    # Final verification and clipping
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers
# EVOLVE-BLOCK-END
