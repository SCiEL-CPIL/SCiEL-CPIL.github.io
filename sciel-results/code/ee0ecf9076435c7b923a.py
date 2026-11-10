"""Improved circle packing: hexagonal lattice with boundary optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist


def _score(points):
    """Compute the packing radius for given centers in [-0.5, 0.5]^2."""
    # Distance to boundary
    boundary_dists = np.minimum(
        np.minimum(0.5 - points[:, 0], 0.5 + points[:, 0]),
        np.minimum(0.5 - points[:, 1], 0.5 + points[:, 1])
    )
    # Pairwise distances
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = np.min(d)
    return min(boundary_dists.min(), min_pair_dist / 2.0)


def _hex_lattice_initialization():
    """Generate a hexagonal (close-packed) lattice arrangement."""
    # For hexagonal packing, rows are offset
    # Try different numbers of rows/columns
    best_points = None
    best_radius = 0
    
    # Hexagonal packing: row spacing = s * sqrt(3)/2, col spacing = s
    # With alternating rows offset by s/2
    # Try to fit ~200 circles
    
    for n_rows in range(13, 18):
        for n_cols in range(13, 18):
            if n_rows * n_cols < N:
                continue
            if n_rows * n_cols > N + 5:
                continue
            
            # Hexagonal lattice points
            points = []
            for i in range(n_rows):
                for j in range(n_cols):
                    x = j * 1.0 / (n_cols - 1) if n_cols > 1 else 0.5
                    y = i * 1.0 / (n_rows - 1) if n_rows > 1 else 0.5
                    # Offset alternating rows
                    if i % 2 == 1:
                        x = x + 0.5 / (n_cols - 1) if n_cols > 1 else x
                    # Scale to fit in [-0.5, 0.5]
                    x = (x - 0.5) * (n_cols - 1) / (n_cols - 1)  # keep as is
                    points.append([x, y])
            
            points = np.array(points)
            if len(points) < N:
                # Pad with points near center
                while len(points) < N:
                    points = np.vstack([points, np.array([0.0, 0.0])])
            points = points[:N]
            
            r = _score(points)
            if r > best_radius:
                best_radius = r
                best_points = points.copy()
    
    return best_points if best_points is not None else None


def _grid_initialization():
    """Generate a square grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _optimize(points, max_time=100):
    """Optimize circle positions using local search."""
    start_time = time.time()
    best_points = points.copy()
    best_radius = _score(points)
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def negative_score(params):
        pts = params.reshape(-1, 2)
        # Check validity
        if np.any(pts < -0.5) or np.any(pts > 0.5):
            return 1.0  # penalty
        return -_score(pts)
    
    # Multiple restarts with different initializations
    initializations = []
    
    # Grid
    initializations.append(_grid_initialization())
    
    # Hexagonal
    hex_pts = _hex_lattice_initialization()
    if hex_pts is not None:
        initializations.append(hex_pts)
    
    # Random with repulsion
    rng = np.random.default_rng(42)
    for _ in range(3):
        pts = rng.uniform(-0.45, 0.45, size=(N, 2))
        initializations.append(pts)
    
    for init_pts in initializations:
        if time.time() - start_time > max_time - 10:
            break
        
        result = minimize(
            negative_score,
            init_pts.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if time.time() - start_time > max_time - 5:
            break
        
        if result.fun < -best_radius:
            best_radius = -result.fun
            best_points = result.x.reshape(-1, 2)
    
    return best_points, best_radius


def construct():
    """Construct centers of N equal circles in unit square."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Get initializations
    grid_pts = _grid_initialization()
    hex_pts = _hex_lattice_initialization()
    
    best_points = grid_pts.copy()
    best_radius = _score(grid_pts)
    
    if hex_pts is not None:
        hex_radius = _score(hex_pts)
        if hex_radius > best_radius:
            best_radius = hex_radius
            best_points = hex_pts.copy()
    
    # Try optimization if we have time
    if time.time() - start_time < deadline - 20:
        opt_points, opt_radius = _optimize(best_points, max_time=min(120, deadline - time.time() - 10))
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_points = opt_points
    
    # Final validation
    best_points = np.clip(best_points, -0.5, 0.5)
    
    return best_points
# EVOLVE-BLOCK-END
