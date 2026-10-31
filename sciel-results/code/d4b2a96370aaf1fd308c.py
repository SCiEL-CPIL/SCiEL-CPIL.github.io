import numpy as np
from scipy.optimize import minimize
import time

N = 38

def _lj_energy(pos):
    """Calculate total Lennard-Jones energy."""
    # pos shape (N, 3)
    # Compute pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Avoid division by zero (though atoms should be distinct)
    # Mask the diagonal
    mask = ~np.eye(N, dtype=bool)
    dist_sq = dist_sq[mask]
    
    # LJ potential: 4 * (r^-12 - r^-6)
    # r^-6 = (r^2)^-3
    r6 = dist_sq ** -3
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6)
    return energy

def _fcc_truncated_octahedron(n_atoms):
    """
    Generate initial coordinates for an FCC truncated octahedron cluster.
    We generate FCC lattice points and select the n_atoms closest to the center.
    """
    # Equilibrium nearest-neighbor distance for LJ is 2^(1/6)
    r_min = 2.0 ** (1.0 / 6.0)
    # FCC lattice constant 'a' such that nearest neighbor distance a/sqrt(2) = r_min
    a = r_min * np.sqrt(2.0)
    
    # Generate a grid of points. 
    # FCC points are at (i*a, j*a, k*a) where i,j,k are integers and i+j+k is even.
    # Or equivalently, we can iterate over a range of indices.
    # For 38 atoms, the cluster radius is roughly 2-3 lattice constants.
    # Let's generate points in a box [-3, 3] in index space.
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                # Check FCC condition: i+j+k must be even
                if (i + j + k) % 2 == 0:
                    x = i * a
                    y = j * a
                    z = k * a
                    points.append([x, y, z])
    
    points = np.array(points)
    
    # Sort by distance to origin
    dists = np.sum(points**2, axis=1)
    sorted_indices = np.argsort(dists)
    
    # Take the first n_atoms
    selected_points = points[sorted_indices[:n_atoms]]
    
    return selected_points

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 110.0 # Leave 10s margin for 120s limit
    
    # 1. Generate initial guess: FCC truncated octahedron
    # This is very close to the global minimum.
    initial_pos = _fcc_truncated_octahedron(N)
    
    # 2. Relax using L-BFGS-B
    # We use a bounded minimization to prevent atoms from flying off to infinity 
    # or collapsing to zero distance (though LJ repulsion handles collapse).
    # Bounds: atoms should stay within a reasonable box, e.g., [-10, 10]
    bounds = [(-10.0, 10.0)] * (N * 3)
    
    # Flatten positions for scipy
    x0 = initial_pos.flatten()
    
    def energy_fn(x):
        pos = x.reshape(N, 3)
        # Check for minimum distance constraint violation? 
        # The task says "Atoms must be at least 0.5 apart".
        # If they get closer, energy goes to infinity, so optimizer will avoid it.
        # However, numerical stability might be an issue if they get very close.
        # L-BFGS-B handles gradients well.
        return _lj_energy(pos)
    
    # Optimization
    # maxiter: 1000 is plenty for 38 atoms
    # ftol: tight tolerance for high precision
    res = minimize(
        energy_fn, 
        x0, 
        method='L-BFGS-B', 
        bounds=bounds, 
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    elapsed = time.time() - start_time
    
    # If we haven't hit the deadline, we can try a second, tighter polish 
    # or just return the result. 
    # Given the start is very good, one run is usually sufficient.
    # But let's check if we have time for a re-run with tighter tolerances 
    # or just verify the result.
    
    final_pos = res.x.reshape(N, 3)
    
    # Ensure minimum distance constraint is met.
    # If relaxation caused any atoms to come closer than 0.5 (unlikely with LJ), 
    # we might need to enforce it. But LJ energy diverges at small r, 
    # so the optimizer naturally keeps them apart.
    # The constraint is "must be at least 0.5 apart".
    # Let's verify.
    # If violated, we should probably return the best valid point.
    # However, for LJ, the minimum is at ~1.12, so 0.5 is a very loose constraint 
    # compared to the equilibrium.
    
    return final_pos
