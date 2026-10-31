"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit to be safe against 120s timeout

    # 1. Generate FCC lattice points
    # FCC lattice: (i, j, k) where i+j+k is even.
    # We need enough points to get 38 closest to the origin.
    # The 38th point is at r^2=10 (in units of a/2? No, let's use integer coords).
    # If we use integer coords (i,j,k) for FCC (sum even), the nearest neighbor is at (1,1,0) with dist sqrt(2).
    # We want the nearest neighbor distance to be ~1.1225.
    # So we will scale the integer coordinates by a factor.
    
    # Generate a grid of points
    # Max coordinate needed: The 38th point is at r^2=10. sqrt(10) ~ 3.16.
    # So we need points up to index 4 or 5 to be safe.
    limit = 5
    points = []
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            for k in range(-limit, limit + 1):
                if (i + j + k) % 2 == 0:
                    points.append((i, j, k))
    
    points = np.array(points, dtype=float)
    
    # Sort by distance to origin
    dists = np.sum(points**2, axis=1)
    sorted_indices = np.argsort(dists)
    points = points[sorted_indices][:N]
    
    # Scale the points
    # The nearest neighbor in this integer lattice is at distance sqrt(2).
    # We want the nearest neighbor distance to be r_min = 2**(1/6) ~ 1.12246
    # So scale factor = r_min / sqrt(2)
    scale = (2.0 ** (1.0/6.0)) / np.sqrt(2.0)
    positions = points * scale
    
    # 2. Relax the structure
    # Define energy function
    def energy_fn(pos):
        # pos is (N, 3)
        # Calculate pairwise distances
        # Use broadcasting for efficiency
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Avoid division by zero (diagonal)
        np.fill_diagonal(dists_sq, 1.0)
        
        # LJ potential: 4 * (r^-12 - r^-6)
        # r^-6 = (r^2)^-3
        inv_r2 = 1.0 / dists_sq
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2
        
        # Sum over all pairs (i < j)
        # The matrix has 2 * sum(i<j) + N diagonal terms.
        # We set diagonal to 1, so inv_r12 - inv_r6 = 1 - 1 = 0.
        # So sum of matrix is 2 * sum(i<j) potential.
        total_energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
        return total_energy

    # Initial energy
    current_energy = energy_fn(positions)
    best_positions = positions.copy()
    best_energy = current_energy

    # Optimization
    # L-BFGS-B is suitable for this.
    # We don't have explicit bounds, but the atoms shouldn't fly apart.
    # A simple relaxation is enough since we start at the global min geometry.
    
    try:
        result = minimize(
            energy_fn, 
            positions.ravel(), 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'gtol': 1e-10, 'disp': False}
        )
        
        if result.success or result.fun < best_energy:
            new_pos = result.x.reshape(N, 3)
            new_energy = result.fun
            if new_energy < best_energy:
                best_positions = new_pos
                best_energy = new_energy
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        return best_positions

    # Optional: Second pass with tighter tolerance if time permits
    if time.time() < deadline - 5.0:
        try:
            result2 = minimize(
                energy_fn, 
                best_positions.ravel(), 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'gtol': 1e-12, 'disp': False}
            )
            if result2.fun < best_energy:
                best_positions = result2.x.reshape(N, 3)
                best_energy = result2.fun
        except Exception:
            pass

    return best_positions

# EVOLVE-BLOCK-END
