"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    # Generate FCC lattice points
    # FCC points are integer triples (x, y, z) such that x + y + z is even.
    # We generate points in a cube large enough to contain 38 points.
    points = []
    # Range [-3, 3] is sufficient for 38 points centered at origin.
    for x in range(-3, 4):
        for y in range(-3, 4):
            for z in range(-3, 4):
                if (x + y + z) % 2 == 0:
                    points.append((x, y, z))
    
    # Sort by distance from origin (squared distance)
    points.sort(key=lambda p: p[0]**2 + p[1]**2 + p[2]**2)
    
    # Take the first 38 points
    selected_points = points[:N]
    
    # Convert to numpy array
    coords = np.array(selected_points, dtype=float)
    
    # Scale to Lennard-Jones units
    # The nearest neighbor distance in the integer FCC grid is sqrt(2).
    # The LJ minimum distance is 2^(1/6).
    # Scale factor = 2^(1/6) / sqrt(2) = 2^(1/6 - 1/2) = 2^(-1/3)
    scale = 2 ** (-1/3)
    coords = coords * scale
    
    # Optional: Perform a quick relaxation to ensure minimum energy
    # Define energy function
    def energy(pos):
        # pos is shape (N, 3)
        # Compute all pair distances
        # diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # dist_sq = np.sum(diff**2, axis=-1)
        # mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        # dist_sq = dist_sq[mask]
        # dist = np.sqrt(dist_sq)
        # inv_r6 = 1.0 / (dist**6)
        # return 4.0 * np.sum(inv_r6**2 - inv_r6)
        
        # More efficient vectorized calculation
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        # Avoid division by zero for diagonal
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        # Only upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        dist_pairs = dist[mask]
        inv_r6 = 1.0 / (dist_pairs**6)
        return 4.0 * np.sum(inv_r6**2 - inv_r6)

    def force(pos):
        # Gradient of energy
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        
        # dE/dr = 4 * (-12 r^-13 + 6 r^-7)
        # dE/dx = dE/dr * dx/dr = dE/dr * x / r
        # dE/dr = -24 r^-13 + 24 r^-7 = 24 (r^-7 - r^-13)
        # term = 24 * (1/r^7 - 1/r^13)
        
        # We need to sum over all j != i
        # F_i = sum_j 24 * (r_ij^-7 - r_ij^-13) * (r_ij_vec / r_ij)
        # Note: r_ij_vec = pos_i - pos_j
        
        inv_r2 = 1.0 / dist_sq
        inv_r6 = inv_r2**3
        # Coefficient for vector (pos_i - pos_j)
        # dE/dx_i = 24 * (r^-13 - r^-7) * (x_i - x_j) / r 
        # Wait, E = 4(r^-12 - r^-6). dE/dr = 4(-12r^-13 + 6r^-7) = 24(r^-7 - r^-13).
        # dE/dx_i = dE/dr * (x_i - x_j)/r = 24(r^-7 - r^-13) * (x_i - x_j) / r
        # = 24 (r^-8 - r^-14) * (x_i - x_j)
        
        # Let's compute the force matrix
        # Force on i due to j: F_ij = 24 * (r^-8 - r^-14) * (pos_i - pos_j)
        # Total force on i: sum_j F_ij
        
        # Vectorized:
        # dist_sq is (N, N)
        # r^-8 = (r^-2)^4
        # r^-14 = (r^-2)^7
        
        inv_r8 = inv_r2**4
        inv_r14 = inv_r2**7
        
        # Coefficient for each pair (i,j)
        # coeff = 24 * (inv_r8 - inv_r14)
        # But we only want j != i. The diagonal is 0 because (pos_i - pos_i) = 0.
        
        coeff = 24.0 * (inv_r8 - inv_r14)
        
        # Force_i = sum_j coeff_ij * (pos_i - pos_j)
        # This can be computed as:
        # F = coeff @ pos - (coeff * pos.T).T ? No.
        # F_i = sum_j coeff_ij * pos_i - sum_j coeff_ij * pos_j
        # F_i = pos_i * sum_j coeff_ij - sum_j (coeff_ij * pos_j)
        
        sum_coeff = np.sum(coeff, axis=1)
        term1 = pos * sum_coeff[:, np.newaxis]
        
        # term2: sum_j (coeff_ij * pos_j)
        # This is coeff.T @ pos
        term2 = coeff.T @ pos
        
        forces = term1 - term2
        
        # Gradient of energy is -Force
        return -forces

    # Relax
    res = minimize(energy, coords.flatten(), jac=force, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
    
    return res.x.reshape(N, 3)

# EVOLVE-BLOCK-END
