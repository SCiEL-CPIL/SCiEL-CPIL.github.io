"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

def _lj_energy_and_grad(pos):
    """
    Calculate Lennard-Jones energy and gradient.
    pos: (N, 3) array
    """
    N = pos.shape[0]
    # Calculate all pair distances
    # Use broadcasting to get distance matrix
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Avoid division by zero (though unlikely in valid configs)
    dist_sq = np.maximum(dist_sq, 1e-12)
    
    # LJ potential: 4 * (r^-12 - r^-6)
    # r^-6 = dist_sq^-3
    # r^-12 = dist_sq^-6
    r6 = dist_sq ** (-3)
    r12 = r6 ** 2
    
    # Energy is sum over i < j
    # The matrix dist_sq has 0 on diagonal.
    # We sum all elements and divide by 2 to count each pair once.
    # But we must exclude diagonal.
    energy = 4.0 * (np.sum(r12) - np.sum(r6)) / 2.0
    
    # Gradient
    # dE/dr = 4 * (-12 r^-13 + 6 r^-7)
    # dr/rdx = -dx / r
    # dE/dx = 4 * (-12 r^-13 + 6 r^-7) * (-dx / r) = 4 * (12 r^-14 - 6 r^-8) * dx
    # Note: r = sqrt(dist_sq). r^-14 = dist_sq^-7. r^-8 = dist_sq^-4.
    # Let's derive carefully.
    # E = sum 4 (r^-12 - r^-6)
    # dE/dr = 4 (-12 r^-13 - (-6) r^-7) = 4 (-12 r^-13 + 6 r^-7)
    # r = sqrt(dx^2 + dy^2 + dz^2)
    # dr/dx = dx / r
    # dE/dx = dE/dr * dr/dx = 4 (-12 r^-13 + 6 r^-7) * (dx / r)
    # = 4 (-12 r^-14 + 6 r^-8) * dx
    # = 4 * dx * (6 r^-8 - 12 r^-14)
    
    # Vectorized calculation for gradient
    # We need sum over j for each i.
    # term = 4 * (6 * r^-8 - 12 * r^-14)
    # r^-8 = dist_sq^-4
    # r^-14 = dist_sq^-7
    r8 = dist_sq ** (-4)
    r14 = dist_sq ** (-7)
    
    # Coefficient for dx, dy, dz
    coeff = 4.0 * (6.0 * r8 - 12.0 * r14)
    
    # grad[i] = sum_j coeff[i,j] * (pos[i] - pos[j])
    # diff is pos[i] - pos[j]
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _generate_fcc_start():
    """
    Generate 38 atoms in an FCC lattice configuration closest to the origin.
    """
    # Generate FCC lattice points
    # FCC points in cubic cell of side 1: (0,0,0), (0.5,0.5,0), etc.
    # It is easier to use the integer representation where (i,j,k) is in FCC if i+j+k is even.
    # This lattice has nearest neighbor distance sqrt(2).
    
    # We need 38 atoms. 
    # Shell 0: 1 atom
    # Shell 1: 12 atoms (dist sqrt(2))
    # Shell 2: 6 atoms (dist 2)
    # Shell 3: 24 atoms (dist sqrt(6))
    # Total so far: 1+12+6+24 = 43.
    # We take the 38 closest.
    
    # Max coordinate needed: sqrt(6) ~ 2.45. So range -3 to 3 is safe.
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if (i + j + k) % 2 == 0:
                    points.append((i, j, k))
    
    # Sort by distance squared from origin
    points.sort(key=lambda p: p[0]**2 + p[1]**2 + p[2]**2)
    
    # Take first 38
    selected = points[:N]
    pos = np.array(selected, dtype=float)
    
    # Scale to optimal nearest neighbor distance
    # Nearest neighbor in integer lattice is sqrt(2).
    # Target distance is 2^(1/6).
    # Scale factor s = 2^(1/6) / sqrt(2)
    scale = 2.0 ** (1.0/6.0) / np.sqrt(2.0)
    pos *= scale
    
    return pos

def construct():
    start_time = time.time()
    timeout = 110.0  # Leave 10s margin
    
    # 1. Generate initial FCC-like structure
    pos = _generate_fcc_start()
    
    # 2. Optimize using L-BFGS
    # We use the analytic gradient for speed and precision
    res = minimize(
        _lj_energy_and_grad, 
        pos, 
        method='L-BFGS-B', 
        jac=True, 
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10}
    )
    
    current_pos = res.x
    current_energy = res.fun
    
    # 3. Check time and potentially do a second pass or restart if needed
    # Given the strong prior of FCC, one pass is usually sufficient for LJ38.
    # However, if the symmetry was broken by the 38th atom selection, 
    # L-BFGS should still find the local min which is the global one.
    
    elapsed = time.time() - start_time
    if elapsed < timeout:
        # Optional: Try a small perturbation restart if energy is not "good enough"
        # But for LJ38, the FCC basin is deep. 
        # Let's just ensure we return the best.
        pass
        
    return current_pos
# EVOLVE-BLOCK-END
