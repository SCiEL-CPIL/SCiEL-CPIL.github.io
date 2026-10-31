"""
Strategy: Direct 114-Dimensional Basin Hopping with High-Quality Multi-Start Initialization.
Rationale: The previous rigid-body approach (6D) was too restrictive, preventing the cluster 
from relaxing into its true global minimum geometry (which involves specific shell 
relaxations not captured by simple scaling/shear). This strategy abandons the rigid-body 
constraint and operates in the full 114-dimensional coordinate space. 
However, to avoid the "icosahedral funnel" trap, we initialize the search with a 
high-quality FCC truncated octahedron that has been locally relaxed. 
We then perform a basin hopping search: 
1. Start with the relaxed FCC structure.
2. Apply a random perturbation (kick) to all atoms.
3. Perform a local L-BFGS minimization.
4. Accept the new structure if it has lower energy (or with a small probability if higher, 
   though for this specific problem, simple hill-climbing basin hopping is often sufficient 
   if the start is good).
5. Repeat until the time budget is exhausted.
This combines the robustness of global search with the precision of local relaxation, 
starting from the known global minimum topology to ensure we are in the correct basin.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0
START_TIME = time.time()

def lj_energy_vectorized(positions):
    """Compute LJ energy efficiently."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def lj_force_vectorized(positions):
    """Compute LJ forces efficiently (negative gradient of energy)."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    # F_ij = 24 * (2*r^-12 - r^-6) * r_vec
    # Factor of 2 comes from d/dr (r^-12 - r^-6) = -12r^-13 + 6r^-7
    # Energy = 4 * sum(r^-12 - r^-6)
    # dE/dr = 4 * (-12r^-13 + 6r^-7) = 24 * (r^-7 - 2r^-13)
    # Force on i = -dE/dr_i = - sum_j 24 * (r^-7 - 2r^-13) * (r_i - r_j) / r
    # Let's use the standard form: F = 24 * (2*r^-12 - r^-6) * (r_i - r_j) / r^2 * r_vec ? 
    # Actually, F_ij = 24 * (2/r^12 - 1/r^6) * r_vec
    # Wait, dE/dr = 24/r^7 - 48/r^13. Force = -dE/dr * r_hat = (48/r^13 - 24/r^7) * r_hat
    # r_hat = r_vec / r. So F = (48/r^14 - 24/r^8) * r_vec
    # Let's stick to numerical gradient or standard L-BFGS without analytical gradient to avoid errors.
    # L-BFGS can handle numerical gradients.
    pass

def check_validity(positions, min_dist=0.5):
    """Check minimum distance constraint."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1e10)
    min_r2 = np.min(r2)
    return min_r2 >= (min_dist * min_dist)

def make_fcc_truncated_octahedron_raw():
    """
    Generate the 38-atom FCC truncated octahedron points.
    """
    points = []
    max_coord = 5
    scale = 1.0 / math.sqrt(2.0)
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    r2 = i * i + j * j + k * k
                    if r2 <= 30:
                        points.append((i * scale, j * scale, k * scale, r2))
    
    points.sort(key=lambda x: x[3])
    selected = [p[:3] for p in points[:38]]
    
    coords = np.array(selected, dtype=float)
    coords -= np.mean(coords, axis=0)
    
    return coords

def local_minimize(positions):
    """
    Perform a local L-BFGS minimization.
    Returns the minimized positions and energy.
    """
    def energy_func(pos):
        pos = pos.reshape(N, 3)
        return lj_energy_vectorized(pos)
    
    try:
        result = minimize(
            energy_func,
            positions.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-12}
        )
        if result.success:
            min_pos = result.x.reshape(N, 3)
            min_energy = result.fun
            if check_validity(min_pos):
                return min_pos, min_energy
    except Exception:
        pass
    
    # Fallback to input if minimization fails
    return positions, lj_energy_vectorized(positions)

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # 1. Generate initial FCC structure
    try:
        raw_coords = make_fcc_truncated_octahedron_raw()
    except Exception:
        rng = np.random.default_rng(0)
        raw_coords = rng.uniform(-2, 2, size=(N, 3))
        raw_coords -= np.mean(raw_coords, axis=0)
    
    # 2. Initial local relaxation to get into the basin
    current_pos, current_energy = local_minimize(raw_coords)
    
    # 3. Basin Hopping Loop
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(42)
    
    # Parameters for basin hopping
    kick_scale = 0.1  # Initial kick magnitude
    
    while time.time() - START_TIME < DEADLINE - 5:
        # Apply random perturbation
        # We use a Gaussian kick with adaptive scale
        kick = rng.normal(0, kick_scale, size=(N, 3))
        perturbed_pos = current_pos + kick
        
        # Check validity of perturbed position (optional, L-BFGS can handle it if not too close)
        # But if atoms overlap, energy is huge, L-BFGS might fail.
        if not check_validity(perturbed_pos, min_dist=0.3):
            # If too close, try a smaller kick or reject
            continue
        
        # Local minimization
        new_pos, new_energy = local_minimize(perturbed_pos)
        
        # Acceptance criterion: Metropolis or simple hill climbing
        # For this problem, simple hill climbing (accept if lower) is often effective 
        # if we start in the right basin. However, to escape local minima, 
        # we might need to accept some worse moves. 
        # Given the tight time limit and known global minimum, hill climbing 
        # with occasional larger kicks is a good strategy.
        
        if new_energy < current_energy:
            # Accept
            current_pos = new_pos
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
                
            # Slightly reduce kick scale to refine
            kick_scale *= 0.95
            if kick_scale < 0.01:
                kick_scale = 0.01
        else:
            # Reject (or accept with small prob, but let's stick to hill climbing for stability)
            # Occasionally increase kick scale to explore
            if rng.random() < 0.1:
                kick_scale *= 1.1
                if kick_scale > 0.5:
                    kick_scale = 0.5
            
            # Keep current_pos as is
    
    # Final polish on the best position found
    if time.time() - START_TIME < DEADLINE - 2:
        def energy_func(pos):
            pos = pos.reshape(N, 3)
            return lj_energy_vectorized(pos)
        
        try:
            result = minimize(
                energy_func,
                best_pos.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14}
            )
            polished = result.x.reshape(N, 3)
            if check_validity(polished):
                e_polished = lj_energy_vectorized(polished)
                if e_polished < best_energy:
                    best_pos = polished
                    best_energy = e_polished
        except Exception:
            pass

    # Final validity check
    if not check_validity(best_pos):
        # Fallback to initial raw coords if somehow invalid
        best_pos = raw_coords
        if not check_validity(best_pos):
            rng = np.random.default_rng(123)
            pts = []
            for _ in range(N):
                while True:
                    p = rng.uniform(-2, 2, size=3)
                    if all(np.linalg.norm(p - q) > 0.5 for q in pts):
                        pts.append(p)
                        break
            best_pos = np.array(pts)
    
    return best_pos

# EVOLVE-BLOCK-END
