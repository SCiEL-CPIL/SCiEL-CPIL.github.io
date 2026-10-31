import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

_START_TIME = time.time()

def lj_energy(positions):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                return 1e20
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy

def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r = np.sqrt(np.dot(rij, rij))
            if r < min_dist:
                return False
    return True

def make_fcc_38():
    """
    Generate the 38 closest points to the origin in an FCC lattice.
    FCC points: (i/sqrt(2), j/sqrt(2), k/sqrt(2)) where i+j+k is even.
    """
    scale = 1.0 / math.sqrt(2.0)
    max_coord = 4
    points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * scale, j * scale, k * scale
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2, i, j, k))
                    
    # Sort by distance
    points.sort(key=lambda p: p[3])
    
    # Select 38
    selected = [p[:3] for p in points[:38]]
    return np.array(selected, dtype=float)

def relax(positions, maxiter=2000):
    """Relax using L-BFGS-B with high precision."""
    def energy_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        return lj_energy(pos)
        
    x0 = positions.flatten()
    try:
        result = minimize(energy_func, x0, method='L-BFGS-B', 
                          options={'maxiter': maxiter, 'gtol': 1e-12, 'maxls': 100})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # 1. Generate the canonical FCC 38-atom cluster
    best_positions = make_fcc_38()
    best_energy = lj_energy_vectorized(best_positions)
    
    # 2. Relax the initial candidate
    if _time_left() > 15:
        relaxed = relax(best_positions)
        if check_validity(relaxed):
            e = lj_energy_vectorized(relaxed)
            if e < best_energy:
                best_energy = e
                best_positions = relaxed
    
    # 3. Try multiple perturbations and relaxations to escape local minima
    rng = np.random.default_rng(42)
    
    # Define step function for basin hopping
    def step(pos_flat, step_size):
        pos = pos_flat.reshape(N, 3)
        return (pos + rng.normal(0, step_size, pos.shape)).flatten()
        
    def objective(pos_flat):
        pos = pos_flat.reshape(N, 3)
        return lj_energy(pos)
    
    x0 = best_positions.flatten()
    
    # Try different step sizes for basin hopping
    step_sizes = [0.1, 0.2, 0.3, 0.5]
    
    for step_size in step_sizes:
        if _time_left() < 10:
            break
            
        # Run a few iterations of manual basin hopping
        current_x = x0.copy()
        current_energy = best_energy
        
        for _ in range(3):
            if _time_left() < 5:
                break
                
            # Perturb
            new_x = step(current_x, step_size)
            
            # Relax
            new_pos = new_x.reshape(N, 3)
            relaxed_pos = relax(new_pos, maxiter=1000)
            
            if not check_validity(relaxed_pos):
                continue
                
            new_energy = lj_energy_vectorized(relaxed_pos)
            
            # Accept if better
            if new_energy < best_energy:
                best_energy = new_energy
                best_positions = relaxed_pos
                x0 = relaxed_pos.flatten()
                current_x = x0.copy()
    
    # 4. Final high-precision polish
    if _time_left() > 5:
        final_relaxed = relax(best_positions, maxiter=5000)
        if check_validity(final_relaxed):
            e = lj_energy_vectorized(final_relaxed)
            if e < best_energy:
                best_energy = e
                best_positions = final_relaxed

    return best_positions
