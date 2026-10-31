import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.optimize import basinhopping

N = 38
DEADLINE = 145.0
START_TIME = time.time()

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

def relax(positions):
    """Relax using L-BFGS-B with high precision."""
    def energy_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        return lj_energy(pos)
        
    x0 = positions.flatten()
    try:
        result = minimize(energy_func, x0, method='L-BFGS-B', 
                          options={'maxiter': 1000, 'gtol': 1e-10, 'maxls': 100})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def perturb(positions, rng, step=0.1):
    """Perturb positions for basin hopping."""
    return positions + rng.normal(0, step, positions.shape)

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # 1. Generate the canonical FCC 38-atom cluster
    best_positions = make_fcc_38()
    best_energy = lj_energy_vectorized(best_positions)
    
    # 2. Relax the initial candidate
    if time.time() - START_TIME < DEADLINE - 10:
        relaxed = relax(best_positions)
        if check_validity(relaxed):
            e = lj_energy_vectorized(relaxed)
            if e < best_energy:
                best_energy = e
                best_positions = relaxed
    
    # 3. Basin Hopping to ensure we are in the global basin
    # The FCC fragment is very likely the global minimum, but symmetry breaking
    # in the "38 closest points" selection might lead to a local min.
    # We run a few basin hopping steps to verify/improve.
    
    rng = np.random.default_rng(42)
    
    # Define step function for basinhopping
    def step(pos_flat):
        pos = pos_flat.reshape(N, 3)
        return perturb(pos, rng, step=0.1).flatten()
        
    def objective(pos_flat):
        pos = pos_flat.reshape(N, 3)
        return lj_energy(pos)
        
    x0 = best_positions.flatten()
    
    try:
        # Run basin hopping for a short duration
        # maxfun=100 is roughly 100 energy evaluations, which is fast.
        # We use a custom time check in the callback? No, basinhopping doesn't support time callbacks easily.
        # We will run a fixed number of iterations if time permits.
        
        # Let's do 5 iterations of basin hopping
        for _ in range(5):
            if time.time() - START_TIME > DEADLINE - 5:
                break
                
            result = basinhopping(objective, x0, stepfunc=step, 
                                  miniter=1, maxiter=10, 
                                  niter=1, 
                                  disp=False, 
                                  takestep=step, 
                                  accept=basinhopping.accept,
                                  T=10.0)
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_positions = result.x.reshape(N, 3)
                x0 = result.x
            else:
                # If no improvement, try a smaller perturbation or break
                break
                
    except Exception:
        pass

    # 4. Final high-precision polish
    if time.time() - START_TIME < DEADLINE - 2:
        final_relaxed = relax(best_positions)
        if check_validity(final_relaxed):
            e = lj_energy_vectorized(final_relaxed)
            if e < best_energy:
                best_energy = e
                best_positions = final_relaxed

    return best_positions
