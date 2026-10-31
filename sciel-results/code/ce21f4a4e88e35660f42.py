"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

# Deadline: kill before 145s wall-clock, leave safety margin
DEADLINE = 145.0
START_TIME = time.time()


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


def lj_grad_vectorized(positions):
    """Vectorized LJ gradient computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    # dE/dr = 4 * (-12 * r^-13 + 6 * r^-7)
    # grad_r = (dE/dr) * (r_vec / r) = 4 * (-12 * r^-14 + 6 * r^-8) * r_vec
    # Note: r_vec = diff[i,j] = pos[j] - pos[i]
    # grad_i = - sum_j grad_r_ij
    # grad_j = + sum_i grad_r_ij
    
    # Force on i due to j is F_ij = -dE/dr * r_hat = -dE/dr * (r_vec/r)
    # dE/dr = 4 * (-12 * r^-13 + 6 * r^-7)
    # F_ij = 4 * (12 * r^-14 - 6 * r^-8) * r_vec
    # Total force on i = sum_j F_ij
    # Total force on j = sum_i F_ji = - sum_i F_ij
    
    coeff = 4.0 * (12.0 * inv_r12 - 6.0 * inv_r6) * inv_r2
    # coeff shape (N, N)
    # Force on i from j: coeff[i,j] * diff[i,j]
    # Sum over j for each i
    forces = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return forces


def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    return np.min(r2) >= min_dist * min_dist


def make_fcc_truncated_octahedron():
    """
    Create an FCC truncated octahedron cluster of 38 atoms.
    We generate FCC lattice points and select the 38 closest to the origin.
    """
    scale = 1.0 / math.sqrt(2.0)
    
    points = []
    max_coord = 5
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    r2 = i*i + j*j + k*k
                    if r2 <= 50:
                        points.append((i * scale, j * scale, k * scale, r2))
    
    points.sort(key=lambda x: x[3])
    selected = [p[:3] for p in points[:38]]
    return np.array(selected, dtype=float)


def simulated_annealing(positions, steps=5000, T_start=1.0, T_end=0.01, seed=42):
    """
    Perform Simulated Annealing on the LJ cluster.
    """
    rng = np.random.default_rng(seed)
    current_energy = lj_energy_vectorized(positions)
    best_energy = current_energy
    best_positions = positions.copy()
    
    # Precompute pairs for efficiency? No, vectorized is fast enough for N=38.
    
    for i in range(steps):
        if time.time() - START_TIME > DEADLINE - 15:
            break
            
        # Cool down
        T = T_start * (T_end / T_start) ** (i / steps)
        
        # Random move: perturb one atom
        idx = rng.integers(0, N)
        delta = rng.normal(0, 0.1, size=3)
        
        new_positions = positions.copy()
        new_positions[idx] += delta
        
        # Check validity (simple distance check)
        if not check_validity(new_positions):
            continue
            
        new_energy = lj_energy_vectorized(new_positions)
        delta_E = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta_E / T)
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            positions = new_positions
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = positions.copy()
                
    return best_positions, best_energy


def relax_ljfg(positions, high_precision=False):
    """Relax using L-BFGS if available, otherwise return as-is."""
    try:
        from science_chemistry import relax
        return relax(positions)
    except ImportError:
        def energy_func(pos):
            pos = pos.reshape(N, 3)
            return lj_energy_vectorized(pos)
        
        def grad_func(pos):
            pos = pos.reshape(N, 3)
            return lj_grad_vectorized(pos).flatten()
            
        options = {'maxiter': 100, 'gtol': 1e-6}
        if high_precision:
            options = {'maxiter': 500, 'gtol': 1e-8}
            
        result = minimize(energy_func, positions.flatten(), jac=grad_func, method='L-BFGS-B', 
                         options=options)
        return result.x.reshape(N, 3)
    except Exception:
        return positions


def construct():
    """Main entry point: use Simulated Annealing starting from FCC structure."""
    global START_TIME
    START_TIME = time.time()
    
    # Start with FCC truncated octahedron
    try:
        initial_pos = make_fcc_truncated_octahedron()
    except Exception:
        # Fallback to random if FCC fails
        rng = np.random.default_rng(0)
        initial_pos = rng.uniform(-2, 2, size=(N, 3))
        
    if not check_validity(initial_pos):
        # Ensure validity
        initial_pos = np.array([
            [0,0,0], [1.12,0,0], [-1.12,0,0], [0,1.12,0], [0,-1.12,0],
            [0,0,1.12], [0,0,-1.12], [0.7,0.7,0.7], [-0.7,-0.7,-0.7],
            [0.7,-0.7,-0.7], [-0.7,0.7,-0.7], [-0.7,-0.7,0.7], [0.7,0.7,-0.7]
        ])
        # Add more random points if needed
        while len(initial_pos) < N:
            p = np.array([rng.uniform(-2,2), rng.uniform(-2,2), rng.uniform(-2,2)])
            valid = True
            for q in initial_pos:
                if np.linalg.norm(p-q) < 0.5:
                    valid = False
                    break
            if valid:
                initial_pos = np.vstack([initial_pos, p])
            else:
                p[0] += 0.1
        initial_pos = initial_pos[:N]

    best_positions = initial_pos
    best_energy = lj_energy_vectorized(initial_pos)
    
    # Run Simulated Annealing with multiple restarts if time permits
    # We will run SA for a fixed number of steps, then polish.
    # If time remains, we can do another SA restart from the best found so far.
    
    num_sa_runs = 3
    sa_steps = 20000
    
    for run in range(num_sa_runs):
        if time.time() - START_TIME > DEADLINE - 20:
            break
            
        # Start SA from current best (or initial for first run)
        start_pos = best_positions if run > 0 else initial_pos
        # Add small noise to escape local minima for subsequent runs
        if run > 0:
            rng = np.random.default_rng(run)
            start_pos = start_pos + rng.normal(0, 0.05, size=(N, 3))
            
        sa_pos, sa_energy = simulated_annealing(start_pos, steps=sa_steps, seed=run)
        
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_positions = sa_pos
            
    # Polish the best result with L-BFGS
    if time.time() - START_TIME < DEADLINE - 5:
        try:
            polished = relax_ljfg(best_positions, high_precision=True)
            if check_validity(polished):
                polished_energy = lj_energy_vectorized(polished)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_positions = polished
        except Exception:
            pass
            
    # Final check
    if not check_validity(best_positions):
        # Fallback to a valid structure
        rng = np.random.default_rng(123)
        fallback = rng.uniform(-1, 1, size=(N, 3))
        # Simple rejection sampling to ensure distance
        valid_fallback = []
        attempts = 0
        while len(valid_fallback) < N and attempts < 10000:
            p = rng.uniform(-2, 2, size=3)
            ok = True
            for q in valid_fallback:
                if np.linalg.norm(p-q) < 0.5:
                    ok = False
                    break
            if ok:
                valid_fallback.append(p)
            attempts += 1
        if len(valid_fallback) == N:
            best_positions = np.array(valid_fallback)
        else:
            # Last resort: grid
            side = int(np.ceil(N ** (1/3)))
            grid = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
            grid.sort(key=lambda p: (p[0] - (side-1)/2)**2 + (p[1] - (side-1)/2)**2 + (p[2] - (side-1)/2)**2)
            best_positions = np.array(grid[:N], dtype=float) * 1.12

    return best_positions

# EVOLVE-BLOCK-END
