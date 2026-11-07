"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 280.0  # Safety margin before 300s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _is_valid(positions, min_dist=0.35):
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    # Efficient vectorized check
    for i in range(n - 1):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True

def _generate_fcc(n):
    """Generate a cluster of n atoms in an FCC-like structure."""
    max_r = int(np.ceil(n ** (1/3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                    candidates.append((np.linalg.norm(p), p))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)

def _generate_bcc(n):
    """Generate a cluster of n atoms in a BCC-like structure."""
    max_r = int(np.ceil(n ** (1/3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k])
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                candidates.append((np.linalg.norm(p1), p1))
                candidates.append((np.linalg.norm(p2), p2))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)

def _generate_random_sphere(n, rng):
    """Generate n random points inside a sphere."""
    # Radius roughly matches FCC density
    r = 2.5 * (n / 4.0) ** (1/3)
    pts = rng.uniform(-r, r, (n, 3))
    # Keep only those inside sphere
    norms = np.linalg.norm(pts, axis=1)
    mask = norms <= r
    while np.sum(mask) < n:
        new_pts = rng.uniform(-r, r, (n, 3))
        new_norms = np.linalg.norm(new_pts, axis=1)
        new_mask = new_norms <= r
        pts = np.concatenate([pts, new_pts])
        mask = np.concatenate([mask, new_mask])
    pts = pts[mask][:n]
    return pts

def _bh_step(positions, kind, rng, sigma):
    """Perform one basin hopping step: perturb and relax."""
    perturbed = positions + rng.normal(0, sigma, positions.shape)
    try:
        energy, relaxed_pos = relax(perturbed, 'sutton_chen', kind)
        if np.isfinite(energy) and _is_valid(relaxed_pos):
            return energy, relaxed_pos
    except Exception:
        pass
    return np.inf, None

def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    rng = np.random.default_rng(42 + n)
    
    # Initial structures: FCC, BCC, Random
    generators = [_generate_fcc, _generate_bcc]
    # Add random
    initial_structures = []
    for gen in generators:
        try:
            pos = gen(n)
            if _is_valid(pos):
                initial_structures.append(pos)
        except Exception:
            pass
    
    # Add a few random structures
    for _ in range(3):
        try:
            pos = _generate_random_sphere(n, rng)
            if _is_valid(pos):
                initial_structures.append(pos)
        except Exception:
            pass
    
    # Relax all initial structures
    for pos in initial_structures:
        if _time_left() < 5.0:
            break
        try:
            energy, relaxed_pos = relax(pos, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed_pos):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos.copy()
        except Exception:
            continue
    
    if best_pos is None:
        # Fallback: return FCC
        best_pos = _generate_fcc(n)
        if not _is_valid(best_pos):
            best_pos = _generate_bcc(n)
        return best_pos
    
    # Basin Hopping with adaptive sigma
    sigma = 0.5
    current_energy = best_energy
    current_pos = best_pos
    no_improve_count = 0
    
    while _time_left() > 10.0:
        # Perturb
        perturbed = current_pos + rng.normal(0, sigma, current_pos.shape)
        
        # Relax
        try:
            new_energy, new_pos = relax(perturbed, 'sutton_chen', kind)
        except Exception:
            continue
        
        if not np.isfinite(new_energy) or not _is_valid(new_pos):
            continue
        
        # Metropolis acceptance
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            no_improve_count = 0
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        else:
            delta_e = new_energy - current_energy
            # Temperature-like acceptance probability
            # Use a simple cooling schedule
            T = 0.1 * (1.0 - no_improve_count / 100.0)
            if T > 0 and rng.random() < np.exp(-delta_e / T):
                current_energy = new_energy
                current_pos = new_pos
            no_improve_count += 1
        
        # Adapt sigma
        if no_improve_count > 20:
            sigma *= 0.8
            no_improve_count = 0
        elif no_improve_count < 5:
            sigma *= 1.1
    
    # Final check
    if best_pos is None:
        best_pos = _generate_fcc(n)
        if not _is_valid(best_pos):
            best_pos = _generate_bcc(n)
    
    return best_pos

# EVOLVE-BLOCK-END
