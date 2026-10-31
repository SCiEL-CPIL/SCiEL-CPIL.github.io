"""LJ38 cluster global minimum search using Multi-Start Local Optimization."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _lj_energy(coords):
    """Compute total Lennard-Jones energy for a set of coordinates."""
    n = len(coords)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diff = coords[i] - coords[j]
            r2 = np.dot(diff, diff)
            if r2 < 1e-12: continue
            r6 = r2 ** 3
            energy += 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return energy


def _lj_force(coords):
    """Compute total Lennard-Jones force for a set of coordinates."""
    n = len(coords)
    forces = np.zeros_like(coords)
    for i in range(n):
        for j in range(i + 1, n):
            diff = coords[i] - coords[j]
            r2 = np.dot(diff, diff)
            if r2 < 1e-12: continue
            r6 = r2 ** 3
            r12 = r6 * r6
            factor = 48.0 * (1.0 / (r12 * r2) - 0.5 / (r6 * r2))
            f = factor * diff
            forces[i] += f
            forces[j] -= f
    return forces


def _relax(coords):
    """Relax coordinates to a local minimum using L-BFGS."""
    def energy_func(x):
        return _lj_energy(x.reshape(-1, 3))
    
    def force_func(x):
        return -_lj_force(x.reshape(-1, 3)).flatten()
    
    x0 = coords.flatten()
    try:
        result = minimize(
            energy_func,
            x0,
            jac=force_func,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.fun < energy_func(x0):
            return result.x.reshape(-1, 3), result.fun
    except Exception:
        pass
    return coords, _lj_energy(coords)


def _init_fcc(n_atoms=38):
    """Generate FCC truncated octahedron coordinates."""
    a = 2.0 ** (2.0 / 3.0)
    points = []
    max_idx = 5
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for bx, by, bz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    x = (i + bx) * a
                    y = (j + by) * a
                    z = (k + bz) * a
                    points.append((x, y, z))
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    return points[sorted_idx[:n_atoms]]


def construct():
    """Construct initial positions for LJ38 cluster using multi-start strategy."""
    best_coords = None
    best_energy = float('inf')
    
    # Strategy 1: FCC Init
    coords_fcc = _init_fcc(N)
    relaxed_fcc, e_fcc = _relax(coords_fcc)
    if e_fcc < best_energy:
        best_energy = e_fcc
        best_coords = relaxed_fcc
    
    # Strategy 2: Random Init (Uniform in sphere)
    np.random.seed(42)
    for _ in range(5):
        # Generate random points in a sphere of radius ~5
        r = np.random.uniform(0, 5, size=(N, 3))
        # Ensure no overlaps by checking min dist, if too close, jitter
        for attempt in range(10):
            min_d = float('inf')
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(r[i]-r[j])
                    if d < min_d: min_d = d
            if min_d >= 0.5:
                break
            r *= 1.1
        
        relaxed_rand, e_rand = _relax(r)
        if e_rand < best_energy:
            best_energy = e_rand
            best_coords = relaxed_rand
            
    # Strategy 3: Perturbed FCC
    coords_pert = _init_fcc(N)
    coords_pert += np.random.normal(0, 0.1, size=(N, 3))
    # Ensure min dist
    for attempt in range(10):
        min_d = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(coords_pert[i]-coords_pert[j])
                if d < min_d: min_d = d
        if min_d >= 0.5:
            break
        coords_pert *= 1.05
        
    relaxed_pert, e_pert = _relax(coords_pert)
    if e_pert < best_energy:
        best_energy = e_pert
        best_coords = relaxed_pert

    # Final check for min distance constraint
    min_dist = 0.5
    for _ in range(1000):
        min_pair_dist = float('inf')
        min_pair_idx = (0, 1)
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_coords[i] - best_coords[j])
                if d < min_pair_dist:
                    min_pair_dist = d
                    min_pair_idx = (i, j)
        if min_pair_dist >= min_dist:
            break
        i, j = min_pair_idx
        diff = best_coords[i] - best_coords[j]
        norm = np.linalg.norm(diff)
        if norm > 1e-10:
            diff = diff / norm
        else:
            diff = np.array([1.0, 0.0, 0.0])
        best_coords[i] += diff * 0.01
        best_coords[j] -= diff * 0.01

    return best_coords
# EVOLVE-BLOCK-END
