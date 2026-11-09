"""Morse cluster global minima search with integrated Basin Hopping and L-BFGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import morse_energy_gradient

# Global deadline shared across all six construct() calls
_START_TIME = time.time()
_TOTAL_BUDGET = 310.0  # 20s safety margin before 330s kill

def _time_left():
    return _TOTAL_BUDGET - (time.time() - _START_TIME)

def _morse_energy(pos, rho):
    """Compute total Morse energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    e = np.exp(rho * (1.0 - d))
    return float(np.sum(e * (e - 2.0)))

def _morse_gradient(pos, rho):
    """Compute Morse gradient."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    dist = np.where(dist < 1e-8, 1e-8, dist) # Avoid division by zero
    
    e = np.exp(rho * (1.0 - dist))
    # Gradient of E = sum exp(rho(1-r))(exp(rho(1-r))-2)
    # dE/dr = -rho * exp(rho(1-r)) * (2*exp(rho(1-r)) - 2)
    # Force on i from j: -dE/dr * (r_i - r_j)/r
    # Vectorized:
    # grad[i] = sum_j -dE/dr_ij * (pos[i] - pos[j])/r_ij
    # Note: dE/dr is scalar per pair.
    # Let's compute pairwise forces.
    
    # f_mag = -rho * e * (2*e - 2)
    f_mag = -rho * e * (2.0 * e - 2.0)
    
    # We need to sum vector contributions.
    # diff is (n, n, 3). dist is (n, n).
    # Force on i from j is f_mag_ij * (pos[i] - pos[j]) / dist_ij
    # This is f_mag[:, :, np.newaxis] * diff / dist[:, :, np.newaxis]
    # But we must be careful with the diagonal (zero).
    f_mag[:, :, np.newaxis] /= dist[:, :, np.newaxis]
    # Mask diagonal to 0
    mask = ~np.eye(n, dtype=bool)
    f_mag[:, :, np.newaxis] *= mask[:, :, np.newaxis]
    
    # Sum over j (axis 1) to get force on each i
    grad = np.sum(f_mag * diff, axis=1)
    
    return grad

def _build_fcc_cluster(n, spacing=1.0):
    """Build an FCC-like compact cluster."""
    points = []
    # Generate FCC lattice points
    # FCC nearest neighbor distance is 1. Lattice constant a = 1/sqrt(2).
    # Points at (i, j, k) * a where i+j+k is even? No, that's BCC.
    # FCC is corners + face centers.
    # Lattice vectors: a/2(1,1,0), a/2(1,0,1), a/2(0,1,1)
    # Let's just generate a cubic grid and filter by distance or use a simpler packing.
    # For simplicity, use a "close packed" spherical shell approach or just a dense grid.
    # Dense grid with spacing 1.0 is not FCC (NN dist 1.0).
    # Let's use a simple spherical shell packing which is often competitive for small n.
    
    # 1. Center
    points.append(np.zeros(3))
    
    # 2. Icosahedral shell (12)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico12 = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico12 /= np.linalg.norm(ico12[0])
    # Radius for icosahedron to have NN distance ~1?
    # Edge length of icosahedron with circumradius R is R * 1.0514...
    # We want edge ~ 1. So R ~ 0.95.
    # Actually, let's just place them at R=1.0 and let relaxation fix it.
    for v in ico12:
        points.append(v * 1.0)
        
    # 3. Octahedral shell (6) - wait, icosahedron is denser.
    # Let's add more shells.
    # Cuboctahedron (12)
    cubo12 = np.array([
        [1,1,0], [1,-1,0], [-1,1,0], [-1,-1,0],
        [1,0,1], [1,0,-1], [-1,0,1], [-1,0,-1],
        [0,1,1], [0,1,-1], [0,-1,1], [0,-1,-1]
    ], dtype=float)
    cubo12 /= np.linalg.norm(cubo12[0])
    for v in cubo12:
        points.append(v * 1.8)
        
    # 4. Add random points on a sphere if needed
    if len(points) < n:
        rng = np.random.default_rng(42)
        radius = 2.5
        while len(points) < n:
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * radius
            # Check distance
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < 0.8:
                    ok = False
                    break
            if ok:
                points.append(p)
            else:
                radius += 0.1
                if radius > 10:
                    break
                    
    return np.array(points[:n])

def _build_icosahedral_shells(n, spacing=1.0):
    """Build cluster using icosahedral shells."""
    points = [np.zeros(3)]
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico12 = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico12 /= np.linalg.norm(ico12[0])
    
    # Shell 1: Icosahedron (12)
    for v in ico12:
        points.append(v * 1.0)
        
    # Shell 2: 30 points (Truncated Icosahedron? Or just random on sphere)
    # Let's use a generic spherical shell
    if len(points) < n:
        rng = np.random.default_rng(123)
        radius = 2.0
        while len(points) < n:
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * radius
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < 0.9:
                    ok = False
                    break
            if ok:
                points.append(p)
            else:
                radius += 0.05
                if radius > 8:
                    # Fallback
                    p = rng.normal(0, 1, 3)
                    p = p / np.linalg.norm(p) * (radius + rng.random()*0.5)
                    points.append(p)
                    
    return np.array(points[:n])

def _build_random_compact(n, spacing=1.0, seed=42):
    """Build a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 1.2
    points = []
    for _ in range(n * 10):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        ok = True
        for q in points:
            if np.linalg.norm(p - q) < 0.8:
                ok = False
                break
        if ok:
            points.append(p)
            
    # Fill if not enough
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points[:n])

def _ensure_min_distance(pos, min_d=0.5):
    """Ensure no two atoms are closer than min_d."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(20):
        min_found = float('inf')
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_found:
                    min_found = d
                if d < min_d:
                    diff = pos[i] - pos[j]
                    if np.linalg.norm(diff) < 1e-10:
                        diff = np.array([1.0, 0, 0])
                    diff /= np.linalg.norm(diff)
                    # Push apart
                    push = (min_d - d) * 0.5 + 1e-6
                    pos[i] += diff * push
                    pos[j] -= diff * push
                    changed = True
        if min_found >= min_d or not changed:
            break
    return pos

def _local_relax(pos, rho, max_iter=100):
    """Local relaxation using L-BFGS-B."""
    x0 = pos.flatten()
    def obj(x):
        p = x.reshape(-1, 3)
        return _morse_energy(p, rho)
    def grad(x):
        p = x.reshape(-1, 3)
        return _morse_gradient(p, rho).flatten()
        
    try:
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B', options={'maxiter': max_iter, 'gtol': 1e-6})
        if res.success:
            return res.x.reshape(-1, 3), res.fun
        else:
            return res.x.reshape(-1, 3), obj(res.x)
    except Exception:
        return pos, obj(x0)

def _basin_hopping(pos, rho, n_iter=50, step_size=0.5, T=0.1, time_limit=10.0):
    """Basin Hopping algorithm."""
    start_time = time.time()
    current_pos = pos.copy()
    current_energy = _morse_energy(current_pos, rho)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(42)
    
    for i in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Step
        new_pos = current_pos + rng.normal(0, step_size, current_pos.shape)
        
        # Local relax
        relaxed_pos, relaxed_energy = _local_relax(new_pos, rho)
        
        # Metropolis accept
        delta_e = relaxed_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            current_pos = relaxed_pos
            current_energy = relaxed_energy
            
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
            
        # Anneal temperature
        T *= 0.95
        
    return best_pos, best_energy

def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Check time budget
    tl = _time_left()
    if tl < 10.0:
        # Very little time left
        pos = _build_random_compact(size, seed=size)
        pos = _ensure_min_distance(pos)
        return pos
        
    # Generate diverse starting configurations
    start_configs = []
    try:
        start_configs.append(_build_fcc_cluster(size))
    except Exception:
        pass
    try:
        start_configs.append(_build_icosahedral_shells(size))
    except Exception:
        pass
    try:
        start_configs.append(_build_random_compact(size, seed=size))
    except Exception:
        pass
    try:
        start_configs.append(_build_random_compact(size, seed=size+100))
    except Exception:
        pass
        
    if not start_configs:
        pos = np.zeros((size, 3))
        return pos
        
    best_energy = float('inf')
    best_pos = start_configs[0]
    
    # Time budget for search
    # We have tl seconds.
    # Let's allocate time per starting config.
    n_configs = len(start_configs)
    time_per_config = max(2.0, (tl - 10.0) / n_configs)
    
    for i, config in enumerate(start_configs):
        tl = _time_left()
        if tl < 15.0:
            break
            
        # Run basin hopping for this config
        # Adjust n_iter based on time
        # Estimate cost per iteration: local relax + energy eval
        # For n=80, local relax might take 0.1s. So 10s -> 100 iters.
        # For n=38, local relax might take 0.05s. So 10s -> 200 iters.
        est_cost = 0.05 + size * 0.001
        n_iter = int(time_per_config / est_cost)
        n_iter = max(10, min(n_iter, 200))
        
        try:
            pos, energy = _basin_hopping(config, parameter, n_iter=n_iter, time_limit=time_per_config * 0.9)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
        except Exception:
            continue
            
    # Final check and relaxation
    best_pos = _ensure_min_distance(best_pos)
    # One final local relax to ensure we are at a local minimum
    try:
        final_pos, final_energy = _local_relax(best_pos, parameter, max_iter=50)
        if final_energy < best_energy:
            best_pos = final_pos
            best_energy = final_energy
    except Exception:
        pass
        
    best_pos = _ensure_min_distance(best_pos)
    return best_pos
# EVOLVE-BLOCK-END
