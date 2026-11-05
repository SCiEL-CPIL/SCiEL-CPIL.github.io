"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _dzugutov_energy(pos):
    """Compute Dzugutov energy for a cluster."""
    n = len(pos)
    if n < 2:
        return 0.0
    
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    # Mask for valid pairs
    mask = dist < b
    
    energy = 0.0
    
    # Only compute for pairs within cutoff
    for i in range(n):
        for j in range(i + 1, n):
            r = dist[i, j]
            if r < a:
                v = A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                v = B * np.exp(d / (r - b))
            else:
                continue
            
            energy += v
            
    return energy

def _dzugutov_gradient(pos):
    """Compute Dzugutov gradient for a cluster."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    grad = np.zeros_like(pos)
    
    # Only compute for pairs within cutoff
    for i in range(n):
        for j in range(i + 1, n):
            r = dist[i, j]
            if r < a:
                v = A * (r ** (-m) - B) * np.exp(c / (r - a))
                dv_dr = A * (-m * r ** (-m - 1) * np.exp(c / (r - a)) + (r ** (-m) - B) * np.exp(c / (r - a)) * (-c / (r - a) ** 2))
            elif r < b:
                v = B * np.exp(d / (r - b))
                dv_dr = B * np.exp(d / (r - b)) * (-d / (r - b) ** 2)
            else:
                continue
            
            # Gradient contribution
            rij = pos[i] - pos[j]
            factor = dv_dr / r
            grad[i] -= factor * rij
            grad[j] += factor * rij
            
    return grad

def _generate_icosahedron(radius=1.0):
    """Generate vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts * (radius / norms[:, np.newaxis])
    return verts

def _generate_polytetrahedral(n, spacing=1.1):
    """Generate a polytetrahedral-like structure."""
    # Start with an icosahedron core
    ico = _generate_icosahedron(radius=spacing * 0.9)
    points = [ico[0]]  # Central atom
    
    # Add icosahedron vertices
    for v in ico[1:]:
        if len(points) < n:
            points.append(v)
    
    # Add more atoms in tetrahedral arrangements
    rng = np.random.default_rng(42)
    tet_verts = np.array([
        [1, 1, 1], [1, -1, -1],
        [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    tet_verts = tet_verts * (spacing * 0.8 / np.linalg.norm(tet_verts[0]))
    
    while len(points) < n:
        # Pick a random existing atom and add tetrahedral neighbors
        base_idx = rng.integers(0, len(points))
        base = points[base_idx]
        for v in tet_verts:
            if len(points) < n:
                candidate = base + v
                # Check minimum distance
                if all(np.linalg.norm(candidate - q) >= 0.5 * spacing for q in points):
                    points.append(candidate)
    
    return np.array(points[:n])

def _local_minimize_lbfgs(pos):
    """Local minimization using L-BFGS-B."""
    try:
        res = minimize(_dzugutov_energy, pos, jac=_dzugutov_gradient, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        return res.x, res.fun
    except Exception:
        return pos, _dzugutov_energy(pos)

def construct(instance):
    """Find low-energy structures for Dzugutov clusters."""
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    time_limit = 290.0  # Safety margin for 300s limit
    
    # Generate diverse starting structures
    candidates = []
    
    # 1. Polytetrahedral structures (best guess for Dzugutov)
    candidates.append(_generate_polytetrahedral(size, spacing=1.1))
    
    # 2. Icosahedral-like structure
    ico = _generate_icosahedron(radius=1.1 * size ** (1/3) * 0.6)
    if len(ico) >= size:
        candidates.append(ico[:size])
    else:
        # Pad with random points
        rng = np.random.default_rng(size)
        extra = rng.uniform(-2, 2, (size - len(ico), 3))
        candidates.append(np.vstack([ico, extra]))
        
    # 3. Random compact clusters
    for seed in [size, size + 1]:
        rng = np.random.default_rng(seed)
        radius = 1.1 * size ** (1.0 / 3.0) * 0.8
        points = []
        attempts = 0
        while len(points) < size and attempts < 5000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                    points.append(p)
            attempts += 1
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        candidates.append(np.array(points))
    
    best_energy = np.inf
    best_pos = None
    
    # Basin Hopping strategy
    # We will perform basin hopping on each candidate
    # Basin hopping: local minimize -> random kick -> local minimize -> accept/reject
    
    rng = np.random.default_rng(12345)
    
    for init_pos in candidates:
        if time.time() - start_time > time_limit:
            break
            
        current_pos = init_pos.copy()
        current_energy, _ = _local_minimize_lbfgs(current_pos)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
            
        # Perform basin hopping iterations
        # Number of iterations depends on time budget
        max_hops = 50
        hop_size = 0.5
        
        for hop in range(max_hops):
            if time.time() - start_time > time_limit:
                break
                
            # Random kick
            kick = rng.normal(0, hop_size, current_pos.shape)
            # Ensure minimum distance constraint is not violated too badly
            # (L-BFGS will likely fix it, but if it's too bad, reject)
            new_pos = current_pos + kick
            
            # Local minimize
            new_pos, new_energy = _local_minimize_lbfgs(new_pos)
            
            # Metropolis acceptance
            if new_energy < current_energy:
                current_pos = new_pos
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos.copy()
            else:
                # Accept with probability exp(-delta_E / T)
                # T should be scaled to typical energy differences
                # Dzugutov energies are O(-100 to -1000)
                # Let's use a simple adaptive T or fixed T
                T = 5.0 # Typical energy scale
                delta_E = new_energy - current_energy
                if delta_E < 0:
                    accept = True
                else:
                    accept = np.exp(-delta_E / T) > rng.random()
                
                if accept:
                    current_pos = new_pos
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_pos = new_pos.copy()
                    
            # Adapt hop size
            if current_energy < best_energy:
                hop_size *= 0.9
            else:
                hop_size *= 1.1
            hop_size = np.clip(hop_size, 0.1, 2.0)
    
    if best_pos is None:
        # Fallback: return a simple random cluster
        rng = np.random.default_rng(0)
        best_pos = rng.uniform(-2, 2, (size, 3))
        
    # Final relaxation with the provided relax function if time permits
    if time.time() - start_time < time_limit - 10:
        try:
            energy, relaxed_pos = relax(best_pos, potential, None)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos
        except Exception:
            pass
            
    return best_pos
# EVOLVE-BLOCK-END
