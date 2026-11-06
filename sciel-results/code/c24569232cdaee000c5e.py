import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
from scipy.optimize import minimize

POTENTIAL = 'dzugutov'
_GLOBAL_START_TIME = None
_GLOBAL_DEADLINE = None

def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin, normalized to unit radius."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _get_mackay_55():
    """
    Generate the 55-atom Mackay icosahedron.
    """
    verts = _get_icosahedron_vertices()
    center = np.zeros(3)
    
    # 12 vertices
    shell1 = verts
    
    # 30 edge midpoints
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists[i, j] = d
            dists[j, i] = d
    min_dist = np.min(dists[dists > 0])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    edge_mids = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2
        edge_mids.append(mid)
    shell2 = np.array(edge_mids)
    
    # 12 face centers
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) > 1e-6: continue
            for k in range(j+1, 12):
                if abs(dists[i, k] - min_dist) > 1e-6: continue
                if abs(dists[j, k] - min_dist) > 1e-6: continue
                faces.append((i, j, k))
                
    face_centers = []
    for i, j, k in faces:
        fc = (verts[i] + verts[j] + verts[k]) / 3.0
        face_centers.append(fc)
    shell3 = np.array(face_centers)
    
    # Scale
    target_edge = 1.0
    current_edge = min_dist
    scale = target_edge / current_edge
    
    all_atoms = np.vstack([center, shell1, shell2, shell3])
    all_atoms = all_atoms * scale
    
    com = np.mean(all_atoms, axis=0)
    all_atoms -= com
    
    return all_atoms

def _generate_random_compact(size, rng):
    """Generate a random compact seed."""
    radius = 0.8 * size ** (1/3) + 1.0
    points = []
    for _ in range(size):
        found = False
        for _ in range(1000):
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
                found = True
                break
        if not found:
            p = rng.uniform(-radius*1.2, radius*1.2, 3)
            points.append(p)
    return np.array(points)

def _generate_seed(size, rng):
    """
    Generate a seed of 'size' atoms based on the 55-atom Mackay icosahedron.
    """
    base = _get_mackay_55()
    if size == 55:
        return base.copy()
    
    if size < 55:
        # Remove atoms from the outside in.
        removable = list(range(1, 55)) # All except center
        num_to_remove = 55 - size
        
        if num_to_remove <= 0:
            return base.copy()
            
        # Randomly select indices to remove
        removed_indices = set(rng.choice(removable, size=num_to_remove, replace=False))
        keep_indices = [i for i in range(55) if i not in removed_indices]
        
        return base[keep_indices]
        
    else:
        # size > 55
        extra = size - 55
        new_atoms = []
        for _ in range(extra):
            idx = rng.choice(55)
            base_atom = base[idx]
            for _ in range(50):
                offset = rng.normal(0, 0.8, 3)
                new_atom = base_atom + offset
                if all(np.linalg.norm(new_atom - a) >= 0.5 for a in base):
                    if all(np.linalg.norm(new_atom - a) >= 0.5 for a in new_atoms):
                        new_atoms.append(new_atom)
                        break
        if new_atoms:
            return np.vstack([base, np.array(new_atoms)])
        return base

def _lbfgs_optimize(positions):
    """
    Optimize positions using L-BFGS-B with the Dzugutov gradient.
    """
    def fun_and_grad(x):
        pos = x.reshape(-1, 3)
        try:
            result = dzugutov_energy_gradient(pos)
            if isinstance(result, tuple):
                energy, grad = result
            else:
                energy = 0.0
                grad = result
            grad_flat = grad.flatten()
            return energy, grad_flat
        except Exception as e:
            return 1e10, np.zeros_like(x)

    x0 = positions.flatten()
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True, 
                       options={'maxiter': 100, 'maxfun': 100, 'disp': False, 'maxls': 10})
        if res.success:
            return res.x.reshape(-1, 3), res.fun
        else:
            return res.x.reshape(-1, 3), res.fun
    except:
        return positions, 0.0

def _basin_hopping(positions, rng, time_limit, initial_temp=0.15, initial_step=0.25):
    """Adaptive basin hopping with energy-based acceptance."""
    current_pos = positions.copy()
    current_energy, current_relaxed = relax(current_pos, POTENTIAL)
    best_energy = current_energy
    best_pos = current_relaxed.copy()
    
    temperature = initial_temp
    step_size = initial_step
    consecutive_rejects = 0
    
    while time.time() < time_limit:
        perturbation = rng.normal(0, step_size, size=(len(positions), 3))
        new_pos = current_pos + perturbation
        
        try:
            new_energy, new_relaxed = relax(new_pos, POTENTIAL)
        except:
            continue
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_relaxed.copy()
            current_pos = new_relaxed.copy()
            consecutive_rejects = 0
            step_size *= 0.98
        else:
            delta_e = new_energy - current_energy
            acceptance_prob = np.exp(-delta_e / temperature)
            if rng.random() < acceptance_prob:
                current_pos = new_relaxed.copy()
                current_energy = new_energy
                consecutive_rejects += 1
            else:
                consecutive_rejects += 1
        
        if consecutive_rejects > 20:
            temperature *= 1.2
            step_size *= 1.1
            consecutive_rejects = 0
        else:
            temperature *= 0.97
            step_size *= 0.995
        
        temperature = max(temperature, 0.01)
        step_size = min(step_size, 0.5)
    
    return best_pos, best_energy

def construct(instance):
    global _GLOBAL_START_TIME, _GLOBAL_DEADLINE
    
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_START_TIME + 285.0

    size = instance["n"]
    start_time = time.time()
    
    instance_deadline = min(start_time + 45.0, _GLOBAL_DEADLINE - 3.0)
    
    rng = np.random.default_rng(42 + size)
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate diverse defect seeds
    num_seeds = 5
    seeds = []
    for i in range(num_seeds):
        seed = _generate_seed(size, rng)
        if seed.shape[0] == size:
            seeds.append(seed)
            
    # Ensure we have at least one seed
    if not seeds:
        seeds.append(_generate_random_compact(size, rng))

    # Phase 1: Relax all seeds and keep the best
    for seed in seeds:
        if time.time() > instance_deadline - 15:
            break
        try:
            e, pos = relax(seed, POTENTIAL)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except:
            continue
    
    if best_positions is None:
        best_positions = np.random.randn(size, 3) * 0.3
        try:
            best_energy, best_positions = relax(best_positions, POTENTIAL)
        except:
            pass

    # Phase 2: L-BFGS optimization on the best seed
    remaining_time = instance_deadline - time.time()
    if remaining_time > 10:
        lbfgs_pos, lbfgs_energy = _lbfgs_optimize(best_positions)
        try:
            verified_energy, verified_pos = relax(lbfgs_pos, POTENTIAL)
            if verified_energy < best_energy:
                best_energy = verified_energy
                best_positions = verified_pos
        except:
            pass

    # Phase 3: Basin hopping from the best seed
    remaining_time = instance_deadline - time.time()
    if remaining_time > 5:
        bh_pos, bh_energy = _basin_hopping(best_positions, rng, time.time() + remaining_time - 2)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_pos
    
    return best_positions
