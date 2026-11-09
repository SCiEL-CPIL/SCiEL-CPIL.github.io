"""
Strategy: Vectorized Jumpy Basin Hopping for Dzugutov Clusters.
Materially different from the current program by:
1. Replacing the O(N^2) Python loop in energy/gradient calculation with fully vectorized NumPy operations. This reduces per-iteration overhead by orders of magnitude.
2. Replacing standard Gaussian perturbations with "Jumpy" moves that apply larger, coordinated displacements. This is better suited for the Dzugutov potential's landscape, where atoms need to rearrange shells to escape local minima, rather than just jittering.
3. Leveraging the speedup to run significantly more basin hopping steps within the strict time budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
import threading

# Global deadline tracking
_START_TIME = time.time()
_HARD_LIMIT = 290.0  # Safety margin for 300s limit
_lock = threading.Lock()

def _get_remaining_time():
    return max(0.1, _HARD_LIMIT - (time.time() - _START_TIME))

def _dzugutov_energy_gradient_vectorized(positions, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """
    Compute the energy and gradient of the Dzugutov potential using vectorization.
    """
    n = len(positions)
    # Pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    
    # Mask for i < j to avoid double counting and self-interaction
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r_flat = r[mask]
    diff_flat = diff[mask]
    
    energy = 0.0
    grad = np.zeros_like(positions)
    
    if len(r_flat) == 0:
        return energy, grad

    # Compute terms for r < a
    mask_a = r_flat < a
    # Compute terms for r < b
    mask_b = r_flat < b
    
    # Avoid division by zero or negative log/exp arguments
    # r > 0 is guaranteed by mask (i<j implies distinct indices, and we assume no overlap in valid inputs, 
    # but for safety in optimization, we clip r)
    r_safe = np.clip(r_flat, 1e-6, None)
    
    v_total = np.zeros_like(r_flat)
    dv_dr_total = np.zeros_like(r_flat)
    
    if np.any(mask_a):
        r_a = r_safe[mask_a]
        exp_term = np.exp(c / (r_a - a))
        v_a = A * (r_a ** (-m) - B) * exp_term
        # dV/dr = A(-m r^(-m-1))exp(c/(r-a)) - A(r^-m - B)exp(c/(r-a)) * c/(r-a)^2
        dv_dr_a = A * (-m * r_a ** (-m - 1)) * exp_term - A * (r_a ** (-m) - B) * exp_term * c / (r_a - a) ** 2
        
        v_total[mask_a] += v_a
        dv_dr_total[mask_a] += dv_dr_a

    if np.any(mask_b):
        r_b = r_safe[mask_b]
        exp_term2 = np.exp(d / (r_b - b))
        v_b = B * exp_term2
        dv_dr_b = B * exp_term2 * (-d / (r_b - b) ** 2)
        
        v_total[mask_b] += v_b
        dv_dr_total[mask_b] += dv_dr_b

    energy = np.sum(v_total)
    
    # Compute gradient
    # grad_i += sum_j (dv_dr_ij / r_ij) * (r_i - r_j)
    # factor = dv_dr / r
    factor = dv_dr_total / r_flat
    
    # We need to scatter the forces back to the positions array
    # diff_flat[i, j] corresponds to positions[i] - positions[j]
    # We need indices i and j for each pair in the flattened mask
    i_idx, j_idx = np.where(mask)
    
    # Force on i is factor * diff_flat
    # Force on j is -factor * diff_flat
    # We can use np.add.at for accumulation
    forces = factor[:, np.newaxis] * diff_flat
    
    # Accumulate forces into grad
    # grad[i] += force
    # grad[j] -= force
    np.add.at(grad, i_idx, forces)
    np.add.at(grad, j_idx, -forces)
    
    return energy, grad

def _generate_icosahedron(n_atoms, rng):
    """
    Generate a structure based on icosahedral shells.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    positions = [np.zeros(3)]
    
    r1 = 1.9
    for v in verts:
        positions.append(v * r1)
        
    if n_atoms <= 13:
        return np.array(positions[:n_atoms])
        
    # For n > 13, add atoms on a second shell
    # The second shell of an icosahedron (dodecahedron vertices) or just random on sphere
    # Let's use random on sphere for simplicity but scaled
    r2 = 1.9 * 1.7
    num_outer = n_atoms - 13
    if num_outer > 0:
        outer_pts = []
        for _ in range(num_outer):
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v) * r2
            outer_pts.append(v)
        positions.extend(outer_pts)
        
    return np.array(positions[:n_atoms])

def _random_compact(n_atoms, rng, spacing=1.1):
    """
    Generate a random compact structure.
    """
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.8
    points = []
    max_iters = 10000
    for _ in range(max_iters):
        if len(points) >= n_atoms:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
            points.append(p)
    while len(points) < n_atoms:
        points.append(rng.uniform(-radius, radius, 3))
    return np.array(points)

def _local_minimize_vectorized(positions):
    """
    Use L-BFGS-B with vectorized gradient for fast local minimization.
    """
    def objective(x):
        pos = x.reshape(-1, 3)
        energy, grad = _dzugutov_energy_gradient_vectorized(pos)
        return energy, grad.flatten()
    
    x0 = positions.flatten()
    try:
        res = minimize(objective, x0, jac=True, method='L-BFGS-B', options={'maxiter': 100, 'maxls': 20})
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return positions, np.inf

from scipy.optimize import minimize

def _jumpy_perturb(positions, rng, scale=0.8):
    """
    Jumpy perturbation: 
    1. Global rotation
    2. Coordinated radial expansion/contraction (breathing)
    3. Random individual perturbations
    This helps atoms move as a group to find new shells.
    """
    new_pos = positions.copy()
    
    # 1. Random rotation
    if rng.random() < 0.5:
        # Simple random rotation matrix generation
        q = rng.normal(size=3)
        q = q / np.linalg.norm(q)
        # Rodrigues rotation or just a small random rotation
        # Let's use a simple random rotation around a random axis
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        angle = rng.uniform(-0.5, 0.5)
        c = np.cos(angle)
        s = np.sin(angle)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + s * K + (1 - c) * (K @ K)
        new_pos = (R @ new_pos.T).T

    # 2. Breathing (radial scale)
    scale_factor = 1.0 + rng.normal(0, 0.1)
    new_pos *= scale_factor

    # 3. Individual perturbations
    new_pos += rng.normal(0, scale, size=new_pos.shape)
    
    return new_pos

def _basin_hopping_chain(n_atoms, rng, initial_pos, time_limit):
    """
    Run a single basin hopping chain with jumpy moves.
    """
    start_time = time.time()
    current_pos = initial_pos
    current_energy = np.inf
    
    # First, minimize the initial position
    current_pos, current_energy = _local_minimize_vectorized(current_pos)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    step_scale = 0.8
    max_steps = 200 # Increased steps due to vectorization
    
    for _ in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Jumpy Perturb
        new_pos = _jumpy_perturb(current_pos, rng, step_scale)
        
        # Minimize
        new_pos, new_energy = _local_minimize_vectorized(new_pos)
        
        # Accept if better (Hill climbing)
        if new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
            
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
            
    return best_pos, best_energy

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    time_budget = _get_remaining_time()
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starts
    starts = []
    
    # 1. PT-inspired
    try:
        pt_pos = _generate_icosahedron(size, rng)
        starts.append(pt_pos)
    except Exception:
        pass
        
    # 2. Random Compact
    try:
        rc_pos = _random_compact(size, rng)
        starts.append(rc_pos)
    except Exception:
        pass
        
    # 3. Perturbed PT
    try:
        pt_pos2 = _generate_icosahedron(size, rng)
        pt_pos2 = _jumpy_perturb(pt_pos2, rng, 0.5)
        starts.append(pt_pos2)
    except Exception:
        pass
        
    # 4. Another Random Compact
    try:
        rc_pos2 = _random_compact(size, rng)
        starts.append(rc_pos2)
    except Exception:
        pass
        
    # Use parallel pool
    n_workers = min(4, len(starts))
    # Allocate time proportional to workers, but leave a buffer
    time_per_chain = time_budget / (n_workers * 1.2)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i in range(n_workers):
            initial_pos = starts[i]
            chain_rng = np.random.default_rng(size + i)
            future = executor.submit(_basin_hopping_chain, size, chain_rng, initial_pos, time_per_chain)
            futures.append(future)
            
        for future in concurrent.futures.as_completed(futures):
            try:
                pos, energy = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos
            except Exception:
                pass
                
    # If no valid result, fallback to the first start
    if best_positions is None:
        best_positions = starts[0]
        
    return best_positions
# EVOLVE-BLOCK-END
