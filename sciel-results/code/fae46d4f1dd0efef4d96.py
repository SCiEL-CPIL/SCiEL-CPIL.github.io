"""Morse cluster global minima search using basin hopping with structured initializers."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(positions, rho):
    """Compute exact Morse energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    # Morse potential: exp(rho*(1-r))*(exp(rho*(1-r)) - 2)
    arg = rho * (1.0 - dist)
    exp_arg = np.exp(arg)
    pair_e = exp_arg * (exp_arg - 2.0)
    return np.sum(pair_e)


def _morse_energy_gradient(positions, rho):
    """Compute gradient of Morse energy."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1.0)  # avoid division by zero
    arg = rho * (1.0 - dist)
    exp_arg = np.exp(arg)
    # dE/dr = -rho * exp(2*arg) * (2*exp(arg) - 1)
    # Actually: E = exp(arg)*(exp(arg)-2), dE/darg = exp(arg)*(exp(arg)-2) + exp(arg)*exp(arg) = exp(2arg) - 2*exp(arg)
    # darg/dr = -rho
    # dE/dr = -rho * (exp(2*arg) - 2*exp(arg))
    dE_dr = -rho * (np.exp(2.0 * arg) - 2.0 * exp_arg)
    # Gradient: sum over j of dE/dr_ij * (r_i - r_j)/r_ij
    # diff[i,j] = pos[i] - pos[j], so direction from j to i
    # grad_i = sum_j dE_dr[i,j] * diff[i,j] / dist[i,j]
    np.fill_diagonal(dist, 1.0)
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(n):
            if i != j:
                grad[i] += dE_dr[i, j] * diff[i, j] / dist[i, j]
    return grad


def _generate_fcc(n, spacing=1.0):
    """Generate n atoms in FCC-like arrangement."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Generate FCC lattice points
    coords = []
    max_c = int(np.ceil(n ** (1.0/3.0))) + 2
    
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                for bx, by, bz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + bx, j + by, k + bz]) * spacing
                    coords.append(p)
    
    coords = np.array(coords)
    if len(coords) < n:
        return coords[:n]
    
    # Select n points closest to center
    dists = np.sum(coords**2, axis=1)
    idx = np.argsort(dists)[:n]
    return coords[idx]


def _generate_icosahedral(n, spacing=1.0):
    """Generate n atoms in icosahedral shell arrangement."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = []
    for x, y, z in [(0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
                     (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
                     (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)]:
        p = np.array([x, y, z])
        p = p / np.linalg.norm(p)
        ico_verts.append(p)
    
    # Shell 1: 12 icosahedron vertices
    for v in ico_verts:
        points.append(v * spacing)
    
    # Shell 2: 30 points (cuboctahedron-like)
    if n > 13:
        shell2 = []
        for v in ico_verts:
            for u in ico_verts:
                if np.linalg.norm(v - u) < 1.0:  # adjacent
                    mid = (v + u) / 2
                    mid = mid / np.linalg.norm(mid)
                    if all(np.linalg.norm(mid - p/spacing) > 0.3 for p in points[:-1]):
                        shell2.append(mid * 1.6 * spacing)
        # Deduplicate
        unique_shell2 = []
        for p in shell2:
            if all(np.linalg.norm(p - q) > 0.5 * spacing for q in unique_shell2):
                unique_shell2.append(p)
        points.extend(unique_shell2[:n - len(points)])
    
    # Fill remaining with random points in sphere
    while len(points) < n:
        rng = np.random.default_rng(42)
        p = rng.uniform(-2.5, 2.5, 3) * spacing
        if all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
            points.append(p)
    
    return np.array(points[:n])


def _basin_hopping(positions, rho, max_iter=200, step_size=0.3, temperature=0.5, seed=42):
    """Simple basin hopping with local relaxation."""
    rng = np.random.default_rng(seed)
    n = len(positions)
    
    current_pos = positions.copy()
    current_energy = _morse_energy(current_pos, rho)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    for it in range(max_iter):
        # Random perturbation
        perturbation = rng.normal(0, step_size, size=current_pos.shape)
        trial_pos = current_pos + perturbation
        
        # Local relaxation
        try:
            relaxed_energy, relaxed_pos = relax(trial_pos, 'morse', rho)
        except:
            relaxed_energy = _morse_energy(trial_pos, rho)
            relaxed_pos = trial_pos
        
        # Metropolis acceptance
        delta_e = relaxed_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / temperature):
            current_pos = relaxed_pos
            current_energy = relaxed_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        
        # Cool down
        temperature *= 0.99
    
    return best_energy, best_pos


def _local_minimize(positions, rho):
    """Local minimization using gradient descent with line search."""
    pos = positions.copy()
    n = len(pos)
    lr = 0.01
    max_iter = 500
    
    for it in range(max_iter):
        grad = _morse_energy_gradient(pos, rho)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-6:
            break
        
        # Gradient descent step
        trial_pos = pos - lr * grad
        
        # Check minimum distance constraint
        min_dist = 0.5
        diff = trial_pos[:, np.newaxis, :] - trial_pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        if np.min(dist) < min_dist:
            # Scale back
            scale = min_dist / (np.min(dist) * 0.99)
            trial_pos = pos - lr * grad * scale
        
        energy = _morse_energy(trial_pos, rho)
        if energy < _morse_energy(pos, rho):
            pos = trial_pos
        else:
            lr *= 0.5
    
    return _morse_energy(pos, rho), pos


def construct(instance):
    """Find low-energy Morse cluster configurations."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    time_limit = 48.0  # ~300s / 6 instances with margin
    
    best_energy = np.inf
    best_positions = None
    
    # Try different initial configurations
    initializers = []
    
    # FCC
    fcc_pos = _generate_fcc(size, spacing=1.0)
    initializers.append(fcc_pos)
    
    # Icosahedral
    ico_pos = _generate_icosahedral(size, spacing=1.0)
    initializers.append(ico_pos)
    
    # Random compact
    rng = np.random.default_rng(size)
    radius = size ** (1.0/3.0) * 0.8
    for _ in range(3):
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
                points.append(p)
        initializers.append(np.array(points))
    
    # For each initializer, do basin hopping
    for init_idx, init_pos in enumerate(initializers):
        if time.time() - start_time > time_limit:
            break
        
        # First do local relaxation
        try:
            e0, pos0 = relax(init_pos, 'morse', rho)
        except:
            e0 = _morse_energy(init_pos, rho)
            pos0 = init_pos
        
        if e0 < best_energy:
            best_energy = e0
            best_positions = pos0.copy()
        
        # Basin hopping from this starting point
        if time.time() - start_time > time_limit:
            break
        
        bh_energy, bh_pos = _basin_hopping(pos0, rho, max_iter=50, step_size=0.2, 
                                           temperature=0.3, seed=size + init_idx)
        
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_pos.copy()
    
    # Final local relaxation of best
    if best_positions is not None and time.time() - start_time < time_limit - 5:
        try:
            final_energy, final_pos = relax(best_positions, 'morse', rho)
            if final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos
        except:
            pass
    
    # Ensure minimum distance constraint
    if best_positions is not None:
        diff = best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        if np.min(dist) < 0.5:
            # Try to fix by small perturbations
            for _ in range(100):
                diff = best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff**2, axis=2))
                np.fill_diagonal(dist, np.inf)
                
                min_idx = np.unravel_index(np.argmin(dist), dist.shape)
                if dist[min_idx] >= 0.5:
                    break
                
                # Push apart
                i, j = min_idx
                direction = best_positions[i] - best_positions[j]
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                shift = (0.5 - dist[min_idx]) / 2 + 0.01
                best_positions[i] += direction * shift
                best_positions[j] -= direction * shift
    
    return best_positions
# EVOLVE-BLOCK-END
