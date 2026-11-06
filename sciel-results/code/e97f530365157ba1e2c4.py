import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time management
_GLOBAL_START_TIME = None
_GLOBAL_DEADLINE = None

def _get_global_deadline():
    global _GLOBAL_START_TIME, _GLOBAL_DEADLINE
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        # Evaluator kills at 330s. We have 6 instances.
        # Let's aim to finish well before 300s to be safe.
        # 330 / 6 = 55s per instance.
        # Let's set a deadline for the whole run, but also per-instance limits.
        _GLOBAL_DEADLINE = _GLOBAL_START_TIME + 290.0 # 10s safety margin
    return _GLOBAL_DEADLINE

def _morse_energy(pos, rho):
    """
    Compute Morse energy.
    E = sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij)) - 2)
    """
    n = len(pos)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    # Zero out diagonal
    np.fill_diagonal(dists, 0.0)
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    r = dists[iu]
    # Filter out very small distances to avoid exp overflow? 
    # Morse potential goes to infinity as r->0.
    # The constraint is r >= 0.5.
    # exp(rho(1-r)) can be large if r is small and rho is large.
    # rho=10, r=0.5 -> exp(10*0.5) = exp(5) ~ 148.
    # r=0.1 -> exp(9) ~ 8103.
    # It's fine for double precision unless r is extremely close to 0.
    x = rho * (1.0 - r)
    exp_x = np.exp(x)
    e = exp_x * (exp_x - 2.0)
    return np.sum(e)

def _morse_gradient(pos, rho):
    """
    Compute gradient of Morse energy.
    dE/dr_ij = 2 * exp(2x) * rho - 2 * exp(x) * rho  (Wait, let's derive)
    E_ij = exp(x)(exp(x) - 2) = exp(2x) - 2exp(x)
    dE_ij/dx = 2exp(2x) - 2exp(x)
    dx/dr = -rho
    dE_ij/dr = -rho(2exp(2x) - 2exp(x))
    dE_ij/dpos_i = dE_ij/dr * d(r)/dpos_i = dE_ij/dr * (pos_i - pos_j)/r
    """
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, 1.0) # Avoid div by 0
    
    iu = np.triu_indices(n, k=1)
    r = dists[iu]
    
    # Safety: if r < 1e-6, gradient is huge. But constraint is 0.5.
    # We assume valid inputs.
    
    x = rho * (1.0 - r)
    exp_x = np.exp(x)
    exp_2x = exp_x * exp_x
    
    # dE/dr = -rho * (2*exp(2x) - 2*exp(x))
    dE_dr = -rho * (2.0 * exp_2x - 2.0 * exp_x)
    
    grad = np.zeros_like(pos)
    
    # For each pair (i, j), contribution to grad_i and grad_j
    # dE/dpos_i = (dE_dr / r) * (pos_i - pos_j)
    # dE/dpos_j = (dE_dr / r) * (pos_j - pos_i)
    
    # Vectorized update is tricky with triu. Let's use a loop for clarity and robustness 
    # given n <= 80. n^2/2 = 3160 pairs. This is very fast in Python if vectorized, 
    # but even a loop is < 1ms.
    
    # Let's try vectorized accumulation.
    # Create a matrix of forces F_ij = (dE_dr / r) * (pos_i - pos_j)
    # This is O(N^3) memory if not careful.
    
    # Simpler: iterate over pairs.
    for i, j in zip(iu[0], iu[1]):
        rij = diff[i, j]
        r_ij = dists[i, j]
        if r_ij < 1e-6:
            continue
        factor = dE_dr[i, j] / r_ij
        force = factor * rij
        grad[i] += force
        grad[j] -= force
        
    return grad

def _min_dist_check(pos, min_dist=0.5):
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_dist

def _fix_min_dist(pos, min_dist=0.5):
    """
    Simple relaxation to ensure min distance.
    If atoms are too close, push them apart.
    """
    n = len(pos)
    if n < 2:
        return pos
    
    pos = pos.copy()
    for _ in range(50): # Max iterations
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        min_d = np.min(dists)
        if min_d >= min_dist:
            break
            
        # Find pairs violating constraint
        iu = np.triu_indices(n, k=1)
        r = dists[iu]
        mask = r < min_dist
        
        if not np.any(mask):
            break
            
        # Push apart
        # Simple iterative correction
        for i, j in zip(iu[0][mask], iu[1][mask]):
            rij = diff[i, j]
            r_ij = dists[i, j]
            if r_ij < 1e-6:
                # Random small push
                rij = np.random.normal(0, 0.01, 3)
                r_ij = np.linalg.norm(rij)
            
            # Target distance is min_dist + small epsilon
            target = min_dist + 0.01
            # Scale vector to target length
            # Move each atom half the distance
            shift = (target - r_ij) / (2.0 * r_ij) * rij
            pos[i] += shift
            pos[j] -= shift
            
    return pos

def _generate_fcc_cluster(n):
    """Generate an FCC cluster of n atoms centered at origin."""
    side = int(np.ceil(n ** (1.0/3.0))) + 3
    pts = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                pts.append([i, j, k])
                pts.append([i+0.5, j+0.5, k])
                pts.append([i+0.5, j, k+0.5])
                pts.append([i, j+0.5, k+0.5])
    
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    indices = np.argsort(dists)
    pts = pts[indices[:n]]
    return pts

def _generate_hcp_cluster(n):
    """Generate an HCP-like cluster."""
    pts = []
    max_r = int(np.ceil(np.sqrt(n))) + 2
    layer_height = np.sqrt(2.0/3.0)
    
    for layer in range(-5, 6):
        z = layer * layer_height
        offset_x = 0.5 if layer % 2 != 0 else 0.0
        for i in range(-max_r, max_r+1):
            for j in range(-max_r, max_r+1):
                x = i + 0.5 * j + offset_x
                y = j * np.sqrt(3) / 2.0
                if np.linalg.norm([x, y]) <= max_r:
                    pts.append([x, y, z])
                    
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    indices = np.argsort(dists)
    pts = pts[indices[:n]]
    return pts

def _generate_icosahedron(n):
    """
    Generate Mackay icosahedron shells.
    Magic numbers: 13, 55, 147.
    """
    phi = (1 + np.sqrt(5)) / 2
    # 13 atom icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts *= 0.5 # Scale so edge length is 1
    
    pts = [np.zeros(3)] + [v for v in verts]
    
    if n <= 13:
        return np.array(pts[:n])
    
    # 55 atom icosahedron
    # 1 center + 12 vertices + 30 edge centers + 12 face centers
    # Edge centers are midpoints of edges.
    # Vertices indices: 0-11.
    # Edges connect vertices that are distance 1 apart.
    # In the scaled coords, edge length is 1.
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, 1.0, atol=1e-3):
                edges.append((i, j))
    
    edge_centers = [(verts[i] + verts[j]) / 2.0 for i, j in edges]
    
    # Face centers
    # Triangular faces of icosahedron.
    # There are 20 faces.
    # We can find them by checking triples of vertices that form a triangle.
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if np.isclose(d1, 1.0, atol=1e-3) and np.isclose(d2, 1.0, atol=1e-3) and np.isclose(d3, 1.0, atol=1e-3):
                    # Check if it's a face (normal points outwards)
                    # Actually, just take the centroid
                    face_center = (verts[i] + verts[j] + verts[k]) / 3.0
                    faces.append(face_center)
                    
    # Remove duplicates in faces (each face is unique)
    # There should be 20.
    
    pts_55 = pts + edge_centers + faces
    
    if n <= 55:
        # If n is between 13 and 55, we can't just truncate 55 easily without breaking symmetry.
        # But for n=38, 50, it's better to use the 55 structure and remove the least bound atoms?
        # Or just use FCC/HCP which might be better for non-magic numbers?
        # For rho=10 (short range), icosahedral is strong.
        # Let's return the 55 atom structure truncated to n? 
        # Truncating by distance from center is a heuristic.
        pts_55 = np.array(pts_55)
        pts_55 -= np.mean(pts_55, axis=0)
        dists = np.linalg.norm(pts_55, axis=1)
        indices = np.argsort(dists)
        return pts_55[indices[:n]]
        
    # For n > 55, we would need 147 shell.
    # Given n=75, 80, FCC is usually the winner for large n even with short range, 
    # or a distorted icosahedron.
    # Let's fallback to FCC for n > 55 in this generator to keep it simple,
    # but the caller should try FCC anyway.
    return _generate_fcc_cluster(n)

def _local_optimize(pos, rho, deadline):
    """
    Perform local optimization using L-BFGS-B.
    """
    if time.time() > deadline:
        return pos, _morse_energy(pos, rho)
        
    # Use the provided relax function if it's fast and robust?
    # The task says chem_clusters.relax is available.
    # Let's use scipy.optimize.minimize directly for control over time.
    
    def objective(x):
        p = x.reshape(-1, 3)
        return _morse_energy(p, rho)
        
    def gradient(x):
        p = x.reshape(-1, 3)
        return _morse_gradient(p, rho).flatten()
        
    try:
        res = minimize(objective, pos.flatten(), jac=gradient, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success:
            return res.x.reshape(-1, 3), res.fun
        else:
            # If it fails, return the best point found?
            # L-BFGS-B usually returns a point even if not fully converged.
            return res.x.reshape(-1, 3), res.fun
    except Exception:
        return pos, _morse_energy(pos, rho)

def construct(instance):
    """
    Find the lowest-energy arrangement of n atoms bound by Morse potentials.
    Strategy: Multi-start Basin Hopping with structured seeds (FCC, HCP, Icosahedron).
    """
    n = instance["n"]
    rho = instance.get("rho", 3)
    potential = instance["potential"]
    
    deadline = _get_global_deadline()
    # Allow 45s per instance to be safe (6 * 45 = 270s < 300s)
    instance_deadline = min(deadline, time.time() + 45.0)
    
    # 1. Generate Seeds
    seeds = []
    
    try:
        seeds.append(_generate_fcc_cluster(n))
    except:
        pass
        
    try:
        seeds.append(_generate_hcp_cluster(n))
    except:
        pass
        
    # Icosahedron is important for short range (high rho) and specific n
    if rho >= 6 or n <= 55:
        try:
            seeds.append(_generate_icosahedron(n))
        except:
            pass
            
    # Add a random compact seed
    try:
        rng = np.random.default_rng(42 + n + rho)
        # Generate random points in a sphere of radius ~ n^(1/3)
        R = n ** (1/3) * 1.5
        r = rng.uniform(0, R, size=(n, 3))
        theta = rng.uniform(0, 2*np.pi, size=(n, 1))
        phi = rng.uniform(0, np.pi, size=(n, 1))
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        rand_seed = np.column_stack((x, y, z))
        seeds.append(rand_seed)
    except:
        pass

    if not seeds:
        seeds.append(np.zeros((n, 3)) + np.random.normal(0, 0.1, (n, 3)))

    best_energy = np.inf
    best_pos = seeds[0]
    
    # 2. Basin Hopping Loop
    # We will perform a limited number of basin hopping steps.
    # Each step: local opt, then kick, then local opt.
    
    # Determine number of BH steps based on time
    # Assume local opt takes ~0.1s for n=80.
    # 45s / 0.2s = 225 steps.
    # Let's cap at 100 steps to be safe.
    
    max_bh_steps = 100
    rng = np.random.default_rng(12345 + n + rho)
    
    current_pos = None
    current_energy = np.inf
    
    for seed_idx, seed in enumerate(seeds):
        if time.time() > instance_deadline - 5.0:
            break
            
        # Local optimize the seed
        pos, energy = _local_optimize(seed, rho, instance_deadline)
        
        # Check constraint
        if not _min_dist_check(pos, 0.5):
            pos = _fix_min_dist(pos, 0.5)
            pos, energy = _local_optimize(pos, rho, instance_deadline)
            if not _min_dist_check(pos, 0.5):
                continue # Skip invalid
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
            
        # Basin Hopping from this seed
        current_pos = pos.copy()
        current_energy = energy
        
        # Adaptive kick size
        kick_size = 0.5
        
        for step in range(max_bh_steps):
            if time.time() > instance_deadline - 2.0:
                break
                
            # Kick
            noise = rng.normal(0, kick_size, size=(n, 3))
            new_pos = current_pos + noise
            
            # Fix min distance
            new_pos = _fix_min_dist(new_pos, 0.5)
            
            # Local optimize
            new_pos, new_energy = _local_optimize(new_pos, rho, instance_deadline)
            
            if not _min_dist_check(new_pos, 0.5):
                continue
                
            if new_energy < current_energy:
                current_pos = new_pos
                current_energy = new_energy
                # Accept and maybe increase kick size slightly?
                # Standard BH accepts if new < old.
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = current_pos.copy()
            else:
                # Metropolis accept? Or just reject?
                # For global min search, Metropolis helps escape.
                # dE = new_energy - current_energy
                # Accept with prob exp(-dE/T)
                # T can be adaptive.
                dE = new_energy - current_energy
                if dE < 0:
                    # Should have been caught above, but just in case
                    current_pos = new_pos
                    current_energy = new_energy
                else:
                    # Simple rejection for now to keep it deterministic-ish and fast
                    # Or accept with small prob
                    if rng.random() < np.exp(-dE / 0.5): # T=0.5
                        current_pos = new_pos
                        current_energy = new_energy
                        
            # Adapt kick size
            # If we haven't improved in a while, reduce kick?
            # Or keep it constant.
            
    return best_pos
