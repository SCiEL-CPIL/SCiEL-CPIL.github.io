import time
import numpy as np
import concurrent.futures as _cf
from scipy.optimize import minimize
import sys

# Global timer and deadline management
_T0 = None
_HARD_DEADLINE = None
_CALL_COUNTER = 0

def _init_timer():
    global _T0, _HARD_DEADLINE
    if _T0 is None:
        _T0 = time.time()
        # Contract: six calls together < 300 s. 
        # We assume the evaluator calls construct 6 times.
        # We will set a hard deadline for the ENTIRE session.
        # Since we don't know exactly when the 1st call is relative to the others,
        # we assume the first call starts at T=0 and the 6th finishes by T=300.
        # A safe deadline for the *last* call is 300s from the start.
        # However, each call should ideally be fast. 
        # Let's give each call a budget, but also track global time.
        _HARD_DEADLINE = _T0 + 288.0  # Safety margin before 300s kill
    return _T0

def _morse_energy(P, rho):
    """
    Compute exact Morse energy.
    E = sum_{i<j} exp(rho(1-r_ij)) * (exp(rho(1-r_ij)) - 2)
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if n < 2:
        return 0.0
    
    # Calculate all pair distances
    # D_ij = P_i - P_j
    D = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    r = np.linalg.norm(D, axis=2)
    
    # Mask diagonal (self-interaction)
    np.fill_diagonal(r, np.inf)
    
    # Calculate Morse potential for each pair
    # t = exp(rho * (1 - r))
    # E_pair = t * (t - 2)
    t = np.exp(rho * (1.0 - r))
    E_pair = t * (t - 2.0)
    
    # Sum over upper triangle (i < j)
    # np.triu_indices(n, k=1) gives indices for i < j
    iu, ju = np.triu_indices(n, k=1)
    return float(np.sum(E_pair[iu, ju]))

def _morse_gradient(P, rho):
    """
    Compute gradient of Morse energy.
    dE/dr_ij = rho * t * (2t - 2)
    grad_i = - sum_{j != i} (dE/dr_ij) * (P_i - P_j) / r_ij
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if n < 2:
        return np.zeros_like(P)
        
    D = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    r = np.linalg.norm(D, axis=2)
    np.fill_diagonal(r, np.inf)
    
    t = np.exp(rho * (1.0 - r))
    # dE/dr = rho * t * (2*t - 2)
    dEdr = rho * t * (2.0 * t - 2.0)
    
    # Gradient contribution from pair (i, j) to atom i:
    # - (dEdr_ij) * (P_i - P_j) / r_ij
    # We need to sum this over all j for each i.
    
    # inv_r is 1/r, 0 on diagonal
    inv_r = np.zeros_like(r)
    mask = r < np.inf
    inv_r[mask] = 1.0 / r[mask]
    
    # W_ij = - (dEdr_ij) / r_ij
    # Note: dEdr is symmetric, r is symmetric.
    W = -dEdr * inv_r
    
    # grad_i = sum_j W_ij * (P_i - P_j)
    # grad_i = sum_j W_ij * P_i - sum_j W_ij * P_j
    # grad_i = (sum_j W_ij) * P_i - (W @ P)_i
    
    row_sum = np.sum(W, axis=1)
    grad = row_sum[:, np.newaxis] * P - W @ P
    
    return grad

def _enforce_min_dist(P, min_d=0.5):
    """
    Nudge atoms to ensure minimum distance.
    """
    P = np.array(P, dtype=float, copy=True)
    n = P.shape[0]
    if n < 2:
        return P
        
    for _ in range(100):
        d = np.linalg.norm(P[:, np.newaxis, :] - P[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        bad_pairs = np.argwhere(d < min_d)
        if len(bad_pairs) == 0:
            break
            
        for i, j in bad_pairs:
            if i == j: continue
            gap = min_d - d[i, j] + 0.01
            v = P[i] - P[j]
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-8:
                # Random direction
                v = np.random.normal(size=3)
                norm_v = np.linalg.norm(v)
            v = v / norm_v
            P[i] += 0.5 * gap * v
            P[j] -= 0.5 * gap * v
    return P

def _local_minimize(P0, rho, max_iter=200):
    """
    Local minimization using L-BFGS-B.
    Returns (energy, positions) or (None, None) on failure.
    """
    n = P0.shape[0]
    
    def fun(x):
        P = x.reshape(n, 3)
        return _morse_energy(P, rho)
        
    def jac(x):
        P = x.reshape(n, 3)
        return _morse_gradient(P, rho).ravel()
        
    try:
        res = minimize(
            fun, P0.ravel(), 
            jac=jac, 
            method='L-BFGS-B', 
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        P_opt = res.x.reshape(n, 3)
        E_opt = _morse_energy(P_opt, rho)
        return E_opt, P_opt
    except Exception:
        return None, None

def _generate_icosahedron(n, rng):
    """
    Generate an icosahedral-like structure.
    For n=13, it's the perfect icosahedron.
    For larger n, we add shells or jitter.
    """
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices (radius 1)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # If n <= 13, we just use the 13 atoms (center + 12 verts)
    if n <= 13:
        P = np.vstack([np.zeros(3), verts[:n-1]])
        # If n < 13, we have to pick a subset? No, global min for n<13 is different.
        # But this is just a seed. Let's assume n>=13 for this generator.
        # If n < 13, we return random or smaller structure.
        if n < 13:
            return _generate_random(n, rng)
        return P
        
    # For n > 13, add atoms in a 2nd shell or jitter the 1st
    # Simple strategy: place remaining atoms on a larger icosahedron shell
    P = [np.zeros(3)]
    P.extend(list(verts)) # 13 atoms
    
    remaining = n - 13
    if remaining > 0:
        # Scale up for second shell
        scale = 1.8 
        shell2 = verts * scale
        # Add jitter to break symmetry
        shell2 = shell2 + rng.normal(0, 0.1, shell2.shape)
        
        # If remaining > 12, we need more. Let's just jitter the existing ones?
        # No, let's create a "fuzzy" icosahedron.
        # A better approach for general n: 
        # Start with icosahedron, then add atoms one by one at the most "open" site? 
        # That's complex. 
        # Alternative: Generate a random point cloud, then project onto icosahedral symmetry? 
        # Let's stick to a simple "Jittered Icosahedron" for seeds.
        
        # We will just add random points near the surface.
        for _ in range(remaining):
            # Pick a random face or vertex direction
            idx = rng.integers(0, 12)
            dir_vec = verts[idx]
            # Place at radius 1.5-2.0
            r = 1.5 + rng.random() * 0.5
            pos = dir_vec * r + rng.normal(0, 0.2, 3)
            P.append(pos)
            
    return np.array(P)

def _generate_decahedron(n, rng):
    """
    Generate a decahedral (Mackay) structure.
    """
    # Mackay decahedron: 5-fold symmetry
    # This is harder to generate programmatically from scratch quickly.
    # We will use a "twinned FCC" approximation.
    # Or simply use a distorted FCC.
    # For now, let's use a random FCC cluster which is a good seed for decahedron relaxation.
    return _generate_fcc(n, rng)

def _generate_fcc(n, rng, jitter=0.1):
    """
    Generate an FCC cluster fragment.
    """
    # Generate points in a cube
    # Number of unit cells
    # n ~ 4/3 * pi * R^3 * density
    # FCC density is high.
    # Let's generate a block of FCC lattice points and cut a sphere.
    
    a = 1.0 # lattice constant
    # Radius to encompass n points
    # Approx volume V = n / (1/vol_per_atom). FCC has 4 atoms per cell of side a.
    # Vol per atom = a^3 / 4.
    # R = (3 * n / (4 * pi * (4/a^3)))^(1/3)
    R = (3.0 * n * a**3 / (16.0 * np.pi)) ** (1.0/3.0) * 1.2 # padding
    
    pts = []
    # Grid range
    max_idx = int(np.ceil(R / a)) + 2
    
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC lattice points
                # Corners
                p1 = np.array([i*a, j*a, k*a])
                # Face centers
                p2 = np.array([(i+0.5)*a, (j+0.5)*a, k*a])
                p3 = np.array([(i+0.5)*a, j*a, (k+0.5)*a])
                p4 = np.array([i*a, (j+0.5)*a, (k+0.5)*a])
                
                for p in [p1, p2, p3, p4]:
                    if np.linalg.norm(p) <= R:
                        pts.append(p)
                        
    # If we have too many, take nearest to center
    if len(pts) > n:
        pts_arr = np.array(pts)
        dists = np.linalg.norm(pts_arr, axis=1)
        idxs = np.argsort(dists)[:n]
        pts = pts_arr[idxs]
    elif len(pts) < n:
        # Add random points
        for _ in range(n - len(pts)):
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) <= R:
                pts.append(p)
            else:
                pts.append(np.zeros(3)) # Fallback
                
    pts_arr = np.array(pts)
    pts_arr += rng.normal(0, jitter, pts_arr.shape)
    return pts_arr

def _generate_random(n, rng):
    """
    Generate a random spherical cluster.
    """
    # Random points in a sphere
    R = (n / (4.0/3.0 * np.pi)) ** (1/3.0) * 1.5
    pts = []
    for _ in range(n):
        while True:
            p = rng.uniform(-1, 1, 3)
            if np.linalg.norm(p) <= 1.0:
                pts.append(p * R)
                break
    return np.array(pts)

def _generate_seeds(n, rng):
    """
    Generate a diverse set of seeds.
    """
    seeds = []
    
    # 1. Icosahedral seed (strong for n < 50)
    seeds.append(_generate_icosahedron(n, rng))
    
    # 2. FCC seed (strong for n > 50)
    seeds.append(_generate_fcc(n, rng, jitter=0.05))
    seeds.append(_generate_fcc(n, rng, jitter=0.2))
    
    # 3. Random spherical
    seeds.append(_generate_random(n, rng))
    
    # 4. Distorted Icosahedron
    if n >= 13:
        ico = _generate_icosahedron(n, rng)
        ico += rng.normal(0, 0.3, ico.shape)
        seeds.append(ico)
        
    return seeds

def _basin_hopping_worker(seed_P, rho, deadline, worker_id):
    """
    Run a single basin hopping chain.
    Returns (energy, positions)
    """
    current_P = seed_P.copy()
    current_E = _morse_energy(current_P, rho)
    best_E = current_E
    best_P = current_P.copy()
    
    rng = np.random.default_rng(42 + worker_id * 1000)
    kick_scale = 0.5
    no_improve_count = 0
    
    # Initial local minimization
    E, P = _local_minimize(current_P, rho)
    if E is not None and E < best_E:
        best_E = E
        best_P = P.copy()
        current_P = P.copy()
        current_E = E

    while time.time() < deadline:
        # Generate a kick
        kick = rng.normal(0, kick_scale, current_P.shape)
        perturbed_P = current_P + kick
        
        # Local minimize
        E, P = _local_minimize(perturbed_P, rho)
        
        if E is None:
            continue
            
        if E < best_E:
            best_E = E
            best_P = P.copy()
            current_P = P.copy()
            current_E = E
            # Reduce kick scale slightly on improvement? Or keep it?
            # Usually keep or reduce. Let's reduce to refine.
            kick_scale = max(0.1, kick_scale * 0.9)
            no_improve_count = 0
        else:
            # If not improving, accept with Metropolis? 
            # For global min search, we just want the best. 
            # We can use a larger kick if stuck.
            no_improve_count += 1
            if no_improve_count > 10:
                # Increase kick scale to escape basin
                kick_scale = min(2.0, kick_scale * 1.2)
                no_improve_count = 0
            # We can also reset current_P to best_P to stay near the best
            # Or stay at current_P to explore further?
            # Standard BH: Move to perturbed point if accepted (Metropolis).
            # Here we just track global best. We will reset current to best
            # if we are stuck for a long time.
            if no_improve_count > 50:
                current_P = best_P.copy()
                current_E = best_E
                no_improve_count = 0
                kick_scale = 0.5
                
    return best_E, best_P

def construct(instance):
    global _T0, _HARD_DEADLINE, _CALL_COUNTER
    _init_timer()
    
    n = instance["n"]
    rho = instance.get("rho", 3.0)
    
    _CALL_COUNTER += 1
    start_time = time.time()
    
    # Determine time budget
    # We have 6 calls in total, 300s total.
    # Each call gets approx 45s.
    # We will use 40s for the search to be safe.
    call_deadline = start_time + 40.0
    
    # Also respect global hard deadline
    effective_deadline = min(call_deadline, _HARD_DEADLINE)
    
    # Generate diverse seeds
    base_rng = np.random.default_rng(42 + n)
    seeds = _generate_seeds(n, base_rng)
    
    # Ensure we have at least 4 seeds for 4 workers
    while len(seeds) < 4:
        seeds.append(_generate_random(n, base_rng))
    
    # Select 4 best/diverse seeds
    # Just take the first 4 for simplicity, as they are generated to be diverse
    active_seeds = seeds[:4]
    
    best_E = np.inf
    best_P = None
    
    # Run parallel basin hopping
    with _cf.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, seed in enumerate(active_seeds):
            # Each worker runs until the deadline
            # We pass the same deadline to all
            futures.append(executor.submit(_basin_hopping_worker, seed, rho, effective_deadline, i))
        
        for future in futures:
            try:
                E, P = future.result()
                if E is not None and E < best_E:
                    best_E = E
                    best_P = P.copy()
            except Exception as e:
                pass # Ignore errors in workers
    
    # Fallback if all workers failed
    if best_P is None:
        best_P = _generate_random(n, base_rng)
        best_E = _morse_energy(best_P, rho)
        
    # Final Polish
    if time.time() < _HARD_DEADLINE - 0.5 and best_P is not None:
        # One more high-precision local minimize
        E, P = _local_minimize(best_P, rho, max_iter=500)
        if E is not None and E < best_E:
            best_E = E
            best_P = P.copy()
            
    # Enforce minimum distance
    best_P = _enforce_min_dist(best_P, 0.5)
    
    return best_P
