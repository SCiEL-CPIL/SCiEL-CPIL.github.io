import time
import numpy as np
import concurrent.futures as _cf
from scipy.optimize import minimize
import sys

# Global timer and deadline management
_T0 = None
_HARD_DEADLINE = None

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
        _HARD_DEADLINE = _T0 + 290.0  # Safety margin before 300s kill
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

def construct(instance):
    global _T0, _HARD_DEADLINE
    _init_timer()
    
    n = instance["n"]
    rho = instance.get("rho", 3.0)
    
    rng = np.random.default_rng(instance.get("seed", 42) + n)
    
    start_time = time.time()
    
    # Determine time budget
    # We have 6 calls in total, 300s total.
    # If we assume equal distribution, 50s per call.
    # But earlier calls should be faster if we find the min quickly.
    # Let's set a per-call deadline.
    call_deadline = start_time + 45.0  # 45s per call is safe.
    
    # Also respect global hard deadline
    # If we are the last call, we have less time? 
    # Actually, the timer starts at the first call.
    # If this is call 1, we have 290s.
    # If this is call 6, we have 290 - (5*45) = 65s? 
    # Let's just use the global deadline as the hard stop.
    # And a soft deadline of 45s from start of this call.
    
    effective_deadline = min(call_deadline, _HARD_DEADLINE)
    
    best_E = np.inf
    best_P = None
    
    # Phase 1: Initial Seeds
    seeds = _generate_seeds(n, rng)
    
    for seed in seeds:
        if time.time() > effective_deadline:
            break
        E, P = _local_minimize(seed, rho)
        if E is not None and E < best_E:
            best_E = E
            best_P = P.copy()
            
    if best_P is None:
        # Fallback
        best_P = _generate_random(n, rng)
        best_E = _morse_energy(best_P, rho)
        
    # Phase 2: Differentiable Particle Swarm Optimization (PSO)
    # This replaces the discrete Basin Hopping with a continuous velocity-based search.
    # Particles represent clusters. Velocity is updated using the analytical gradient.
    
    n_particles = 4
    max_iters = 200
    
    # Initialize particles around the best found structure
    particles = []
    velocities = []
    
    # Particle 0 is the best found so far
    particles.append(best_P.copy())
    velocities.append(np.zeros_like(best_P))
    
    # Other particles are random perturbations of the best found structure
    for i in range(1, n_particles):
        P_i = best_P + rng.normal(0, 1.0, best_P.shape)
        E_i, P_i_opt = _local_minimize(P_i, rho)
        if E_i is not None:
            particles.append(P_i_opt)
            velocities.append(np.zeros_like(P_i_opt))
        else:
            particles.append(P_i)
            velocities.append(np.zeros_like(P_i))
            
    # Track personal bests and global best
    p_best_P = [p.copy() for p in particles]
    p_best_E = [np.inf] * n_particles
    for i in range(n_particles):
        p_best_E[i] = _morse_energy(p_best_P[i], rho)
        
    g_best_P = best_P.copy()
    g_best_E = best_E
    
    # PSO Parameters
    c1 = 1.5  # Cognitive coefficient
    c2 = 1.5  # Social coefficient
    w = 0.9   # Inertia weight
    
    for iter_idx in range(max_iters):
        if time.time() > effective_deadline - 1.0:
            break
            
        for i in range(n_particles):
            # Calculate gradient for the current particle position
            grad = _morse_gradient(particles[i], rho)
            
            # Update velocity
            # v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
            r1 = rng.random(particles[i].shape)
            r2 = rng.random(particles[i].shape)
            
            # Gradient descent term: -grad * step_size
            # We can incorporate the gradient directly into the velocity update or use it to bias the random terms.
            # Standard PSO uses position differences. Here we add a gradient flow term to help escape flat regions.
            grad_step = 0.01 * grad
            
            velocities[i] = w * velocities[i] + c1 * r1 * (p_best_P[i] - particles[i]) + c2 * r2 * (g_best_P - particles[i]) - grad_step
            
            # Update position
            particles[i] = particles[i] + velocities[i]
            
            # Local minimize to ensure we are at a local minimum
            E_new, P_new = _local_minimize(particles[i], rho)
            
            if E_new is not None:
                particles[i] = P_new
                velocities[i] *= 0.5 # Dampen velocity after local minimization
                
                # Update personal best
                if E_new < p_best_E[i]:
                    p_best_E[i] = E_new
                    p_best_P[i] = P_new.copy()
                    
                    # Update global best
                    if E_new < g_best_E:
                        g_best_E = E_new
                        g_best_P = P_new.copy()
                        best_E = E_new
                        best_P = P_new.copy()
                        
                # Reset velocity to zero after local min to avoid overshooting
                velocities[i] = np.zeros_like(P_new)
                
        # Adaptive inertia weight
        w = max(0.5, w * 0.99)

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
