"""Rewritten program for covering the sphere with 130 equal caps."""
import time
import numpy as np
from scipy.optimize import minimize
import math

N = 130

# EVOLVE-BLOCK-START

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius_deg(pts):
    """Compute covering radius in degrees via convex hull facets."""
    pts = _normalize(pts)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
    except Exception:
        return 180.0

    max_rad = 0.0
    for simplex in hull.simplices:
        verts = pts[simplex]
        if len(verts) == 3:
            v0, v1, v2 = verts
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
        elif len(verts) == 4:
            v0, v1, v2 = verts[0], verts[1], verts[2]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
        else:
            continue
        
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-15:
            continue
        normal = normal / norm_len
        
        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        
        for v in verts:
            cos_ang = np.clip(np.dot(v, normal), -1.0, 1.0)
            ang = np.arccos(cos_ang)
            if ang > max_rad:
                max_rad = ang
    
    return math.degrees(max_rad)

def _fibonacci_sphere(n, seed_noise=0.0, seed=42):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if seed_noise > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, seed_noise, (n, 3))
        pts = pts + noise
        pts = _normalize(pts)
    return pts

def _soft_max_loss(P_flat, samples, lam=8.0):
    P = P_flat.reshape(-1, 3)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    Pn = P / norms

    dots = samples @ Pn.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)

    min_dists = np.min(dists, axis=1)
    nearest = np.argmin(dists, axis=1)

    loss = np.sum(np.exp(lam * min_dists))

    weights = lam * np.exp(lam * min_dists)
    dots_nearest = dots[np.arange(len(samples)), nearest]
    denom = np.sqrt(np.maximum(1e-12, 1.0 - dots_nearest**2))
    grad_factor = -weights / denom

    grad_P = np.zeros_like(Pn)
    np.add.at(grad_P, nearest, grad_factor[:, None] * samples)

    grad_dot = np.sum(grad_P * Pn, axis=1, keepdims=True)
    grad_tan = grad_P - grad_dot * Pn

    grad_flat = (grad_tan / norms).flatten()

    return loss, grad_flat

def _spectral_energy_loss(P_flat, L_max=12):
    """
    Spectral energy: minimize variance of pair correlation.
    Energy = sum_{l=1}^{L_max} (l+0.5) * |sum_i Y_l(P_i)|^2 / N
    This promotes uniformity.
    """
    P = P_flat.reshape(-1, 3)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    Pn = P / norms
    n = len(Pn)
    
    # Spherical harmonic coefficients for each l
    # Y_l^m(theta, phi) with theta=arccos(z), phi=atan2(y,x)
    # We compute sum_i Y_l^m(P_i) for each m
    # Energy = sum_l (l+0.5) * sum_m |C_lm|^2 where C_lm = (1/sqrt(N)) * sum_i Y_l^m(P_i)
    
    x, y, z = Pn[:, 0], Pn[:, 1], Pn[:, 2]
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    
    energy = 0.0
    grad_P = np.zeros_like(Pn)
    
    for l in range(1, L_max + 1):
        # Compute Y_l^m for m = -l..l
        # Use recurrence or direct formula
        # For efficiency, we compute the sum directly
        # Y_l^m(theta, phi) = N_lm * P_l^m(cos(theta)) * exp(i*m*phi)
        # where N_lm = sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!)
        
        for m in range(-l, l + 1):
            abs_m = abs(m)
            # Associated Legendre P_l^m(cos(theta))
            # For simplicity, use scipy.special
            from scipy.special import lpmv
            P_lm = lpmv(abs_m, l, np.cos(theta))
            # Normalization
            N_lm = np.sqrt((2*l + 1)/(4*np.pi) * math.factorial(l - abs_m) / math.factorial(l + abs_m))
            if abs_m > 0:
                # Condon-Shortley phase
                P_lm = P_lm * (-1)**abs_m
            
            # Complex exponential
            exp_imphi = np.exp(1j * m * phi)
            
            # Y_lm = N_lm * P_lm * exp_imphi
            Y_lm = N_lm * P_lm * exp_imphi
            
            # Sum over points
            C_lm = np.sum(Y_lm) / np.sqrt(n)
            
            # Energy contribution
            energy += (l + 0.5) * np.abs(C_lm)**2
            
            # Gradient: d/dP_i of |C_lm|^2
            # This is complex; for now, we'll use a simpler gradient
            # Actually, let's compute the gradient properly
            # d|C_lm|^2/dY_lm_i = 2*Re(conj(C_lm) * Y_lm_i / sqrt(N))
            # But Y_lm_i depends on P_i through theta_i and phi_i
            # This is getting complex; let's use a different approach
            
    # Alternative: use the fact that energy = (1/N) * sum_{i,j} K(P_i, P_j) where K is a kernel
    # For now, let's just return energy with zero gradient and use it as a heuristic
    # Actually, let's compute gradient numerically or use a simpler formulation
    
    # Simpler: minimize sum_{i != j} f(dist_ij) where f promotes uniform spacing
    # Let's use: energy = sum_{i<j} (dist_ij - target_dist)^2 where target_dist is average
    
    dots = Pn @ Pn.T
    np.fill_diagonal(dots, 2.0)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # Target: average distance for uniform distribution
    # For N points, average distance ~ sqrt(4pi/N)
    target = math.sqrt(4 * np.pi / n)
    
    # Only consider upper triangle
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu]
    
    energy = np.sum((pair_dists - target)**2)
    
    # Gradient
    # d/dP_i of (dist_ij - target)^2 = 2*(dist_ij - target) * d(dist_ij)/dP_i
    # d(dist_ij)/dP_i = d(arccos(P_i.P_j))/dP_i = -(P_j - (P_i.P_j)*P_i)/sqrt(1-(P_i.P_j)^2)
    
    grad_P = np.zeros_like(Pn)
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        dot_ij = dots[i, j]
        dist_ij = dists[i, j]
        denom = np.sqrt(max(1e-12, 1.0 - dot_ij**2))
        factor = 2 * (dist_ij - target) * (-1.0 / denom)
        tangent_i = Pn[j] - dot_ij * Pn[i]
        tangent_j = Pn[i] - dot_ij * Pn[j]
        grad_P[i] += factor * tangent_i
        grad_P[j] += factor * tangent_j
    
    grad_dot = np.sum(grad_P * Pn, axis=1, keepdims=True)
    grad_tan = grad_P - grad_dot * Pn
    grad_flat = (grad_tan / norms).flatten()
    
    return energy, grad_flat

def _combined_loss(P_flat, samples, lam=8.0, spectral_weight=0.1, L_max=8):
    """Combined soft-max covering + spectral energy loss."""
    P = P_flat.reshape(-1, 3)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    Pn = P / norms

    # Soft-max covering
    dots = samples @ Pn.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    nearest = np.argmin(dists, axis=1)
    cover_loss = np.sum(np.exp(lam * min_dists))
    
    weights = lam * np.exp(lam * min_dists)
    dots_nearest = dots[np.arange(len(samples)), nearest]
    denom = np.sqrt(np.maximum(1e-12, 1.0 - dots_nearest**2))
    grad_factor = -weights / denom
    grad_P = np.zeros_like(Pn)
    np.add.at(grad_P, nearest, grad_factor[:, None] * samples)
    
    # Spectral energy
    n = len(Pn)
    dots_all = Pn @ Pn.T
    np.fill_diagonal(dots_all, 2.0)
    dots_all = np.clip(dots_all, -1.0, 1.0)
    dists_all = np.arccos(dots_all)
    target = math.sqrt(4 * np.pi / n)
    iu = np.triu_indices(n, 1)
    pair_dists = dists_all[iu]
    spec_loss = np.sum((pair_dists - target)**2)
    
    # Spectral gradient
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        dot_ij = dots_all[i, j]
        dist_ij = dists_all[i, j]
        denom_rep = np.sqrt(max(1e-12, 1.0 - dot_ij**2))
        factor = 2 * (dist_ij - target) * (-1.0 / denom_rep)
        tangent_i = Pn[j] - dot_ij * Pn[i]
        tangent_j = Pn[i] - dot_ij * Pn[j]
        grad_P[i] += spectral_weight * factor * tangent_i
        grad_P[j] += spectral_weight * factor * tangent_j
    
    total_loss = cover_loss + spectral_weight * spec_loss
    
    grad_dot = np.sum(grad_P * Pn, axis=1, keepdims=True)
    grad_tan = grad_P - grad_dot * Pn
    grad_flat = (grad_tan / norms).flatten()
    
    return total_loss, grad_flat

def _local_search_greedy(P, samples, max_iters=300, deadline=None):
    n_samples = len(samples)
    rng = np.random.default_rng(42)
    
    def get_max_dist(P):
        dots = samples @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        nearest = np.argmin(dists, axis=1)
        return np.max(min_dists), min_dists, nearest

    current_max, min_dists, nearest = get_max_dist(P)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        worst_idx = np.argmax(min_dists)
        worst_sample = samples[worst_idx]
        p_idx = nearest[worst_idx]
        p_old = P[p_idx].copy()
        
        proj = np.dot(p_old, worst_sample)
        grad_dir = worst_sample - proj * p_old
        grad_norm = np.linalg.norm(grad_dir)
        
        if grad_norm < 1e-8:
            rand = rng.standard_normal(3)
            rand = rand - np.dot(rand, p_old) * p_old
            rand_norm = np.linalg.norm(rand)
            if rand_norm < 1e-8:
                continue
            grad_dir = rand / rand_norm
        else:
            grad_dir = grad_dir / grad_norm
        
        best_step = 0.0
        best_max = current_max
        best_p = None
        
        for step in [0.15, 0.10, 0.05, 0.02, 0.01]:
            new_p = p_old + step * grad_dir
            new_p = _normalize(new_p.reshape(1, 3))[0]
            P[p_idx] = new_p
            
            new_max, _, _ = get_max_dist(P)
            
            if new_max < best_max:
                best_max = new_max
                best_step = step
                best_p = new_p.copy()
            else:
                P[p_idx] = p_old
        
        if best_p is not None:
            P[p_idx] = best_p
            current_max = best_max
            min_dists, nearest = get_max_dist(P)[1:]
        else:
            rand = rng.standard_normal(3)
            rand = rand - np.dot(rand, p_old) * p_old
            rand_norm = np.linalg.norm(rand)
            if rand_norm > 1e-8:
                rand /= rand_norm
            P[p_idx] = _normalize((p_old + 0.05 * rand).reshape(1, 3))[0]
            min_dists, nearest = get_max_dist(P)[1:]
    
    return P, current_max

def _repulsion_local_search(P, samples, deadline):
    n = len(P)
    rng = np.random.default_rng(42)
    
    def get_max_dist(P):
        dots = samples @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        nearest = np.argmin(dists, axis=1)
        return np.max(min_dists), min_dists, nearest

    current_max, min_dists, nearest = get_max_dist(P)
    max_iters = 150
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        worst_idx = np.argmax(min_dists)
        worst_sample = samples[worst_idx]
        p_idx = nearest[worst_idx]
        p_old = P[p_idx].copy()
        
        dots_all = P @ p_old
        dots_all = np.clip(dots_all, -1.0, 1.0)
        dists_all = np.arccos(dots_all)
        dists_all[p_idx] = np.pi
        
        k_rep = 5.0
        rep_weights = np.exp(-k_rep * dists_all)
        rep_weights[p_idx] = 0.0
        
        rep_force = np.zeros(3)
        for j in range(n):
            if j == p_idx:
                continue
            proj = dots_all[j] * p_old
            tangent_dir = P[j] - proj
            tan_norm = np.linalg.norm(tangent_dir)
            if tan_norm > 1e-10:
                tangent_dir /= tan_norm
            rep_force += rep_weights[j] * tangent_dir
        
        proj_worst = np.dot(p_old, worst_sample)
        att_dir = worst_sample - proj_worst * p_old
        att_norm = np.linalg.norm(att_dir)
        if att_norm > 1e-8:
            att_dir /= att_norm
        else:
            att_dir = np.zeros(3)
        
        balance = 0.3
        combined = att_dir - balance * rep_force
        comb_norm = np.linalg.norm(combined)
        if comb_norm < 1e-10:
            combined = att_dir if att_norm > 1e-8 else rng.standard_normal(3)
            comb_norm = np.linalg.norm(combined)
        combined /= comb_norm
        
        best_step = 0.0
        best_max = current_max
        best_p = None
        
        for step in [0.12, 0.08, 0.04, 0.02, 0.01]:
            new_p = p_old + step * combined
            new_p = _normalize(new_p.reshape(1, 3))[0]
            P[p_idx] = new_p
            
            new_max, _, _ = get_max_dist(P)
            
            if new_max < best_max:
                best_max = new_max
                best_step = step
                best_p = new_p.copy()
            else:
                P[p_idx] = p_old
        
        if best_p is not None:
            P[p_idx] = best_p
            current_max = best_max
            min_dists, nearest = get_max_dist(P)[1:]
        else:
            rand = rng.standard_normal(3)
            rand = rand - np.dot(rand, p_old) * p_old
            rand_norm = np.linalg.norm(rand)
            if rand_norm > 1e-8:
                rand /= rand_norm
            P[p_idx] = _normalize((p_old + 0.03 * rand).reshape(1, 3))[0]
            min_dists, nearest = get_max_dist(P)[1:]
    
    return P, current_max

def _spectral_refine(P, samples, deadline):
    """Refine using spectral energy minimization."""
    P = P.copy()
    n = len(P)
    target = math.sqrt(4 * np.pi / n)
    rng = np.random.default_rng(123)
    
    def get_max_dist(P):
        dots = samples @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        return np.max(min_dists)
    
    current_max = get_max_dist(P)
    max_iters = 200
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute spectral gradient
        Pn = _normalize(P)
        dots_all = Pn @ Pn.T
        np.fill_diagonal(dots_all, 2.0)
        dots_all = np.clip(dots_all, -1.0, 1.0)
        dists_all = np.arccos(dots_all)
        
        grad_P = np.zeros_like(Pn)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dot_ij = dots_all[i, j]
                dist_ij = dists_all[i, j]
                denom = np.sqrt(max(1e-12, 1.0 - dot_ij**2))
                factor = 2 * (dist_ij - target) * (-1.0 / denom)
                tangent_i = Pn[j] - dot_ij * Pn[i]
                grad_P[i] += factor * tangent_i
        
        # Normalize gradient
        grad_norms = np.linalg.norm(grad_P, axis=1, keepdims=True)
        grad_norms[grad_norms < 1e-10] = 1.0
        grad_P /= grad_norms
        
        # Move each point slightly in gradient direction
        step = 0.05
        for i in range(n):
            new_p = Pn[i] + step * grad_P[i]
            new_p = _normalize(new_p.reshape(1, 3))[0]
            P[i] = new_p
        
        new_max = get_max_dist(P)
        if new_max > current_max:
            # Revert
            P = _normalize(Pn)
        else:
            current_max = new_max
    
    return P, current_max

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Initialize with Fibonacci sphere + small noise
    P = _fibonacci_sphere(N, seed_noise=0.03, seed=42)
    
    # 2. Generate samples
    S_opt = 10000
    samples_opt = _fibonacci_sphere(S_opt)
    
    # 3. First optimization with combined loss
    time_check = [False]
    def time_callback(xk):
        if time.time() > deadline - 50.0:
            time_check[0] = True
            raise ValueError("Time limit approaching")
    
    try:
        res = minimize(
            lambda x: _combined_loss(x, samples_opt, lam=8.0, spectral_weight=0.1, L_max=8),
            P.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False},
            callback=time_callback
        )
        if res.x is not None:
            P = _normalize(res.x.reshape(-1, 3))
    except Exception:
        pass
    
    # 4. Spectral refinement
    if time.time() < deadline - 40.0:
        S_fine = 20000
        samples_fine = _fibonacci_sphere(S_fine)
        P, _ = _spectral_refine(P, samples_fine, deadline - 35.0)
    
    # 5. Second optimization with pure soft-max
    if time.time() < deadline - 30.0:
        S_opt2 = 12000
        samples_opt2 = _fibonacci_sphere(S_opt2)
        
        def time_callback2(xk):
            if time.time() > deadline - 20.0:
                raise ValueError("Time limit approaching")
        
        try:
            res2 = minimize(
                lambda x: _soft_max_loss(x, samples_opt2, lam=10.0),
                P.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False},
                callback=time_callback2
            )
            if res2.x is not None:
                P = _normalize(res2.x.reshape(-1, 3))
        except Exception:
            pass
    
    # 6. Repulsion-based local search
    if time.time() < deadline - 20.0:
        S_fine2 = 25000
        samples_fine2 = _fibonacci_sphere(S_fine2)
        P, _ = _repulsion_local_search(P, samples_fine2, deadline - 15.0)
    
    # 7. Final greedy local search
    if time.time() < deadline - 8.0:
        S_fine3 = 30000
        samples_fine3 = _fibonacci_sphere(S_fine3)
        P, _ = _local_search_greedy(P, samples_fine3, max_iters=100, 
                                     deadline=deadline - 5.0)
    
    P = _normalize(P)
    
    return P

# EVOLVE-BLOCK-END
