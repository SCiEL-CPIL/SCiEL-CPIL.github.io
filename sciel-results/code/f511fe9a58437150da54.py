import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
_DEADLINE_MARGIN = 5.0

def _log_energy_fsum(pts):
    """Exact log energy using math.fsum (for final validation)."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _energy_and_grad(pts):
    """Vectorized energy and gradient (projected to tangent space)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # Energy: sum over i<j of log(1/d_ij) = -sum_{i<j} log(d_ij)
    # d_ij = sqrt(dists_sq[i,j])
    # log(1/d_ij) = -0.5 * log(dists_sq[i,j])
    iu = np.triu_indices(n, k=1)
    d_sq = dists_sq[iu]
    d_sq = np.maximum(d_sq, 1e-30)
    energy = -0.5 * np.sum(np.log(d_sq))
    
    # Gradient of energy w.r.t. pts[i]:
    # E = sum_{i<j} -log(d_ij) = sum_{i<j} -0.5 log(dists_sq_ij)
    # dE/d(pts[i]) = sum_{j != i} -0.5 * (1/dists_sq_ij) * d(dists_sq_ij)/d(pts[i])
    # d(dists_sq_ij)/d(pts[i]) = 2*(pts[i] - pts[j])
    # So dE/d(pts[i]) = sum_{j != i} -0.5 * (1/dists_sq_ij) * 2*(pts[i] - pts[j])
    #                   = sum_{j != i} -(pts[i] - pts[j]) / dists_sq_ij
    #                   = sum_{j != i} (pts[j] - pts[i]) / dists_sq_ij
    #                   = (sum_j inv_d2[i,j] * pts[j]) - pts[i] * sum_j inv_d2[i,j]
    
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = weighted_sum_pts - pts * sum_inv_d2[:, np.newaxis]
    
    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return energy, grad

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs

def _lbfgs_sphere(pts, time_limit=10.0, max_iter=200, m=10):
    """L-BFGS on the sphere using exponential maps and parallel transport."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    # L-BFGS history
    s_history = []
    y_history = []
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        # Compute search direction using L-BFGS two-loop recursion
        q = grad.copy()
        rho = []
        
        # Two-loop recursion (backward)
        for k in range(len(s_history) - 1, -1, -1):
            s_k = s_history[k]
            y_k = y_history[k]
            
            sy = np.sum(s_k * y_k)
            yy = np.sum(y_k * y_k)
            if yy < 1e-15:
                continue
            alpha_k = np.sum(s_k * q) / yy
            rho.append(alpha_k)
            q -= alpha_k * y_k
        
        r = q
        
        # Two-loop recursion (forward)
        d = r.copy()
        for k in range(len(rho)):
            s_k = s_history[k]
            y_k = y_history[k]
            
            yy = np.sum(y_k * y_k)
            if yy < 1e-15:
                continue
            beta_k = np.sum(y_k * r) / yy
            d += (rho[k] - beta_k) * s_k
        
        # Ensure d is in tangent space
        dot_d = np.sum(d * current, axis=1)
        d -= dot_d[:, np.newaxis] * current
        
        # Check if direction is meaningful
        dir_norm = np.linalg.norm(d)
        if dir_norm < 1e-14:
            break
        
        # Line search (Armijo backtracking)
        alpha = 1.0
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad
        best_alpha = 0.0
        
        c1 = 1e-4
        directional_deriv = np.sum(grad * d)
        
        # If directional derivative is positive, flip direction
        if directional_deriv > 0:
            d = -d
            directional_deriv = -directional_deriv
        
        for ls in range(25):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
            
            # Armijo condition
            if new_energy <= energy + c1 * alpha * directional_deriv:
                alpha *= 1.5  # try larger step
                continue
            
            break
        
        if best_alpha == 0.0:
            # Try smaller steps
            for alpha_try in [0.5, 0.25, 0.1, 0.05, 0.01]:
                new_pts = _exponential_map(current, alpha_try * d)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < energy:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_grad = new_grad
                    best_alpha = alpha_try
                    break
            
            if best_alpha == 0.0:
                break
        
        # Update L-BFGS history
        s_vec = best_pts - current
        y_vec = best_grad - grad
        
        # Project to tangent space
        dot_s = np.sum(s_vec * current, axis=1)
        s_vec -= dot_s[:, np.newaxis] * current
        
        dot_y = np.sum(y_vec * best_pts, axis=1)
        y_vec -= dot_y[:, np.newaxis] * best_pts
        
        s_history.append(s_vec)
        y_history.append(y_vec)
        
        if len(s_history) > m:
            s_history.pop(0)
            y_history.pop(0)
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-12:
            break
    
    return current, energy

def _basin_hopping(pts, time_limit=30.0, max_restarts=8):
    """Perform local optimization followed by basin hopping."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Initial optimization
    time_left = time_limit - (time.time() - start)
    if time_left > 5:
        current, energy = _lbfgs_sphere(current, time_limit=min(15, time_left - 2))
    
    best_pts = current.copy()
    best_energy = energy
    
    # Basin Hopping Loop
    temp = 0.8
    decay = 0.80
    
    for rh in range(max_restarts):
        time_left = time_limit - (time.time() - start)
        if time_left < 5:
            break
        
        # Perturb with adaptive scale
        rng = np.random.default_rng(rh * 7 + 13)
        base_scale = 0.15 * (0.75 ** rh)
        scale = base_scale * (1.0 + 0.2 * math.sin(rh * 1.1 + 0.5))
        perturbations = rng.standard_normal((N, 3)) * scale
        # Project to tangent space
        dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
        perturbations -= dot_prods * current
        
        new_pts = _exponential_map(current, perturbations)
        new_pts = _normalize(new_pts)
        
        # Optimize the perturbed point
        opt_time = min(12, time_left - 3)
        if opt_time < 3:
            break
        new_opt_pts, new_energy = _lbfgs_sphere(new_pts, time_limit=opt_time)
        
        # Metropolis Criterion
        delta_e = new_energy - energy
        accept = delta_e < 0
        if not accept and temp > 1e-6:
            try:
                accept = np.random.rand() < math.exp(-delta_e / temp)
            except OverflowError:
                accept = False
        
        if accept:
            current = new_opt_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = current.copy()
        
        temp *= decay
    
    return best_pts, best_energy

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _halton_sphere(n, base_start=2):
    """Generate Halton sequence points on unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    pts = []
    for i in range(1, n + 1):
        u1 = halton(i, base_start)
        u2 = halton(i, base_start + 1)
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        pts.append([x, y, z])
    
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _farthest_point_sampling(n, seed=0):
    """Generate points using farthest point sampling for good initial spread."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((n, 3))
    
    # Start with a random point
    pts[0] = rng.standard_normal(3)
    pts[0] /= np.linalg.norm(pts[0])
    
    for i in range(1, n):
        # Sample many random candidates and pick the one farthest from all existing points
        best_point = None
        best_min_dist = -1
        
        # Sample 200 random candidates
        n_cand = 200
        candidates = rng.standard_normal((n_cand, 3))
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # Vectorized: compute min distance from each candidate to each placed point
        # candidates: (n_cand, 3), pts[:i]: (i, 3)
        diffs = candidates[:, np.newaxis, :] - pts[:i][np.newaxis, :, :]
        dists_sq = np.sum(diffs * diffs, axis=2)  # (n_cand, i)
        min_dists = np.min(dists_sq, axis=1)  # (n_cand,)
        
        idx = np.argmax(min_dists)
        best_min_dist = min_dists[idx]
        best_point = candidates[idx]
        
        pts[i] = best_point
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initializations
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    
    # Farthest point sampling for better initial spread
    inits.append(_farthest_point_sampling(N, seed=0))
    inits.append(_farthest_point_sampling(N, seed=1))
    inits.append(_farthest_point_sampling(N, seed=2))
    
    # Random inits
    for s in range(3):
        rng = np.random.default_rng(s * 31 + 7)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)
        
    # Perturbed Fibonacci
    for s in range(3):
        rng = np.random.default_rng(s * 17 + 100)
        pts = _fibonacci_sphere(N) + rng.standard_normal((N, 3)) * 0.12
        pts = _normalize(pts)
        inits.append(pts)
        
    best_pts = None
    best_energy = float('inf')
    
    # Process each initialization
    for idx, init_pts in enumerate(inits):
        time_left = deadline - time.time()
        if time_left < 20:
            break
        
        time_per_restart = min(16.0, time_left - 15.0)
        if idx == 0:
            time_per_restart = min(20.0, time_left - 15.0)
        
        try:
            opt_pts, opt_energy = _basin_hopping(init_pts, time_limit=time_per_restart)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
        except Exception:
            pass
            
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _log_energy_fast(best_pts)
        
    # Final Polish with tighter tolerance and more iterations
    time_left = deadline - time.time()
    if time_left > 5:
        polish_time = time_left - 2
        try:
            final_pts, final_energy = _lbfgs_sphere(best_pts, time_limit=polish_time, max_iter=500, m=15)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
        
    # Second polish pass if time remains
    time_left = deadline - time.time()
    if time_left > 3:
        try:
            final_pts2, final_energy2 = _lbfgs_sphere(best_pts, time_limit=time_left - 1, max_iter=200, m=10)
            if final_energy2 < best_energy:
                best_energy = final_energy2
                best_pts = final_pts2
        except Exception:
            pass
            
    return _normalize(best_pts)
