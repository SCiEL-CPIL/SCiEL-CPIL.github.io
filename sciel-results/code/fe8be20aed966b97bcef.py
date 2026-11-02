import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
_DEADLINE_MARGIN = 5.0

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
    
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d_sq = np.maximum(dists_sq[iu], 1e-30)
    return -0.5 * np.sum(np.log(d_sq))

def _min_distance(pts):
    """Compute the minimum pairwise distance."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    return np.sqrt(np.min(dists_sq))

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

def _project_tangent(pts, vecs):
    """Project vectors to tangent space of the sphere at pts."""
    dots = np.sum(vecs * pts, axis=1, keepdims=True)
    return vecs - dots * pts

def _lbfgs_sphere(pts, time_limit=10.0, max_iter=200, m=15):
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
            yy = np.sum(y_k * y_k) + 1e-30
            alpha_k = np.sum(s_k * q) / yy
            rho.append(alpha_k)
            q -= alpha_k * y_k
        
        r = q
        
        # Two-loop recursion (forward)
        d = r.copy()
        for k in range(len(rho)):
            s_k = s_history[k]
            y_k = y_history[k]
            
            yy = np.sum(y_k * y_k) + 1e-30
            beta_k = np.sum(y_k * r) / yy
            d += (rho[k] - beta_k) * s_k
        
        # Ensure d is in tangent space
        dot_d = np.sum(d * current, axis=1)
        d -= dot_d[:, np.newaxis] * current
        
        # Check if direction is meaningful
        dir_norm = np.linalg.norm(d)
        if dir_norm < 1e-14:
            break
        
        # Line search with Wolfe conditions
        alpha = 0.5
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad
        best_alpha = 0.0
        
        c1 = 1e-4
        c2 = 0.9
        directional_deriv = np.sum(grad * d)
        
        # If directional derivative is positive, flip direction
        if directional_deriv > 0:
            d = -d
            directional_deriv = -directional_deriv
        
        # Normalize direction for better scaling
        d_norm = np.linalg.norm(d)
        if d_norm > 1e-15:
            d = d / d_norm
        
        armijo_satisfied = False
        for ls in range(50):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
            
            # Check Wolfe conditions
            new_dir_deriv = np.sum(new_grad * d)
            armijo_cond = new_energy <= energy + c1 * alpha * directional_deriv
            curvature_cond = new_dir_deriv >= c2 * directional_deriv
            
            if armijo_cond and curvature_cond:
                armijo_satisfied = True
                break
            
            if armijo_cond:
                # Accept if Armijo satisfied even if curvature not (weaker condition)
                armijo_satisfied = True
                break
            
            alpha *= 0.6
            if alpha < 1e-14:
                break
        
        if best_alpha == 0.0:
            # Try smaller step
            alpha_try = 0.01
            for _ in range(10):
                new_pts = _exponential_map(current, alpha_try * d)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < energy:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_grad = new_grad
                    best_alpha = alpha_try
                    break
                alpha_try *= 0.5
            
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
        
        # Check if y and s are sufficiently different
        y_norm = np.linalg.norm(y_vec)
        s_norm = np.linalg.norm(s_vec)
        sy = np.sum(s_vec * y_vec)
        if y_norm > 1e-10 and s_norm > 1e-10 and sy > 1e-12:
            s_history.append(s_vec)
            y_history.append(y_vec)
            
            if len(s_history) > m:
                s_history.pop(0)
                y_history.pop(0)
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-13:
            break
    
    return current, energy

def _md_minimize(pts, time_limit=10.0, max_iter=2000):
    """Molecular Dynamics with cooling on the sphere."""
    start = time.time()
    current = pts.copy()
    n = len(current)
    
    # Initial velocity
    rng = np.random.default_rng(42)
    velocities = rng.standard_normal((n, 3))
    velocities = _project_tangent(current, velocities)
    
    dt = 0.01
    damping = 0.95
    noise_scale = 0.5
    
    best_pts = current.copy()
    best_energy = _log_energy_fast(current)
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        # Cooling schedule
        progress = it / max_iter
        noise_scale = 0.5 * (1.0 - progress)
        dt = 0.01 * (1.0 - 0.5 * progress)
        
        energy, grad = _energy_and_grad(current)
        
        # Force is negative gradient
        force = -grad
        
        # Add noise
        noise = rng.standard_normal((n, 3)) * noise_scale
        noise = _project_tangent(current, noise)
        
        force = force + noise
        
        # Update velocity
        velocities = velocities * damping + force * dt
        velocities = _project_tangent(current, velocities)
        
        # Update position
        new_pts = current + velocities * dt
        new_pts = _normalize(new_pts)
        
        # Project velocity to new tangent space
        velocities = _project_tangent(new_pts, velocities)
        
        current = new_pts
        
        # Check energy
        e = _log_energy_fast(current)
        if e < best_energy:
            best_energy = e
            best_pts = current.copy()
            
        # Convergence check
        if np.linalg.norm(grad) < 1e-8 and noise_scale < 0.01:
             break

    return best_pts, best_energy

def _basin_hopping(pts, time_limit=30.0, max_restarts=15):
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
    decay = 0.82
    
    for rh in range(max_restarts):
        time_left = time_limit - (time.time() - start)
        if time_left < 6:
            break
        
        # Perturb with adaptive scale
        rng = np.random.default_rng(rh + 42)
        # Use a mix of scales for diversity
        scale = 0.25 * (0.75 ** rh) * (1.0 + 0.4 * math.sin(rh * 1.7))
        perturbations = rng.standard_normal((N, 3)) * scale
        # Project to tangent space
        dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
        perturbations -= dot_prods * current
        
        new_pts = _exponential_map(current, perturbations)
        new_pts = _normalize(new_pts)
        
        # Optimize the perturbed point
        opt_time = min(10, time_left - 4)
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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Initializations
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    
    # Random inits
    for s in range(4):
        rng = np.random.default_rng(s)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)
        
    best_pts = None
    best_energy = float('inf')
    
    # Process each initialization
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 25:
            break
        
        time_per_restart = 18.0
        if idx == 0:
            time_per_restart = 22.0
        
        try:
            # Use MD to find a good basin
            md_time = time_per_restart * 0.5
            md_pts, md_energy = _md_minimize(init_pts, time_limit=md_time)
            
            # Polish with L-BFGS
            lbfgs_time = time_per_restart - md_time - 2
            if lbfgs_time > 2:
                opt_pts, opt_energy = _lbfgs_sphere(md_pts, time_limit=lbfgs_time)
            else:
                opt_pts, opt_energy = md_pts, md_energy
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
        except Exception:
            pass
            
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _log_energy_fast(best_pts)
        
    # Final Polish with L-BFGS
    if time.time() < deadline - 5:
        polish_time = deadline - time.time() - 2
        try:
            final_pts, final_energy = _lbfgs_sphere(best_pts, time_limit=polish_time, max_iter=500, m=15)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
            
    return _normalize(best_pts)
