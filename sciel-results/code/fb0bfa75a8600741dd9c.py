"""Logarithmic energy minimization of 150 points on the unit sphere.
Uses a multi-start strategy with Velocity-Verlet, Adam, and L-BFGS-B polishing."""

import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient for points on the unit sphere.
    Energy: E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient: grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    The gradient is projected to the tangent space of the sphere.
    """
    n = pts.shape[0]
    # Compute pairwise distances squared
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist2, 1.0)
    
    # Compute energy
    # We only need the upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    energy = -math.fsum(np.log(dists).tolist())
    
    # Compute gradient
    # grad_i = sum_j (x_j - x_i) / dist_ij^2
    # Note: 1/dist2 is symmetric
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad = inv_dist2 @ pts - pts * row_sums
    # row_sums[i] = sum_j inv_dist2[i,j]
    row_sums = np.sum(inv_dist2, axis=1)
    grad = inv_dist2 @ pts
    grad -= pts * row_sums[:, np.newaxis]
    
    # Project gradient to tangent space
    # tangent_grad = grad - (grad . x_i) x_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _energy_only(pts):
    """Compute only the energy."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    return -math.fsum(np.log(dists).tolist())


def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _fibonacci_spiral(n, offset=0.0, twist=1.0):
    """Generate points using a Fibonacci spiral."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i * twist
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _golden_spiral(n, seed=0):
    """
    Generate points using a golden-ratio spiral with slight random perturbation
    to break symmetry and help escape local minima during optimization.
    """
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    # Use golden angle for better distribution
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    phi = golden_angle * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # Add small random perturbation to break exact symmetry
    pts += rng.randn(n, 3) * 0.01
    return _project_to_sphere(pts)


def _geodesic_dome(n, seed=0):
    """
    Generate points via recursive subdivision of an icosahedron (geodesic dome),
    then project to sphere. Provides a different structure than Fibonacci spiral.
    """
    rng = np.random.RandomState(seed)
    # Start with icosahedron vertices
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi_g, 0], [1, phi_g, 0], [-1, -phi_g, 0], [1, -phi_g, 0],
        [0, -1, phi_g], [0, 1, phi_g], [0, -1, -phi_g], [0, 1, -phi_g],
        [phi_g, 0, -1], [phi_g, 0, 1], [-phi_g, 0, -1], [-phi_g, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Edges of icosahedron
    edges = [(0,11),(0,5),(0,1),(0,7),(0,10),(1,5),(5,11),(11,10),(10,7),
             (7,1),(1,6),(6,10),(2,11),(2,5),(2,9),(2,4),(2,8),(3,5),
             (3,11),(3,4),(3,9),(3,6),(4,9),(5,4),(11,4),(10,7),(6,1),
             (8,4),(9,8),(8,5),(6,9),(7,2),(8,3)]
    
    # Recursively subdivide until we have enough points
    while len(verts) < n:
        new_verts = []
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            mid /= np.linalg.norm(mid)
            new_verts.append(mid)
        verts = np.vstack([verts, np.array(new_verts)])
    
    # If we have more than n, randomly select n
    if len(verts) > n:
        idx = rng.choice(len(verts), size=n, replace=False)
        verts = verts[idx]
    
    return verts


def _random_points(n, seed=42):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _adam_optimize(pts0, max_iter=3000, lr_start=0.01, lr_end=0.001,
                   beta1=0.9, beta2=0.999, eps=1e-8, deadline=None, check_interval=20):
    """
    Adam optimization on the sphere with learning rate decay.
    """
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    
    # Cosine decay for learning rate
    def lr_schedule(t):
        return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t / max_iter))

    for t in range(1, max_iter + 1):
        if deadline is not None and (t % check_interval == 0 or t == 1):
            if time.time() > deadline:
                break

        energy, grad = _energy_and_grad(pts)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update = m_hat / (np.sqrt(v_hat) + eps)
        
        lr = lr_schedule(t)
        
        # Step in tangent direction
        pts_new = pts - lr * update
        pts_new = _project_to_sphere(pts_new)

        if not np.all(np.isfinite(pts_new)):
            break

        energy_new = _energy_only(pts_new)
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts_new.copy()
            
        pts = pts_new

    return best_pts, best_energy


def _velocity_verlet_sphere(points, deadline, dt=0.005, n_steps=5000, damping=0.98, 
                            anneal_start=0.5, anneal_end=0.01):
    """
    Velocity-Verlet integration on the sphere with simulated annealing schedule.
    """
    pts = _project_to_sphere(points.copy())
    vel = np.zeros_like(pts)
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    # Initial force
    _, grad0 = _energy_and_grad(pts)
    force = -grad0
    
    # Annealing schedule parameters
    total_steps = min(n_steps, int((deadline - time.time()) / 0.01))
    total_steps = max(total_steps, 100)
    
    for step in range(total_steps):
        if time.time() > deadline - 0.5:
            break
        
        # Annealing schedule: high temperature early, low temperature late
        t_norm = step / total_steps
        if t_norm < anneal_start:
            # High temperature phase: more exploration
            current_damping = 1.0
            current_dt = dt
        elif t_norm < 0.8:
            # Medium temperature: balance exploration and exploitation
            progress = (t_norm - anneal_start) / (0.8 - anneal_start)
            current_damping = 1.0 - progress * (1.0 - damping)
            current_dt = dt * (1.0 - 0.5 * progress)
        else:
            # Low temperature: fine-tuning
            progress = (t_norm - 0.8) / 0.2
            current_damping = damping
            current_dt = dt * 0.5 * (1.0 - 0.5 * progress)
        
        # Velocity-Verlet half step
        vel_half = vel + 0.5 * current_dt * force
        # Project velocity to tangent space
        dot_v = np.sum(vel_half * pts, axis=1, keepdims=True)
        vel_half = vel_half - dot_v * pts
        
        # Position update
        pts_new = pts + current_dt * vel_half
        pts_new = _project_to_sphere(pts_new)
        
        # New force
        _, grad_new = _energy_and_grad(pts_new)
        force_new = -grad_new
        # Project force to tangent space
        dot_f = np.sum(force_new * pts_new, axis=1, keepdims=True)
        force_new = force_new - dot_f * pts_new
        
        # Velocity update with damping
        vel = (vel_half + 0.5 * current_dt * force_new) * current_damping
        dot_v2 = np.sum(vel * pts_new, axis=1, keepdims=True)
        vel = vel - dot_v2 * pts_new
        
        pts = pts_new
        force = force_new
        
        energy = _energy_only(pts)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
        # Adaptive damping or step size if energy increases significantly
        if step > 100 and energy > best_energy * 1.1:
            current_dt *= 0.9
            vel *= 0.9

    return best_pts, best_energy


def _lbfgs_polish(pts, deadline, maxiter=500):
    """
    L-BFGS-B polish using flat Cartesian coordinates with projection.
    """
    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter * 30,
                                'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _energy_only(p_res)
        if e_check < _energy_only(pts):
            return p_res
    except Exception:
        pass
    return pts


def _exp_map_tangent(pts, tangent_vecs):
    """
    Exponential map on S^2: move points along tangent vectors.
    For a point x on S^2 and tangent vector v (v . x = 0),
    the exponential map is: exp_x(v) = x*cos(|v|) + v*sin(|v|)/|v|
    """
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-15)
    cos_t = np.cos(norms)
    sin_t_over_norm = np.sin(safe_norms) / safe_norms
    # Handle zero norms
    zero_mask = (norms < 1e-15).flatten()
    if np.any(zero_mask):
        sin_t_over_norm[zero_mask] = 1.0
        cos_t[zero_mask] = 1.0
    new_pts = pts * cos_t + tangent_vecs * sin_t_over_norm
    return _project_to_sphere(new_pts)


def _log_map_tangent(pts, new_pts):
    """
    Logarithmic map on S^2: given x and y on S^2, return the tangent vector v at x
    such that exp_x(v) = y.
    v = theta / sin(theta) * (y - x*cos(theta)), where theta = arccos(x . y)
    """
    dot = np.sum(pts * new_pts, axis=1, keepdims=True)
    dot = np.clip(dot, -1.0 + 1e-15, 1.0 - 1e-15)
    theta = np.arccos(dot)
    safe_theta = np.maximum(theta, 1e-15)
    factor = theta / np.sin(safe_theta)
    v = factor * (new_pts - pts * np.cos(theta))
    # Project to tangent space to remove numerical drift
    dot_v = np.sum(v * pts, axis=1, keepdims=True)
    v = v - dot_v * pts
    return v


def _riemannian_lbfgs_polish(pts, deadline, maxiter=2000):
    """
    L-BFGS-B polish using the exponential map for proper Riemannian optimization.
    The optimizer works in tangent space coordinates; we map back to the sphere
    via the exponential map for evaluation.
    """
    def fun(tangent_flat):
        # Map from tangent space to sphere
        p = _exp_map_tangent(pts, tangent_flat.reshape(N, 3))
        e, g = _energy_and_grad(p)
        # Map gradient back to tangent space (already tangent from _energy_and_grad)
        return e, g.flatten()

    x0 = np.zeros(N * 3)
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter * 50,
                                'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 60})
        p_res = _exp_map_tangent(pts, res.x.reshape(N, 3))
        e_check = _energy_only(p_res)
        e_orig = _energy_only(pts)
        if e_check < e_orig:
            return p_res
    except Exception:
        pass
    return pts


def _final_grad_polish(pts, deadline, max_iter=500, lr_init=1e-4):
    """
    Final high-precision gradient descent polish with adaptive step size.
    Uses backtracking line search for robustness.
    """
    p = pts.copy()
    best_p = p.copy()
    best_e = _energy_only(p)
    
    lr = lr_init
    for it in range(max_iter):
        if time.time() > deadline - 0.1:
            break
        
        e, g = _energy_and_grad(p)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-14:
            break
        
        # Adaptive learning rate based on gradient norm
        lr_current = lr / (1.0 + g_norm * 10)
        
        # Backtracking line search
        alpha = lr_current
        p_new = p
        e_new = e
        for _ in range(20):
            p_try = _exp_map_tangent(p, -alpha * g)
            e_try = _energy_only(p_try)
            if e_try < e - 1e-4 * alpha * g_norm:
                p_new = p_try
                e_new = e_try
                break
            alpha *= 0.5
        
        p = p_new
        e = e_new
        
        if e < best_e:
            best_e = e
            best_p = p.copy()
        
        # Decay learning rate slowly
        lr *= 0.995

    return best_p


def _local_perturb_restart(pts, rng, perturb_scale=0.05, n_perturb=10):
    """
    Create perturbed versions of pts by randomly moving n_perturb points.
    Returns list of perturbed point arrays.
    """
    results = []
    for _ in range(5):
        p = pts.copy()
        idx = rng.choice(N, size=n_perturb, replace=False)
        noise = rng.randn(n_perturb, 3) * perturb_scale
        p[idx] += noise
        p = _project_to_sphere(p)
        results.append(p)
    return results


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit

    # Define multiple initializations
    # 1. Fibonacci spiral
    init1 = _fibonacci_spiral(N)
    # 2. Golden spiral with perturbation
    init2 = _golden_spiral(N, seed=0)
    # 3. Random points
    init3 = _random_points(N, seed=123)
    # 4. Random points different seed
    init4 = _random_points(N, seed=456)
    # 5. Perturbed Fibonacci
    rng = np.random.RandomState(999)
    init5 = _project_to_sphere(_fibonacci_spiral(N) + rng.randn(N, 3) * 0.1)
    # 6. Geodesic dome
    init6 = _geodesic_dome(N, seed=7)
    # 7. Another golden spiral variant
    init7 = _golden_spiral(N, seed=1)

    inits = [init1, init2, init3, init4, init5, init6, init7]
    
    best_pts_global = None
    best_energy_global = float('inf')

    # Time per restart
    # We have ~140s. Let's do 4 main restarts, each ~30s, and keep the best.
    # Then use remaining time for local perturbation restarts and final polish.
    
    time_per_restart = 25.0
    num_restarts = 5
    
    for i in range(num_restarts):
        current_deadline = start_time + (i + 1) * time_per_restart
        if current_deadline > deadline - 30.0: # Leave 30s for perturbation restarts + final polish
            current_deadline = deadline - 30.0
            
        pts_start = inits[i % len(inits)]
        
        # Phase 1: Velocity Verlet (fast basin finding)
        vv_deadline = current_deadline - 12.0
        if time.time() < vv_deadline:
            pts_vv, e_vv = _velocity_verlet_sphere(pts_start, vv_deadline, dt=0.005, n_steps=2500)
        else:
            pts_vv, e_vv = pts_start, _energy_only(pts_start)
            
        # Phase 2: Adam (refine)
        adam_deadline = current_deadline - 4.0
        if time.time() < adam_deadline:
            pts_adam, e_adam = _adam_optimize(pts_vv, max_iter=1200, lr_start=0.005, lr_end=0.0005, deadline=adam_deadline)
        else:
            pts_adam, e_adam = pts_vv, e_vv
            
        # Phase 3: L-BFGS-B (polish)
        if time.time() < current_deadline - 1.0:
            pts_lbfgs = _lbfgs_polish(pts_adam, current_deadline - 1.0, maxiter=300)
            e_lbfgs = _energy_only(pts_lbfgs)
        else:
            pts_lbfgs, e_lbfgs = pts_adam, e_adam
            
        if e_lbfgs < best_energy_global:
            best_energy_global = e_lbfgs
            best_pts_global = pts_lbfgs.copy()

    # Local Perturbation Restart Phase
    # Try perturbing the best solution to escape local minima
    if best_pts_global is not None and time.time() < deadline - 10.0:
        perturb_deadline = deadline - 10.0
        rng_perturb = np.random.RandomState(42)
        
        # Try several perturbations with decreasing scale
        for scale in [0.08, 0.05, 0.03]:
            if time.time() > perturb_deadline - 5.0:
                break
            
            perturbations = _local_perturb_restart(best_pts_global, rng_perturb, 
                                                   perturb_scale=scale, n_perturb=8)
            for pert_pts in perturbations:
                if time.time() > perturb_deadline - 2.0:
                    break
                
                # Quick Adam refinement
                if time.time() < perturb_deadline - 3.0:
                    pts_ref, e_ref = _adam_optimize(pert_pts, max_iter=500, 
                                                     lr_start=0.003, lr_end=0.0003,
                                                     deadline=perturb_deadline - 3.0)
                    # Quick L-BFGS polish
                    if time.time() < perturb_deadline - 1.0:
                        pts_pol = _lbfgs_polish(pts_ref, perturb_deadline - 1.0, maxiter=100)
                        e_pol = _energy_only(pts_pol)
                    else:
                        pts_pol, e_pol = pts_ref, e_ref
                    
                    if e_pol < best_energy_global:
                        best_energy_global = e_pol
                        best_pts_global = pts_pol.copy()

    # Final Polish Phase
    # Use remaining time to polish the global best with multiple passes at different tolerances
    remaining_deadline = deadline - 2.0
    
    if best_pts_global is None:
        best_pts_global = _fibonacci_spiral(N)
        best_energy_global = _energy_only(best_pts_global)
        
    # Multi-pass polishing with progressively tighter tolerances
    polish_configs = [
        {'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30, 'min_time': 5.0},
        {'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 40, 'min_time': 3.0},
        {'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50, 'min_time': 2.0},
        {'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 60, 'min_time': 1.5},
    ]
    
    for config in polish_configs:
        if time.time() > remaining_deadline - config['min_time']:
            break
        
        def fun_tight(x):
            p = _project_to_sphere(x.reshape(N, 3))
            e, g = _energy_and_grad(p)
            return e, g.flatten()
        
        x0 = best_pts_global.flatten()
        try:
            res = minimize(fun_tight, x0, method='L-BFGS-B', jac=True,
                           options={'maxiter': config['maxiter'], 
                                    'maxfun': config['maxiter'] * 50,
                                    'ftol': config['ftol'], 
                                    'gtol': config['gtol'], 
                                    'maxls': config['maxls']})
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _energy_only(p_res)
            if e_check < best_energy_global:
                best_energy_global = e_check
                best_pts_global = p_res
        except Exception:
            pass
    
    # Final Adam fine-tune with very small learning rate
    if time.time() < remaining_deadline - 1.0:
        pts_final, e_final = _adam_optimize(best_pts_global, max_iter=200, 
                                             lr_start=0.0001, lr_end=0.00001,
                                             deadline=remaining_deadline - 1.0)
        if e_final < best_energy_global:
            best_energy_global = e_final
            best_pts_global = pts_final

    # Riemannian exponential-map polish (operates directly on the sphere)
    if time.time() < remaining_deadline - 1.0:
        pts_riem = _riemannian_lbfgs_polish(best_pts_global, remaining_deadline - 1.0, maxiter=1500)
        e_riem = _energy_only(pts_riem)
        if e_riem < best_energy_global:
            best_energy_global = e_riem
            best_pts_global = pts_riem

    # Final high-precision gradient polish
    if time.time() < remaining_deadline - 0.5:
        pts_gp = _final_grad_polish(best_pts_global, remaining_deadline - 0.5, max_iter=300, lr_init=5e-5)
        e_gp = _energy_only(pts_gp)
        if e_gp < best_energy_global:
            best_energy_global = e_gp
            best_pts_global = pts_gp

    best_pts_global = _project_to_sphere(best_pts_global)

    # Validation
    if not np.all(np.isfinite(best_pts_global)):
        return _fibonacci_spiral(N)

    diff = best_pts_global[:, np.newaxis, :] - best_pts_global[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_spiral(N)

    return best_pts_global

# EVOLVE-BLOCK-END
