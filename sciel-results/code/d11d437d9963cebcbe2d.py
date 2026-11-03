"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _soft_min_logsumexp(d, beta):
    """
    Smooth surrogate for min(d) using log-sum-exp.
    Returns -log(sum(exp(-beta*d)))/beta which approximates min(d) as beta->inf.
    We want to maximize min distance, so we minimize this surrogate.
    """
    d_min = np.min(d)
    terms = np.exp(-beta * (d - d_min))
    s = np.sum(terms)
    return d_min - np.log(s) / beta


def _score(c):
    """
    Compute the actual score (largest common radius) for a configuration.
    c: (N, 3) array of centers.
    Returns the radius r.
    """
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    face_dists = 0.5 - np.abs(c)
    min_face = np.min(face_dists)
    if min_face < -1e-12:
        return 0.0
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r = min(min_pair / 2.0, min_face)
    return r


def _objective(c, beta):
    """
    Objective function to minimize: -soft_min of constraints.
    """
    c = c.reshape(N, 3)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0
    
    face_dists = 0.5 - np.abs(c)
    face_flat = face_dists.flatten()
    
    all_dists = np.concatenate([pair_dists, face_flat])
    
    r_smooth = _soft_min_logsumexp(all_dists, beta)
    
    return -r_smooth


def _generate_bcc_init():
    """
    Generate an initial configuration based on a BCC lattice.
    """
    s = 0.35
    x = np.arange(-1.0, 1.0 + s, s)
    y = np.arange(-1.0, 1.0 + s, s)
    z = np.arange(-1.0, 1.0 + s, s)
    
    pts = []
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                pts.append([x[i], y[j], z[k]])
                pts.append([x[i] + s/2, y[j] + s/2, z[k] + s/2])
    
    pts = np.array(pts)
    mask = np.all(np.abs(pts) <= 0.5, axis=1)
    pts = pts[mask]
    
    if len(pts) > N:
        idx = np.random.choice(len(pts), N, replace=False)
        pts = pts[idx]
    elif len(pts) < N:
        n_pad = N - len(pts)
        pad = np.random.uniform(-0.4, 0.4, size=(n_pad, 3))
        pts = np.vstack([pts, pad])
        
    return pts


def _dynamics_step(c, dt, damping, repulsion_strength, wall_strength):
    """
    Perform one step of velocity-Verlet dynamics with repulsive forces.
    """
    c = c.copy()
    N_pts = len(c)
    
    # Compute pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1e-10)  # avoid division by zero
    
    # Compute forces from sphere-sphere repulsion
    # Force is proportional to 1/d^2 in direction away from each other
    # F_ij = repulsion_strength * (r_target - d) / d^2 * (c_i - c_j) / d
    # We want to push apart when d < 2*r_target, but for dynamics we use:
    # F = repulsion_strength / d^2 in direction (c_i - c_j)/d
    # Actually, let's use a simpler form: F = repulsion_strength * (c_i - c_j) / d^3
    # This gives magnitude repulsion_strength / d^2
    
    # Only consider pairs that are too close (d < 2*r_min_estimate)
    # For efficiency, we can use a cutoff
    cutoff = 0.5  # only interact if within this distance
    mask = dists < cutoff
    np.fill_diagonal(mask, False)
    
    forces = np.zeros_like(c)
    for i in range(N_pts):
        for j in range(N_pts):
            if mask[i, j]:
                d = dists[i, j]
                # Force magnitude: repulsion_strength / d^2
                # Direction: (c_i - c_j) / d
                f_mag = repulsion_strength / (d * d)
                f_dir = (c[i] - c[j]) / d
                forces[i] += f_mag * f_dir
    
    # Compute wall forces (keep spheres inside [-0.5, 0.5]^3)
    for axis in range(3):
        # Left wall at -0.5
        for i in range(N_pts):
            d_left = c[i, axis] + 0.5
            if d_left < 0.1:
                f_mag = wall_strength / max(d_left, 1e-6)
                forces[i, axis] += f_mag  # push right (+x)
        
        # Right wall at +0.5
        for i in range(N_pts):
            d_right = 0.5 - c[i, axis]
            if d_right < 0.1:
                f_mag = wall_strength / max(d_right, 1e-6)
                forces[i, axis] -= f_mag  # push left (-x)
    
    return forces


def _dynamics_optimize(c_init, dt, damping, repulsion_strength, wall_strength, max_steps, deadline):
    """
    Optimize configuration using velocity-Verlet dynamics.
    """
    c = c_init.copy()
    v = np.zeros_like(c)
    N_pts = len(c)
    
    # Precompute initial forces
    F = _dynamics_step(c, dt, damping, repulsion_strength, wall_strength)
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Velocity Verlet: x(t+dt) = x(t) + v(t)*dt + 0.5*F(t)/m * dt^2
        # m = 1
        c_half = c + v * dt + 0.5 * F * dt * dt
        
        # Compute forces at new position
        F_new = _dynamics_step(c_half, dt, damping, repulsion_strength, wall_strength)
        
        # v(t+dt) = v(t) + 0.5*(F(t) + F(t+dt))/m * dt
        v_new = v + 0.5 * (F + F_new) * dt
        
        # Apply damping
        v_new *= damping
        
        c = c_half
        v = v_new
        F = F_new
        
        # Clip to valid range
        c = np.clip(c, -0.4999, 0.4999)
    
    return c


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Initializations
    # 1. BCC
    c_bcc = _generate_bcc_init()
    # 2. Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c_grid = grid[:N].copy()
    
    # 3. Random
    rng = np.random.default_rng(42)
    c_rand = rng.uniform(-0.4, 0.4, size=(N, 3))
    
    initializations = [c_bcc, c_grid, c_rand]
    
    best_c = c_bcc.copy()
    best_score = _score(best_c)
    
    # Phase 1: Multi-start L-BFGS-B optimization with beta annealing
    beta_starts = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    n_restarts = 3
    
    for beta_idx, beta in enumerate(beta_starts):
        if time.time() > deadline:
            break
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            if restart == 0:
                c_init = best_c.copy()
            elif restart == 1:
                c_init = best_c + rng.normal(0, 0.05, size=(N, 3))
            else:
                c_init = initializations[(beta_idx + restart) % len(initializations)].copy()
            
            c_init = np.clip(c_init, -0.49, 0.49)
            
            try:
                res = minimize(
                    _objective,
                    c_init.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                c_opt = res.x.reshape(N, 3)
                c_opt = np.clip(c_opt, -0.4999, 0.4999)
                
                score = _score(c_opt)
                if score > best_score:
                    best_score = score
                    best_c = c_opt.copy()
            except Exception:
                pass
        
        if best_score > 0.145:
            break
    
    # Phase 2: Dynamics-based optimization to escape local minima
    # Use dynamics with repulsive forces to "shake out" the configuration
    if time.time() < deadline - 30.0:
        # Try dynamics from current best and perturbed versions
        dynamics_configs = [
            best_c.copy(),
            best_c + rng.normal(0, 0.02, size=(N, 3)),
            best_c + rng.normal(0, 0.05, size=(N, 3)),
        ]
        
        # Parameters for dynamics
        dt = 0.01
        damping = 0.95
        repulsion_strength = 0.1
        wall_strength = 0.5
        max_steps = 500
        
        for c_dyn_init in dynamics_configs:
            if time.time() > deadline - 5.0:
                break
            
            c_dyn_init = np.clip(c_dyn_init, -0.49, 0.49)
            
            try:
                c_dyn = _dynamics_optimize(
                    c_dyn_init, dt, damping, repulsion_strength, wall_strength,
                    max_steps, deadline - 5.0
                )
                c_dyn = np.clip(c_dyn, -0.4999, 0.4999)
                
                score = _score(c_dyn)
                if score > best_score:
                    best_score = score
                    best_c = c_dyn.copy()
            except Exception:
                pass
    
    # Phase 3: Final polish with high beta L-BFGS-B
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                _objective,
                best_c.flatten(),
                args=(5000.0,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            c_polished = res.x.reshape(N, 3)
            c_polished = np.clip(c_polished, -0.49999, 0.49999)
            
            score = _score(c_polished)
            if score > best_score:
                best_score = score
                best_c = c_polished.copy()
        except Exception:
            pass
    
    return best_c
# EVOLVE-BLOCK-END
