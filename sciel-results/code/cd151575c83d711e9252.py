"""Direct FCC truncated octahedron construction with basin hopping and high-precision relaxation."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _lj_energy_and_grad(pos):
    """Compute total Lennard-Jones energy and gradient."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    r2_safe = np.where(r2 < 1e-12, 1e-12, r2)
    r6 = r2_safe ** 3
    r12 = r6 ** 2
    
    mask = ~np.eye(len(pos), dtype=bool)
    energy = 4.0 * np.sum(1.0 / r12[mask] - 1.0 / r6[mask])
    
    coeff = 4.0 * (-12.0 / (r2_safe ** 7) + 6.0 / (r2_safe ** 4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    e, _ = _lj_energy_and_grad(pos)
    return e

def _make_fcc_truncated_octahedron():
    """Generate the exact 38-atom FCC truncated octahedron positions."""
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if (i + j + k) % 2 == 0:
                    points.append([i, j, k])
    
    pts = np.array(points, dtype=float)
    dists = np.sum(pts ** 2, axis=1)
    sorted_idx = np.argsort(dists)
    
    # Take 38 closest points
    cluster = pts[sorted_idx[:N]].copy()
    
    # The nearest neighbor distance in this lattice is sqrt(2)
    # For LJ minimum, we want r_min = 2^(1/6)
    # So scale = 2^(1/6) / sqrt(2)
    r_min = 2.0 ** (1.0 / 6.0)
    scale = r_min / np.sqrt(2.0)
    cluster *= scale
    
    return cluster

def _relax(pos, max_iter=2000, tol=1e-12):
    """L-BFGS relaxation with tight tolerance."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()
    
    try:
        res = minimize(
            obj_grad,
            pos.ravel(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 100}
        )
        return res.x.reshape(-1, 3)
    except Exception:
        return pos

def _polish(pos, max_iter=5000, tol=1e-14):
    """High-precision polish pass."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()
    
    try:
        res = minimize(
            obj_grad,
            pos.ravel(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 200}
        )
        return res.x.reshape(-1, 3)
    except Exception:
        return pos

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure minimum separation between atoms."""
    for _ in range(100):
        min_dist = np.inf
        min_pair = None
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)
        if min_dist >= min_sep:
            break
        # Push apart
        diff = pos[min_pair[0]] - pos[min_pair[1]]
        norm = np.linalg.norm(diff)
        if norm < 1e-12:
            diff = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        push = (min_sep - min_dist) / 2.0
        pos[min_pair[0]] += push * diff / norm
        pos[min_pair[1]] -= push * diff / norm
    return pos

def _optimize_scale(pos, n_iter=50):
    """1D global search over uniform scale factor."""
    center = np.mean(pos, axis=0)
    centered = pos - center
    best_e = np.inf
    best_s = 1.0
    
    # Coarse scan
    for s in np.linspace(0.95, 1.05, 201):
        p = centered * s + center
        e, _ = _lj_energy_and_grad(p)
        if e < best_e:
            best_e = e
            best_s = s
            
    # Fine scan around best
    for _ in range(n_iter):
        span = 0.05 / (10 ** (_ // 3))
        for s in np.linspace(best_s - span, best_s + span, 201):
            if s <= 0.5:
                continue
            p = centered * s + center
            e, _ = _lj_energy_and_grad(p)
            if e < best_e:
                best_e = e
                best_s = s
                
    return centered * best_s + center, best_e

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Stage 1: Construct exact FCC truncated octahedron
    pos = _make_fcc_truncated_octahedron()
    
    # Stage 2: Initial relaxation with moderate tolerance
    try:
        pos = _relax(pos, max_iter=1000, tol=1e-10)
    except Exception:
        pass
    
    if time.time() > deadline - 25.0:
        return _ensure_min_sep(pos)
    
    # Stage 3: High-precision polish
    try:
        pos_polished = _polish(pos, max_iter=5000, tol=1e-14)
        e_orig = _lj_energy(pos)
        e_polished = _lj_energy(pos_polished)
        if e_polished < e_orig:
            pos = pos_polished
    except Exception:
        pass
    
    if time.time() > deadline - 15.0:
        return _ensure_min_sep(pos)
    
    # Stage 4: Basin Hopping with adaptive step size
    best_pos = pos.copy()
    best_energy = _lj_energy(pos)
    
    current_pos = best_pos.copy()
    current_energy = best_energy
    
    rng = np.random.RandomState(42)
    
    # Adaptive step size parameters
    step_size = 0.5
    min_step = 0.05
    max_step = 1.5
    acceptance_rate = 0.5
    n_recent = 10
    recent_accepts = []
    
    # Basin hopping loop
    while time.time() < deadline - 10.0:
        # Perturb current position
        noise = rng.normal(0, step_size, size=current_pos.shape)
        perturbed = current_pos + noise
        
        # Relax perturbed position
        try:
            relaxed = _relax(perturbed, max_iter=500, tol=1e-10)
            energy = _lj_energy(relaxed)
        except Exception:
            continue
        
        # Check separation constraint
        relaxed = _ensure_min_sep(relaxed)
        energy = _lj_energy(relaxed)
        
        # Acceptance criterion (Metropolis)
        delta_e = energy - current_energy
        T = 1.0
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            current_pos = relaxed
            current_energy = energy
            recent_accepts.append(1)
            
            # Update global best if improved
            if energy < best_energy:
                best_energy = energy
                best_pos = current_pos.copy()
        else:
            recent_accepts.append(0)
        
        # Adaptive step size
        if len(recent_accepts) >= n_recent:
            acceptance_rate = np.mean(recent_accepts[-n_recent:])
            if acceptance_rate > 0.7:
                step_size = max(min_step, step_size * 0.8)
            elif acceptance_rate < 0.3:
                step_size = min(max_step, step_size * 1.2)
            recent_accepts = recent_accepts[-n_recent:]
    
    # Stage 5: Final high-precision polish on best found
    if time.time() < deadline - 5.0:
        try:
            # Try scale optimization
            p_scaled, e_scaled = _optimize_scale(best_pos, n_iter=20)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled
            
            # Ultra-tight polish
            final = _polish(best_pos, max_iter=5000, tol=1e-15)
            e_final = _lj_energy(final)
            if e_final < best_energy:
                best_pos = final
                best_energy = e_final
        except Exception:
            pass
    
    return _ensure_min_sep(best_pos)
# EVOLVE-BLOCK-END
