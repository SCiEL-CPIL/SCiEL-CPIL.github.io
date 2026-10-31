"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _objective(x, penalty_weight=500.0):
    """
    Objective: minimize -sum(radii) + penalty * (smooth violations).
    x = [x_1,...,x_N, y_1,...,y_N, r_1,...,r_N]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-8)
    
    # Boundary penalty (squared)
    left_v = np.maximum(0.0, radii - pos[:, 0])
    right_v = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
    bot_v = np.maximum(0.0, radii - pos[:, 1])
    top_v = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(left_v**2) + np.sum(right_v**2) + np.sum(bot_v**2) + np.sum(top_v**2)
    
    # Overlap penalty (squared, normalized)
    dx = pos[:, 0, None] - pos[None, :, 0]
    dy = pos[:, 1, None] - pos[None, :, 1]
    d2 = dx*dx + dy*dy
    md = radii[:, None] + radii[None, :]
    md2 = md * md
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol = np.maximum(0.0, md2 - d2)
    denom = md2 + 1e-8
    overlap_pen = np.sum((viol**2 / denom)[mask])
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _is_valid(pos, radii, tol=1e-6):
    """Check strict validity."""
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[:, 0, None] - pos[None, :, 0]
    dy = pos[:, 1, None] - pos[None, :, 1]
    d2 = dx*dx + dy*dy
    md = radii[:, None] + radii[None, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(d2[mask] < (md[mask] - tol)**2):
        return False
    return True

def _post_process(pos, radii):
    """Ensure validity by shrinking radii where needed."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(500):
        valid = True
        for i in range(N):
            max_r = min(pos[i, 0], 1.0-pos[i, 0], pos[i, 1], 1.0-pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max(max_r * (1 - 1e-5), 1e-6)
                valid = False
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                sr = radii[i] + radii[j]
                if d < sr:
                    overlap = sr - d
                    margin = overlap * 0.5 + 1e-6
                    tr = radii[i] + radii[j]
                    if tr > 1e-8:
                        radii[i] = max(radii[i] - margin * radii[i] / tr, 1e-6)
                        radii[j] = max(radii[j] - margin * radii[j] / tr, 1e-6)
                        valid = False
        if valid:
            break
    return pos, radii

def _initial_greedy():
    """Greedy best-fit initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    rng = np.random.RandomState(42)
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        for _ in range(5000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            max_r = min(x, 1.0-x, y, 1.0-y)
            for j in range(i):
                d = np.sqrt((x-pos[j,0])**2 + (y-pos[j,1])**2)
                max_r = min(max_r, d - radii[j])
            if max_r > best_r:
                best_r = max_r
                best_x, best_y = x, y
        pos[i] = [best_x, best_y]
        radii[i] = max(best_r * 0.97, 0.005)
    return pos, radii

def _initial_hex():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    pts = []
    r0 = 0.1
    dy = 2 * r0 * np.sqrt(3) / 2
    for row in range(6):
        for col in range(6):
            x = 0.1 + col * 2 * r0 + (row % 2) * r0
            y = 0.1 + row * dy
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append([x, y])
    for i in range(min(N, len(pts))):
        pos[i] = pts[i]
        radii[i] = 0.095
    if N > len(pts):
        rng = np.random.RandomState(7)
        for i in range(len(pts), N):
            for _ in range(1000):
                x = rng.uniform(0.05, 0.95)
                y = rng.uniform(0.05, 0.95)
                ok = True
                for j in range(i):
                    if np.sqrt((x-pos[j,0])**2 + (y-pos[j,1])**2) < radii[j] + 0.05:
                        ok = False
                        break
                if ok:
                    pos[i] = [x, y]
                    radii[i] = 0.05
                    break
            if i >= len(pts):
                pass
    return pos, radii

def _initial_random(seed):
    """Random initial state, post-processed to validity."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.03, 0.1, N)
    return _post_process(pos, radii)

def _force_relax(pos, radii, steps=2000, t0=0.008, t1=0.0005, time_limit=None):
    """Force-directed relaxation with annealing."""
    pos = pos.copy()
    radii = radii.copy()
    rng = np.random.RandomState(123)
    for step in range(steps):
        if time_limit and time.time() > time_limit:
            break
        temp = t0 * (t1 / t0) ** (step / steps)
        F = np.zeros_like(pos)
        # Boundary forces
        for i in range(N):
            if pos[i,0] < radii[i]: F[i,0] += (radii[i] - pos[i,0]) * 30
            if pos[i,0] > 1 - radii[i]: F[i,0] -= (pos[i,0] - (1 - radii[i])) * 30
            if pos[i,1] < radii[i]: F[i,1] += (radii[i] - pos[i,1]) * 30
            if pos[i,1] > 1 - radii[i]: F[i,1] -= (pos[i,1] - (1 - radii[i])) * 30
        # Overlap forces
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                d = np.sqrt(dx*dx + dy*dy) + 1e-12
                sr = radii[i] + radii[j]
                if d < sr:
                    ov = sr - d
                    fm = ov * 15
                    wi = radii[j] / (radii[i] + radii[j] + 1e-8)
                    wj = radii[i] / (radii[i] + radii[j] + 1e-8)
                    F[i,0] += fm * dx / d * wi
                    F[i,1] += fm * dy / d * wi
                    F[j,0] -= fm * dx / d * wj
                    F[j,1] -= fm * dy / d * wj
        noise = rng.normal(0, temp, pos.shape)
        pos += F * 0.05 + noise
        pos[:, 0] = np.clip(pos[:, 0], 0.0, 1.0)
        pos[:, 1] = np.clip(pos[:, 1], 0.0, 1.0)
        pos, radii = _post_process(pos, radii)
    return pos, radii

def _optimize_from(x0, time_budget, penalty_stages=None):
    """Multi-stage L-BFGS-B optimization from x0 within time_budget seconds."""
    if penalty_stages is None:
        penalty_stages = [200.0, 1000.0, 5000.0, 20000.0, 100000.0]
    
    bounds = [(0.001, 0.999)] * (2*N) + [(0.001, 0.45)] * N
    x = x0.copy()
    
    for pw in penalty_stages:
        if time.time() > time_budget:
            break
        try:
            res = minimize(_objective, x, method='L-BFGS-B', bounds=bounds,
                          args=(pw,), options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.x is not None:
                x = res.x
        except Exception:
            pass
    return x

def _high_precision_polish(x, time_budget):
    """Final ultra-high-precision polish passes."""
    bounds = [(0.001, 0.999)] * (2*N) + [(0.001, 0.45)] * N
    x_cur = x.copy()
    
    for pw in [50000.0, 200000.0, 1000000.0]:
        if time.time() > time_budget:
            break
        try:
            res = minimize(_objective, x_cur, method='L-BFGS-B', bounds=bounds,
                          args=(pw,), options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.x is not None:
                x_cur = res.x
        except Exception:
            pass
    
    # Powell refinement
    if time.time() < time_budget:
        try:
            res = minimize(_objective, x_cur, method='Powell', bounds=bounds,
                          args=(200000.0,), options={'maxiter': 2000, 'ftol': 1e-14})
            if res.x is not None:
                x_cur = res.x
        except Exception:
            pass
    
    return x_cur

def construct():
    start = time.time()
    deadline = start + 145.0  # 5s safety margin
    
    # Generate multiple initializations
    inits = []
    pos_g, rad_g = _initial_greedy()
    inits.append((pos_g, rad_g))
    
    pos_h, rad_h = _initial_hex()
    inits.append((pos_h, rad_h))
    
    for s in [1, 2, 3, 4]:
        pos_r, rad_r = _initial_random(s)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_g.copy()
    best_radii = rad_g.copy()
    best_score = float(np.sum(best_radii))
    
    # Per-initialization time budget
    per_init_budget = min(20.0, (deadline - start) / max(len(inits), 1))
    
    for idx, (p0, r0) in enumerate(inits):
        if time.time() > deadline - 25:
            break
        x0 = np.concatenate([p0.flatten(), r0])
        init_deadline = min(deadline - 20, start + (idx + 1) * per_init_budget + 10)
        
        # Stage optimization
        x_opt = _optimize_from(x0, init_deadline, penalty_stages=[100.0, 500.0, 2000.0, 10000.0])
        
        pos_opt = x_opt[:2*N].reshape(N, 2)
        radii_opt = np.maximum(x_opt[2*N:], 1e-4)
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
        score = float(np.sum(radii_pp))
        
        if _is_valid(pos_pp, radii_pp) and score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
    
    # Force-directed relaxation on best
    if time.time() < deadline - 20:
        fa_deadline = min(deadline - 15, time.time() + 15)
        pos_fa, rad_fa = _force_relax(best_pos, best_radii, steps=3000, time_limit=fa_deadline)
        score = float(np.sum(rad_fa))
        if _is_valid(pos_fa, rad_fa) and score > best_score:
            best_score = score
            best_pos = pos_fa.copy()
            best_radii = rad_fa.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        x_final = np.concatenate([best_pos.flatten(), best_radii])
        polish_deadline = deadline - 2
        x_polished = _high_precision_polish(x_final, polish_deadline)
        pos_pol = x_polished[:2*N].reshape(N, 2)
        rad_pol = np.maximum(x_polished[2*N:], 1e-4)
        pos_pol, rad_pol = _post_process(pos_pol, rad_pol)
        score = float(np.sum(rad_pol))
        if _is_valid(pos_pol, rad_pol) and score > best_score:
            best_score = score
            best_pos = pos_pol.copy()
            best_radii = rad_pol.copy()
    
    # Final validation and fallback
    best_pos, best_radii = _post_process(best_pos, best_radii)
    if not _is_valid(best_pos, best_radii):
        # Fallback: 5x5 grid
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = _post_process(pos, radii)
    
    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
