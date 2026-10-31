"""Circle packing in unit square - 26 circles maximizing sum of radii.
Uses repulsion-based gradient descent with multiple restarts and local search."""
import numpy as np
import time
from scipy.optimize import minimize, Bounds

N = 26

# EVOLVE-BLOCK-START

def _repulsion_forces(pos, radii):
    """Compute repulsion forces on positions and radii.
    Returns (force_pos, force_radii) where force_pos has shape (N,2) and force_radii has shape (N,).
    """
    n = len(radii)
    force_pos = np.zeros((n, 2))
    force_radii = np.zeros(n)
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq)
            sum_r = radii[i] + radii[j]
            
            if dist < 1e-10:
                # Random small nudge
                force_pos[i, 0] += 0.01
                continue
            
            # Overlap penalty: want dist >= sum_r
            if dist < sum_r:
                # Strong repulsion when overlapping
                overlap = sum_r - dist
                # Force magnitude proportional to overlap
                f_mag = overlap * 10.0
                # Direction: push apart
                ux = dx / dist
                uy = dy / dist
                force_pos[i, 0] += f_mag * ux
                force_pos[i, 1] += f_mag * uy
                force_pos[j, 0] -= f_mag * ux
                force_pos[j, 1] -= f_mag * uy
                # Also encourage larger radii when not overlapping too much
                # But when overlapping, we don't directly change radii here
            else:
                # No overlap: gentle attraction to allow packing
                # Actually, we want to maximize sum of radii, so we should
                # increase radii when there's room. The gradient of sum(radii)
                # is +1 for each radius. But constraints limit this.
                pass
    
    # Boundary forces: push circles inward if they're near walls
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        
        # Left wall
        if x < r:
            force_pos[i, 0] += (r - x) * 10.0
        # Right wall
        if x > 1.0 - r:
            force_pos[i, 0] -= (x - (1.0 - r)) * 10.0
        # Bottom wall
        if y < r:
            force_pos[i, 1] += (r - y) * 10.0
        # Top wall
        if y > 1.0 - r:
            force_pos[i, 1] -= (y - (1.0 - r)) * 10.0
    
    # Gradient of objective: d(sum radii)/d(r_i) = 1
    # So we want to increase radii. But constraints (boundary, overlap) limit this.
    # The net force on radius is +1 (from objective) minus constraint violations.
    for i in range(n):
        # Boundary constraint on radius
        max_r_boundary = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        if radii[i] > max_r_boundary:
            force_radii[i] = -10.0  # Strongly decrease
        else:
            # Check overlap constraints
            can_increase = True
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j]:
                    can_increase = False
                    break
            if can_increase:
                force_radii[i] = 1.0  # Increase radius
            else:
                force_radii[i] = 0.0  # Don't change radius
    
    return force_pos, force_radii


def _gradient_descent(pos, radii, n_steps=5000, lr=0.01, deadline=None):
    """Repulsion-based gradient descent."""
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        force_pos, force_radii = _repulsion_forces(pos, radii)
        
        # Update positions
        pos += lr * force_pos
        # Update radii
        radii += lr * force_radii
        
        # Clamp
        for i in range(n):
            pos[i, 0] = np.clip(pos[i, 0], 0.001, 0.999)
            pos[i, 1] = np.clip(pos[i, 1], 0.001, 0.999)
            radii[i] = np.clip(radii[i], 0.005, 0.49)
        
        # Check validity periodically
        if step % 200 == 0:
            pos_pp, radii_pp = _fix_validity(pos, radii)
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    # Final check
    pos_pp, radii_pp = _fix_validity(pos, radii)
    score = np.sum(radii_pp)
    if score > best_score:
        best_score = score
        best_pos = pos_pp.copy()
        best_radii = radii_pp.copy()
    
    return best_pos, best_radii


def _fix_validity(pos, radii):
    """Fix violations by shrinking radii."""
    pos = pos.copy()
    radii = radii.copy()
    n = len(radii)
    
    for _ in range(500):
        changed = False
        
        # Boundary
        for i in range(n):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.999
                changed = True
        
        # Overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    scale = dist / sum_r * 0.999
                    radii[i] *= scale
                    radii[j] *= scale
                    changed = True
        
        if not changed:
            break
    
    return pos, radii


def _is_valid(pos, radii, tol=1e-7):
    """Check validity."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _initial_grid(seed=0):
    """Grid-based initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    # 5x5 grid = 25 circles, plus 1 in center
    idx = 0
    for i in range(5):
        for j in range(5):
            if idx >= N:
                break
            pos[idx] = [(i + 0.5) / 5.0 + rng.uniform(-0.01, 0.01),
                        (j + 0.5) / 5.0 + rng.uniform(-0.01, 0.01)]
            radii[idx] = 0.1
            idx += 1
    
    if idx < N:
        pos[idx] = [0.5 + rng.uniform(-0.02, 0.02), 0.5 + rng.uniform(-0.02, 0.02)]
        radii[idx] = 0.05
        idx += 1
    
    pos, radii = _fix_validity(pos, radii)
    return pos, radii


def _initial_random(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.15, 0.85, (N, 2))
    radii = rng.uniform(0.05, 0.12, N)
    pos, radii = _fix_validity(pos, radii)
    return pos, radii


def _local_search(pos, radii, deadline, n_restarts=3):
    """Local search with random perturbations."""
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Perturb
        pos_p = pos.copy()
        radii_p = radii.copy()
        for i in range(N):
            pos_p[i, 0] += np.random.uniform(-0.02, 0.02)
            pos_p[i, 1] += np.random.uniform(-0.02, 0.02)
            radii_p[i] *= np.random.uniform(0.95, 1.05)
        
        pos_p, radii_p = _fix_validity(pos_p, radii_p)
        pos_gd, radii_gd = _gradient_descent(pos_p, radii_p, n_steps=3000, lr=0.008, deadline=deadline)
        score = np.sum(radii_gd)
        
        if score > best_score:
            best_score = score
            best_pos = pos_gd.copy()
            best_radii = radii_gd.copy()
    
    return best_pos, best_radii


def _scipy_polish(pos, radii, deadline):
    """Polish with scipy minimize."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = Bounds(
        lower=np.concatenate([np.full(N, 0.01), np.full(N, 0.01), np.full(N, 0.01)]),
        upper=np.concatenate([np.full(N, 0.99), np.full(N, 0.99), np.full(N, 0.49)])
    )
    
    def objective(x):
        p = x[:2*N].reshape(N, 2)
        r = np.maximum(x[2*N:], 0.001)
        # Penalty for violations
        pen = 0.0
        for i in range(N):
            pen += max(0, r[i] - p[i, 0]) ** 2
            pen += max(0, p[i, 0] - (1 - r[i])) ** 2
            pen += max(0, r[i] - p[i, 1]) ** 2
            pen += max(0, p[i, 1] - (1 - r[i])) ** 2
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((p[i,0]-p[j,0])**2 + (p[i,1]-p[j,1])**2)
                if d < r[i] + r[j]:
                    pen += (r[i] + r[j] - d) ** 2
        return -np.sum(r) + 1000 * pen
    
    try:
        result = minimize(objective, x, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12})
        if result.fun < objective(x):
            x = result.x
    except Exception:
        pass
    
    pos_out = x[:2*N].reshape(N, 2)
    radii_out = np.maximum(x[2*N:], 0.005)
    pos_out, radii_out = _fix_validity(pos_out, radii_out)
    return pos_out, radii_out


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Multiple initializations
    inits = []
    for seed in [0, 1, 2]:
        pos, radii = _initial_grid(seed)
        inits.append((pos, radii))
    for seed in [10, 20, 30]:
        pos, radii = _initial_random(seed)
        inits.append((pos, radii))
    
    best_pos = inits[0][0].copy()
    best_radii = inits[0][1].copy()
    best_score = np.sum(best_radii)
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 15:
            break
        
        # Gradient descent
        pos_gd, rad_gd = _gradient_descent(pos_init, rad_init, n_steps=4000, lr=0.01, deadline=deadline)
        score = np.sum(rad_gd)
        if score > best_score:
            best_score = score
            best_pos = pos_gd.copy()
            best_radii = rad_gd.copy()
        
        # Local search
        if time.time() < deadline - 10:
            pos_ls, rad_ls = _local_search(pos_gd, rad_gd, deadline, n_restarts=2)
            score = np.sum(rad_ls)
            if score > best_score:
                best_score = score
                best_pos = pos_ls.copy()
                best_radii = rad_ls.copy()
    
    # Final polish
    if time.time() < deadline - 5:
        pos_pol, rad_pol = _scipy_polish(best_pos, best_radii, deadline)
        score = np.sum(rad_pol)
        if score > best_score:
            best_score = score
            best_pos = pos_pol.copy()
            best_radii = rad_pol.copy()
    
    # Final validation
    best_pos, best_radii = _fix_validity(best_pos, best_radii)
    if not _is_valid(best_pos, best_radii):
        # Fallback
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(N):
            pos[i] = [(i % 5 + 0.5) / 5.0, (i // 5 + 0.5) / 5.0]
            radii[i] = 0.095
        best_pos, best_radii = _fix_validity(pos, radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
