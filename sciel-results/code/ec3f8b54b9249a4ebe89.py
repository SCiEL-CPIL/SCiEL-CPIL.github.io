"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Deadline guard
_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - time.time()


def _soft_pen(xyz, alpha):
    """Compute soft penalty for overlaps and boundary violations.
    
    xyz: shape (N, 3) with columns (x, y, r)
    Returns scalar penalty (lower is better).
    """
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    
    # Boundary penalties: circles must be inside [0,1]^2
    # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    bpen = 0.0
    bpen += np.sum(np.maximum(0, r - x)) ** 2
    bpen += np.sum(np.maximum(0, r - y)) ** 2
    bpen += np.sum(np.maximum(0, r - (1 - x))) ** 2
    bpen += np.sum(np.maximum(0, r - (1 - y))) ** 2
    
    # Overlap penalties: for each pair (i,j), distance >= r_i + r_j
    # Use soft-min / smooth approximation
    n = len(x)
    op = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            sep = r[i] + r[j]
            overlap = np.maximum(0, sep - d)
            op += overlap ** 2
    
    return bpen + op


def _objective(xyz, alpha):
    """Objective: minimize -sum(r) + alpha * penalty."""
    r = xyz[:, 2]
    pen = _soft_pen(xyz, alpha)
    return -np.sum(r) + alpha * pen


def _is_valid(xyz, tol=1e-4):
    """Check if configuration is valid (disjoint interiors, inside unit square)."""
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    n = len(x)
    
    # Boundary check
    if np.any(r < 0):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    
    # Overlap check
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < r[i] + r[j] - tol:
                return False
    return True


def _total_radius(xyz):
    return np.sum(xyz[:, 2])


def _make_grid_initial(n=N):
    """Create a grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    # Use a (side x side) grid, take first n cells
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= n:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            positions.append((cx, cy, r))
    
    # Fill remaining if grid has fewer than n (shouldn't happen with ceil)
    while len(positions) < n:
        positions.append((0.5, 0.5, 0.01))
    
    return np.array(positions[:n], dtype=np.float64)


def _make_random_initial(n=N, seed=None):
    """Create a random initial configuration."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    positions = []
    for _ in range(n):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        r = rng.uniform(0.01, 0.15)
        positions.append((x, y, r))
    
    return np.array(positions, dtype=np.float64)


def _polish_lfbsb(xyz, alpha, max_time, iters=300):
    """Polish using L-BFGS-B with bounds."""
    n = len(xyz)
    # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((1e-6, 0.5))  # r
    
    flat = xyz.flatten()
    
    def obj(flat_vec):
        vec = flat_vec.reshape(n, 3)
        # Clip to valid range
        vec[:, 0] = np.clip(vec[:, 0], 0, 1)
        vec[:, 1] = np.clip(vec[:, 1], 0, 1)
        vec[:, 2] = np.clip(vec[:, 2], 1e-6, 0.5)
        return _objective(vec, alpha)
    
    t0 = time.time()
    try:
        result = minimize(
            obj,
            flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': iters, 'maxfun': iters * 3, 'ftol': 1e-10, 'gtol': 1e-8}
        )
    except Exception:
        return xyz
    
    if time.time() - t0 > max_time:
        return xyz
    
    if result.success or result.fun < _objective(xyz, alpha):
        new_xyz = result.x.reshape(n, 3)
        new_xyz[:, 0] = np.clip(new_xyz[:, 0], 0, 1)
        new_xyz[:, 1] = np.clip(new_xyz[:, 1], 0, 1)
        new_xyz[:, 2] = np.clip(new_xyz[:, 2], 1e-6, 0.5)
        return new_xyz
    return xyz


def _polish_slqp(xyz, max_time, iters=200):
    """Polish using SLSQP with explicit constraints."""
    n = len(xyz)
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-6, 0.5))
    
    flat = xyz.flatten()
    
    def obj(flat_vec):
        vec = flat_vec.reshape(n, 3)
        vec[:, 0] = np.clip(vec[:, 0], 0, 1)
        vec[:, 1] = np.clip(vec[:, 1], 0, 1)
        vec[:, 2] = np.clip(vec[:, 2], 1e-6, 0.5)
        return -np.sum(vec[:, 2])
    
    # Inequality constraints: g(x) >= 0
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        # x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda fv, i=i: fv[i*3] - fv[i*3+2]})
        # 1 - x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda fv, i=i: 1.0 - fv[i*3] - fv[i*3+2]})
        # y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda fv, i=i: fv[i*3+1] - fv[i*3+2]})
        # 1 - y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda fv, i=i: 1.0 - fv[i*3+1] - fv[i*3+2]})
    
    # Pairwise non-overlap: dist(i,j) >= r_i + r_j
    # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    for i in range(n):
        for j in range(i + 1, n):
            def nonoverlap(fv, i=i, j=j):
                dx = fv[i*3] - fv[j*3]
                dy = fv[i*3+1] - fv[j*3+1]
                ri = fv[i*3+2]
                rj = fv[j*3+2]
                return dx*dx + dy*dy - (ri + rj)**2
            constraints.append({'type': 'ineq', 'fun': nonoverlap})
    
    t0 = time.time()
    try:
        result = minimize(
            obj,
            flat,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': iters, 'ftol': 1e-8}
        )
    except Exception:
        return xyz
    
    if time.time() - t0 > max_time:
        return xyz
    
    if result.success:
        new_xyz = result.x.reshape(n, 3)
        new_xyz[:, 0] = np.clip(new_xyz[:, 0], 0, 1)
        new_xyz[:, 1] = np.clip(new_xyz[:, 1], 0, 1)
        new_xyz[:, 2] = np.clip(new_xyz[:, 2], 1e-6, 0.5)
        if _total_radius(new_xyz) > _total_radius(xyz):
            return new_xyz
    return xyz


def _local_search(xyz, max_time, n_restarts=3):
    """Do multiple local searches from perturbed starts."""
    best = xyz
    best_score = _total_radius(xyz)
    
    n = len(xyz)
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > max_time:
            break
        
        # Perturb the best solution
        if restart == 0:
            candidate = best.copy()
        else:
            candidate = best.copy()
            # Random perturbation
            pert = rng.uniform(-0.05, 0.05, size=(n, 3))
            pert[:, 2] = rng.uniform(-0.02, 0.02, size=n)
            candidate += pert
            candidate[:, 0] = np.clip(candidate[:, 0], 0.01, 0.99)
            candidate[:, 1] = np.clip(candidate[:, 1], 0.01, 0.99)
            candidate[:, 2] = np.clip(candidate[:, 2], 0.01, 0.3)
        
        # First polish with lower alpha
        alpha = 10.0
        t0 = time.time()
        candidate = _polish_lfbsb(candidate, alpha, max_time - (time.time() - t0) if time.time() - t0 < max_time else 1.0, iters=200)
        
        # Then polish with higher alpha
        if time.time() < max_time:
            alpha = 50.0
            t0 = time.time()
            candidate = _polish_lfbsb(candidate, alpha, max_time - (time.time() - t0) if time.time() - t0 < max_time else 1.0, iters=200)
        
        # SLSQP cross-check
        if time.time() < max_time:
            candidate = _polish_slqp(candidate, max_time, iters=100)
        
        score = _total_radius(candidate)
        if score > best_score:
            best = candidate
            best_score = score
    
    return best


def _fix_violations(xyz, tol=1e-3):
    """Try to fix small violations by shrinking radii or moving circles."""
    n = len(xyz)
    xyz = xyz.copy()
    
    # Simple iterative fixing
    for _ in range(50):
        changed = False
        # Check boundaries
        for i in range(n):
            x, y, r = xyz[i]
            # If circle goes out of bounds, shrink or move
            if x - r < 0:
                xyz[i, 2] = min(xyz[i, 2], x - 1e-6)
                xyz[i, 0] = max(xyz[i, 0], xyz[i, 2] + 1e-6)
                changed = True
            if x + r > 1:
                xyz[i, 2] = min(xyz[i, 2], 1 - x - 1e-6)
                xyz[i, 0] = min(xyz[i, 0], 1 - xyz[i, 2] - 1e-6)
                changed = True
            if y - r < 0:
                xyz[i, 2] = min(xyz[i, 2], y - 1e-6)
                xyz[i, 1] = max(xyz[i, 1], xyz[i, 2] + 1e-6)
                changed = True
            if y + r > 1:
                xyz[i, 2] = min(xyz[i, 2], 1 - y - 1e-6)
                xyz[i, 1] = min(xyz[i, 1], 1 - xyz[i, 2] - 1e-6)
                changed = True
        
        # Check overlaps: shrink smaller circle
        for i in range(n):
            for j in range(i + 1, n):
                dx = xyz[i, 0] - xyz[j, 0]
                dy = xyz[i, 1] - xyz[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                needed = xyz[i, 2] + xyz[j, 2]
                if d < needed - tol:
                    # Shrink the smaller one
                    if xyz[i, 2] <= xyz[j, 2]:
                        xyz[i, 2] = max(1e-6, xyz[i, 2] - (needed - d) * 0.5)
                    else:
                        xyz[j, 2] = max(1e-6, xyz[j, 2] - (needed - d) * 0.5)
                    changed = True
        
        if not changed:
            break
    
    # Final clip
    xyz[:, 0] = np.clip(xyz[:, 0], 0, 1)
    xyz[:, 1] = np.clip(xyz[:, 1], 0, 1)
    xyz[:, 2] = np.clip(xyz[:, 2], 1e-6, 0.5)
    
    return xyz


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    
    # Start with grid initial
    xyz = _make_grid_initial(N)
    best_xyz = xyz.copy()
    best_score = _total_radius(xyz)
    
    # Phase 1: Initial polish with moderate alpha
    if time.time() < _DEADLINE:
        xyz = _polish_lfbsb(xyz, alpha=20.0, max_time=30.0, iters=300)
        score = _total_radius(xyz)
        if score > best_score:
            best_xyz = xyz.copy()
            best_score = score
    
    # Phase 2: Local search with multiple restarts
    if time.time() < _DEADLINE:
        remaining = _DEADLINE - time.time() - 20.0  # reserve 20s for final
        if remaining > 5:
            xyz = _local_search(best_xyz, max_time=time.time() + remaining, n_restarts=4)
            score = _total_radius(xyz)
            if score > best_score:
                best_xyz = xyz.copy()
                best_score = score
    
    # Phase 3: Final high-alpha polish
    if time.time() < _DEADLINE:
        remaining = _DEADLINE - time.time() - 5.0
        if remaining > 2:
            best_xyz = _polish_lfbsb(best_xyz, alpha=80.0, max_time=remaining, iters=400)
            score = _total_radius(best_xyz)
            if score > best_score:
                best_score = score
    
    # Phase 4: SLSQP cross-check if time permits
    if time.time() < _DEADLINE:
        remaining = _DEADLINE - time.time() - 3.0
        if remaining > 2:
            slqp_result = _polish_slqp(best_xyz, max_time=remaining, iters=150)
            score = _total_radius(slqp_result)
            if score > best_score:
                best_xyz = slqp_result.copy()
                best_score = score
    
    # Fix any remaining violations
    best_xyz = _fix_violations(best_xyz)
    
    # Final validity check - if invalid, fall back to grid
    if not _is_valid(best_xyz, tol=1e-4):
        grid = _make_grid_initial(N)
        if _is_valid(grid, tol=1e-4):
            best_xyz = grid
        else:
            # Last resort: very small circles
            best_xyz = np.zeros((N, 3))
            side = int(np.ceil(np.sqrt(N)))
            for i in range(N):
                row = i // side
                col = i % side
                best_xyz[i, 0] = (col + 0.5) / side
                best_xyz[i, 1] = (row + 0.5) / side
                best_xyz[i, 2] = 0.5 / side - 0.01
    
    return best_xyz
# EVOLVE-BLOCK-END
