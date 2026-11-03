"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-12

def _elapsed(t_start):
    return time.time() - t_start

def _remaining(t_start, deadline):
    return max(0.0, deadline - _elapsed(t_start))

def _softplus(x, k=100.0):
    """Smooth approximation of max(0, x)."""
    return np.where(x > 0, x + np.log1p(np.exp(-k * x)) / k, np.log1p(np.exp(k * x)) / k)

def _validate(pos, r, tol=1e-6):
    """Check validity: in square, no overlaps."""
    n = len(r)
    for i in range(n):
        if r[i] < -tol:
            return False
        if pos[i, 0] < r[i] - tol or pos[i, 0] > 1.0 - r[i] + tol:
            return False
        if pos[i, 1] < r[i] - tol or pos[i, 1] > 1.0 - r[i] + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            if d < r[i] + r[j] - tol:
                return False
    return True

def _fix_boundaries(pos, r):
    """Project positions to satisfy boundary constraints given radii."""
    for i in range(len(r)):
        ri = max(0.0, r[i])
        pos[i, 0] = np.clip(pos[i, 0], ri, 1.0 - ri)
        pos[i, 1] = np.clip(pos[i, 1], ri, 1.0 - ri)
    return pos, r

def _get_contact_edges(pos, threshold_factor=1.05):
    """Get edges from Delaunay triangulation of circle centers."""
    n = len(pos)
    if n < 3:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _penalty_objective(p, n, edges, penalty_weight, softplus_k=100.0):
    """
    Objective: -sum(r) + penalty * sum(violations)
    Variables: [x_0, y_0, ..., x_{n-1}, y_{n-1}, r_0, ..., r_{n-1}]
    """
    pos = p[:2*n].reshape(n, 2)
    r = p[2*n:]
    
    # Objective: maximize sum of radii
    obj = -np.sum(r)
    
    # Penalty for boundary violations
    boundary_viol = 0.0
    for i in range(n):
        ri = r[i]
        # x_i < r_i or x_i > 1-r_i
        boundary_viol += max(0, ri - pos[i, 0]) + max(0, pos[i, 0] + ri - 1.0)
        # y_i < r_i or y_i > 1-r_i
        boundary_viol += max(0, ri - pos[i, 1]) + max(0, pos[i, 1] + ri - 1.0)
        # r_i < 0
        boundary_viol += max(0, -ri)
    
    # Penalty for overlap violations (only for contact edges)
    overlap_viol = 0.0
    for (i, j) in edges:
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        d = np.hypot(dx, dy)
        overlap_viol += max(0, r[i] + r[j] - d)
    
    # Also check all pairs for safety (cheap for n=26)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in set(edges) or (j, i) in set(edges):
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.hypot(dx, dy)
            overlap_viol += max(0, r[i] + r[j] - d)
    
    return obj + penalty_weight * (boundary_viol + overlap_viol)

def _penalty_grad(p, n, edges, penalty_weight):
    """Gradient of penalty objective."""
    pos = p[:2*n].reshape(n, 2)
    r = p[2*n:]
    
    grad = np.zeros(3*n)
    
    # d(-sum r)/dr = -1
    grad[2*n:] = -1.0
    
    # Boundary penalty gradients
    for i in range(n):
        ri = r[i]
        xi, yi = 2*i, 2*i+1
        ridx = 2*n + i
        
        # x_i < r_i: violation = ri - xi
        if ri - pos[i, 0] > 0:
            grad[xi] += penalty_weight * (-1.0)
            grad[ridx] += penalty_weight * (1.0)
        # x_i > 1-r_i: violation = xi + ri - 1
        if pos[i, 0] + ri - 1.0 > 0:
            grad[xi] += penalty_weight * (1.0)
            grad[ridx] += penalty_weight * (1.0)
        # y_i < r_i
        if ri - pos[i, 1] > 0:
            grad[yi] += penalty_weight * (-1.0)
            grad[ridx] += penalty_weight * (1.0)
        # y_i > 1-r_i
        if pos[i, 1] + ri - 1.0 > 0:
            grad[yi] += penalty_weight * (1.0)
            grad[ridx] += penalty_weight * (1.0)
        # r_i < 0
        if -ri > 0:
            grad[ridx] += penalty_weight * (-1.0)
    
    # Overlap penalty gradients for contact edges
    for (i, j) in edges:
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        d = np.hypot(dx, dy)
        if d < EPS:
            d = EPS
        violation = r[i] + r[j] - d
        if violation > 0:
            grad[2*i] += penalty_weight * (dx / d)
            grad[2*i+1] += penalty_weight * (dy / d)
            grad[2*j] += penalty_weight * (-dx / d)
            grad[2*j+1] += penalty_weight * (-dy / d)
            grad[2*n+i] += penalty_weight * (1.0)
            grad[2*n+j] += penalty_weight * (1.0)
    
    # Overlap for non-contact pairs (safety)
    edge_set = set(edges)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in edge_set or (j, i) in edge_set:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.hypot(dx, dy)
            if d < EPS:
                d = EPS
            violation = r[i] + r[j] - d
            if violation > 0:
                grad[2*i] += penalty_weight * (dx / d)
                grad[2*i+1] += penalty_weight * (dy / d)
                grad[2*j] += penalty_weight * (-dx / d)
                grad[2*j+1] += penalty_weight * (-dy / d)
                grad[2*n+i] += penalty_weight * (1.0)
                grad[2*n+j] += penalty_weight * (1.0)
    
    return grad

def _initial_grid():
    """Grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    pos = np.zeros((N, 2))
    r = np.zeros(N)
    for i in range(N):
        row = i // side
        col = i % side
        pos[i, 0] = (col + 0.5) / side
        pos[i, 1] = (row + 0.5) / side
        r[i] = 0.4 / side
    return pos, r

def _initial_hex():
    """Hexagonal packing initialization."""
    pos = np.zeros((N, 2))
    r = np.zeros(N)
    # Hexagonal grid
    spacing = 0.19
    count = 0
    row = 0
    while count < N:
        col = 0
        while count < N:
            x = col * spacing * np.sqrt(3) + 0.1
            if row % 2 == 1:
                x += spacing * np.sqrt(3) / 2
            y = row * spacing * 1.5 + 0.1
            if x < 0.9 and y < 0.9:
                pos[count, 0] = x
                pos[count, 1] = y
                r[count] = spacing * 0.45
                count += 1
            col += 1
        row += 1
    return pos, r

def _initial_corner_weighted(rng):
    """Random initialization with bias toward corners."""
    pos = np.zeros((N, 2))
    r = np.zeros(N)
    for i in range(N):
        # Bias toward corners
        u = rng.random()
        v = rng.random()
        if u < 0.3:
            x = rng.random() * 0.3 + 0.05
        elif u > 0.7:
            x = 0.7 + rng.random() * 0.25
        else:
            x = 0.2 + rng.random() * 0.6
        if v < 0.3:
            y = rng.random() * 0.3 + 0.05
        elif v > 0.7:
            y = 0.7 + rng.random() * 0.25
        else:
            y = 0.2 + rng.random() * 0.6
        pos[i, 0] = x
        pos[i, 1] = y
        r[i] = 0.03
    return pos, r

def _optimize_single(pos, r, t_start, deadline):
    """Optimize a single configuration using penalty-based L-BFGS-B."""
    n = len(r)
    best_pos = pos.copy()
    best_r = r.copy()
    best_score = np.sum(r)
    
    # Adaptive penalty weight schedule
    penalty_weights = [10, 50, 200, 1000, 5000]
    
    for pw in penalty_weights:
        if _elapsed(t_start) > deadline - 15:
            break
        
        x0 = np.concatenate([pos.flatten(), r])
        edges = _get_contact_edges(pos)
        
        def obj(p):
            return _penalty_objective(p, n, edges, pw)
        
        def grad(p):
            return _penalty_grad(p, n, edges, pw)
        
        bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
        
        try:
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < obj(x0):
                new_pos = res.x[:2*n].reshape(n, 2)
                new_r = res.x[2*n:]
                new_pos, new_r = _fix_boundaries(new_pos, new_r)
                
                if _validate(new_pos, new_r):
                    score = np.sum(new_r)
                    if score > best_score:
                        best_score = score
                        best_pos = new_pos.copy()
                        best_r = new_r.copy()
                    pos = new_pos
                    r = new_r
                else:
                    # Try to repair
                    new_pos, new_r = _fix_boundaries(new_pos, new_r)
                    if _validate(new_pos, new_r):
                        score = np.sum(new_r)
                        if score > best_score:
                            best_score = score
                            best_pos = new_pos.copy()
                            best_r = new_r.copy()
                        pos = new_pos
                        r = new_r
        except Exception:
            pass
    
    return best_pos, best_r

def _shake_and_refine(pos, r, t_start, deadline, rng):
    """Perturb and re-optimize to escape local minima."""
    n = len(r)
    best_pos = pos.copy()
    best_r = r.copy()
    best_score = np.sum(r)
    
    for _ in range(3):
        if _elapsed(t_start) > deadline - 20:
            break
        
        # Small perturbation
        shake = 0.02
        pos_shaken = pos + rng.normal(0, shake, (n, 2))
        pos_shaken = np.clip(pos_shaken, 0.01, 0.99)
        r_shaken = r * (1 + rng.normal(0, 0.05, n))
        r_shaken = np.clip(r_shaken, 0.005, 0.5)
        
        pos_shaken, r_shaken = _fix_boundaries(pos_shaken, r_shaken)
        
        if _validate(pos_shaken, r_shaken):
            new_pos, new_r = _optimize_single(pos_shaken, r_shaken, t_start, deadline)
            score = np.sum(new_r)
            if score > best_score:
                best_score = score
                best_pos = new_pos.copy()
                best_r = new_r.copy()
    
    return best_pos, best_r

def construct():
    t_start = time.time()
    DEADLINE = 140.0  # Safety margin for 150s limit
    
    best_score = -1.0
    best_result = None
    
    # Try multiple initialization strategies
    inits = []
    
    # Grid
    pos_g, r_g = _initial_grid()
    inits.append(('grid', pos_g, r_g))
    
    # Hexagonal
    pos_h, r_h = _initial_hex()
    inits.append(('hex', pos_h, r_h))
    
    # Corner-weighted random (multiple seeds)
    for seed in [42, 123, 456, 789, 1024]:
        rng = np.random.default_rng(seed)
        pos_c, r_c = _initial_corner_weighted(rng)
        inits.append((f'corner_{seed}', pos_c, r_c))
    
    # Pure random
    for seed in [11, 22, 33]:
        rng = np.random.default_rng(seed)
        pos_r = rng.random((N, 2)) * 0.8 + 0.1
        r_r = np.full(N, 0.03)
        inits.append((f'rand_{seed}', pos_r, r_r))
    
    for init_name, pos, r in inits:
        if _elapsed(t_start) > DEADLINE - 30:
            break
        
        # Ensure initial validity
        pos, r = _fix_boundaries(pos, r)
        if not _validate(pos, r):
            scale = 0.5
            while not _validate(pos, r * scale) and scale > 0.01:
                scale *= 0.7
            r = r * scale
            pos, r = _fix_boundaries(pos, r)
        
        # Optimize
        pos_opt, r_opt = _optimize_single(pos, r, t_start, DEADLINE)
        
        # Shake and refine
        rng = np.random.default_rng(hash(init_name) % 10000)
        pos_shake, r_shake = _shake_and_refine(pos_opt, r_opt, t_start, DEADLINE, rng)
        
        score = np.sum(r_shake)
        if score > best_score:
            best_score = score
            best_result = np.column_stack([pos_shake, r_shake])
        
        # Early exit if very good
        if best_score > 2.9:
            break
    
    if best_result is None:
        # Fallback: grid
        pos, r = _initial_grid()
        pos, r = _fix_boundaries(pos, r)
        best_result = np.column_stack([pos, r])
    
    # Final validation
    if not _validate(best_result[:, :2], best_result[:, 2]):
        # Try to fix
        pos_fix = best_result[:, :2].copy()
        r_fix = best_result[:, 2].copy()
        pos_fix, r_fix = _fix_boundaries(pos_fix, r_fix)
        if not _validate(pos_fix, r_fix):
            # Shrink radii
            scale = 0.9
            while not _validate(pos_fix, r_fix * scale) and scale > 0.01:
                scale *= 0.8
            r_fix = r_fix * scale
            pos_fix, r_fix = _fix_boundaries(pos_fix, r_fix)
        best_result = np.column_stack([pos_fix, r_fix])
    
    return best_result

if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
