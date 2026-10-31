"""Circle packing in unit square: 26 circles, maximize sum of radii."""
N = 26
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

_START = time.time()
_DEADLINE = 150.0  # seconds, evaluator kills at 150s
_MARGIN = 5.0       # safety margin


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE - _MARGIN - _elapsed()


def _clamp_into_square(circles, eps=1e-10):
    """Ensure circles are inside [0,1]^2 by adjusting centers/radii minimally."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    # Clamp radii to be non-negative
    r = np.maximum(r, 0.0)
    # Clamp centers so that x - r >= 0, x + r <= 1, etc.
    # If r > 0.5, we need to shrink r
    r = np.minimum(r, 0.5)
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    # Re-clamp radii after center adjustment (center might have been clamped)
    r = np.minimum(r, np.minimum(x, 1.0 - x))
    r = np.minimum(r, np.minimum(y, 1.0 - y))
    r = np.maximum(r, 0.0)
    return np.column_stack([x, y, r])


def _is_valid(circles, tol=1e-7):
    """Check if circles are disjoint and inside the unit square."""
    n = circles.shape[0]
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    # Boundary check
    if np.any(x - r < -tol) or np.any(x + r > 1.0 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1.0 + tol):
        return False
    # Pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _score(circles):
    """Sum of radii, or 0 if invalid."""
    if not _is_valid(circles):
        return 0.0
    return float(np.sum(circles[:, 2]))


def _penalty(circles, w_overlap=100.0, w_boundary=100.0):
    """Compute penalty for violations."""
    n = circles.shape[0]
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    pen = 0.0
    # Boundary penalties
    pen += w_boundary * np.sum(np.maximum(0.0, r - x) ** 2)
    pen += w_boundary * np.sum(np.maximum(0.0, r - (1.0 - x)) ** 2)
    pen += w_boundary * np.sum(np.maximum(0.0, r - y) ** 2)
    pen += w_boundary * np.sum(np.maximum(0.0, r - (1.0 - y)) ** 2)
    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                pen += w_overlap * overlap ** 2
    return pen


def _objective(circles):
    """Objective: maximize sum of radii, penalize violations."""
    n = circles.shape[0]
    r = circles[:, 2]
    # Sum of radii (maximize)
    obj = -np.sum(r)
    # Penalties
    x, y = circles[:, 0], circles[:, 1]
    pen = 0.0
    pen += 100.0 * np.sum(np.maximum(0.0, r - x) ** 2)
    pen += 100.0 * np.sum(np.maximum(0.0, r - (1.0 - x)) ** 2)
    pen += 100.0 * np.sum(np.maximum(0.0, r - y) ** 2)
    pen += 100.0 * np.sum(np.maximum(0.0, r - (1.0 - y)) ** 2)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                pen += 100.0 * overlap ** 2
    return obj + pen


def _force_directed_anneal(circles, steps=5000, lr_start=0.01, lr_end=0.0001,
                            repulsion_power=2.0, temp_start=0.05, temp_end=0.001):
    """Phase-based force-directed annealing to escape local minima."""
    n = circles.shape[0]
    circles = circles.copy()
    best_circles = circles.copy()
    best_score = _score(best_circles)
    
    for step in range(steps):
        if _remaining() < 2.0:
            break
        t = step / max(steps - 1, 1)
        lr = lr_start * (lr_end / lr_start) ** t
        temp = temp_start * (temp_end / temp_start) ** t
        
        # Compute forces
        forces = np.zeros_like(circles)
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        
        # Repulsion between overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-12:
                    # Force proportional to overlap, direction away
                    overlap = min_dist - dist
                    force_mag = repulsion_power * overlap ** 2
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                elif dist <= 1e-12:
                    # Random small push
                    forces[i, 0] += 0.01 * np.random.randn()
                    forces[i, 1] += 0.01 * np.random.randn()
                    forces[j, 0] -= 0.01 * np.random.randn()
                    forces[j, 1] -= 0.01 * np.random.randn()
        
        # Boundary forces: push circles away from walls
        for i in range(n):
            # Left wall
            if x[i] < r[i]:
                forces[i, 0] += 10.0 * (r[i] - x[i])
            # Right wall
            if x[i] > 1.0 - r[i]:
                forces[i, 0] -= 10.0 * (x[i] - (1.0 - r[i]))
            # Bottom wall
            if y[i] < r[i]:
                forces[i, 1] += 10.0 * (r[i] - y[i])
            # Top wall
            if y[i] > 1.0 - r[i]:
                forces[i, 1] -= 10.0 * (y[i] - (1.0 - r[i]))
        
        # Apply forces with learning rate and random perturbation
        for i in range(n):
            circles[i, 0] += lr * forces[i, 0] + temp * np.random.randn()
            circles[i, 1] += lr * forces[i, 1] + temp * np.random.randn()
            # Slightly grow radius
            circles[i, 2] += lr * 0.5
        
        # Clamp into square
        circles = _clamp_into_square(circles)
        
        # Track best
        cur_score = _score(circles)
        if cur_score > best_score:
            best_score = cur_score
            best_circles = circles.copy()
    
    return best_circles, best_score


def _slsqp_refine(circles, max_iter=100):
    """Use SLSQP to refine a valid configuration."""
    n = circles.shape[0]
    x0 = circles.flatten().copy()
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.0, 0.5))  # r
    
    # Constraints: non-overlap
    def _make_constraint(i, j):
        def constraint(v):
            x_i, y_i, r_i = v[3*i], v[3*i+1], v[3*i+2]
            x_j, y_j, r_j = v[3*j], v[3*j+1], v[3*j+2]
            dx = x_i - x_j
            dy = y_i - y_j
            dist = np.sqrt(dx*dx + dy*dy)
            return dist - (r_i + r_j)
        return constraint
    
    constraints = []
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': _make_constraint(i, j)})
    
    # Objective: maximize sum of radii
    def obj(v):
        return -np.sum(v[2::3])
    
    def obj_grad(v):
        g = np.zeros_like(v)
        g[2::3] = -1.0
        return g
    
    result = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                       jac=obj_grad, options={'maxiter': max_iter, 'ftol': 1e-10})
    
    if result.success:
        refined = result.x.reshape(n, 3)
        refined = _clamp_into_square(refined)
        if _is_valid(refined):
            return refined
    return circles


def _generate_seed(idx, n=N):
    """Generate diverse initial configurations."""
    np.random.seed(idx * 12345 + 678)
    
    if idx % 5 == 0:
        # Grid-based
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:n])
    elif idx % 5 == 1:
        # Random positions, equal radii
        x = np.random.uniform(0.1, 0.9, n)
        y = np.random.uniform(0.1, 0.9, n)
        r = np.full(n, 0.08)
        return np.column_stack([x, y, r])
    elif idx % 5 == 2:
        # Random with varying radii
        x = np.random.uniform(0.1, 0.9, n)
        y = np.random.uniform(0.1, 0.9, n)
        r = np.random.uniform(0.05, 0.12, n)
        return np.column_stack([x, y, r])
    elif idx % 5 == 3:
        # Clustered
        centers = np.random.uniform(0.2, 0.8, (4, 2))
        x, y, r = [], [], []
        for k in range(n):
            cx, cy = centers[k % 4]
            x.append(cx + np.random.uniform(-0.15, 0.15))
            y.append(cy + np.random.uniform(-0.15, 0.15))
            r.append(np.random.uniform(0.06, 0.10))
        return np.column_stack([x, y, r])
    else:
        # Dense grid with perturbation
        side = int(np.ceil(np.sqrt(n)))
        x, y, r = [], [], []
        for i in range(side):
            for j in range(side):
                if len(x) >= n:
                    break
                x.append((i + 0.5) / side + np.random.uniform(-0.02, 0.02))
                y.append((j + 0.5) / side + np.random.uniform(-0.02, 0.02))
                r.append(0.5 / side - 1e-3 + np.random.uniform(-0.005, 0.005))
        return np.column_stack([x, y, r])


def construct():
    best_circles = None
    best_score = 0.0
    
    # Generate and evaluate multiple seeds
    n_seeds = 20
    for idx in range(n_seeds):
        if _remaining() < 5.0:
            break
        
        circles = _generate_seed(idx)
        circles = _clamp_into_square(circles)
        
        # Force-directed anneal
        annealed, ann_score = _force_directed_anneal(
            circles, steps=3000, lr_start=0.008, lr_end=0.00005,
            repulsion_power=2.0, temp_start=0.03, temp_end=0.0005
        )
        
        if ann_score > best_score:
            best_score = ann_score
            best_circles = annealed.copy()
        
        # SLSQP refinement on the best so far
        if _remaining() > 10.0:
            refined = _slsqp_refine(best_circles, max_iter=80)
            ref_score = _score(refined)
            if ref_score > best_score:
                best_score = ref_score
                best_circles = refined.copy()
    
    # Final polish: multiple SLSQP passes
    if best_circles is not None and _remaining() > 5.0:
        for _ in range(3):
            if _remaining() < 3.0:
                break
            refined = _slsqp_refine(best_circles, max_iter=50)
            ref_score = _score(refined)
            if ref_score > best_score:
                best_score = ref_score
                best_circles = refined.copy()
    
    if best_circles is None:
        # Fallback: trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
