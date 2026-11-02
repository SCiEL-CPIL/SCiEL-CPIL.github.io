"""Heilbronn-11: maximize minimum triangle area using barycentric coords and gradient ascent."""
import numpy as np
import time
import itertools

N = 11
S3 = np.sqrt(3.0)
S3H = S3 / 2.0
TRI_AREA = S3 / 4.0  # area of unit equilateral triangle

# Deadline: 140s safety margin from 150s limit
DEADLINE = 140

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)  # 165


def _bary_to_cart(bary):
    """Convert barycentric coordinates (n,3) to Cartesian (n,2)."""
    # bary columns: weight for V0, V1, V2
    x = bary[:, 1] * 1.0 + bary[:, 2] * 0.5
    y = bary[:, 2] * S3H
    return np.column_stack([x, y])


def _cart_to_bary(pts):
    """Convert Cartesian (n,2) to barycentric (n,3) coordinates."""
    x = pts[:, 0]
    y = pts[:, 1]
    # V0=(0,0), V1=(1,0), V2=(0.5, S3H)
    # b2 = y / S3H
    b2 = y / S3H
    # x = b1 + 0.5*b2 => b1 = x - 0.5*b2
    b1 = x - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack([b0, b1, b2])


def _project_bary(bary):
    """Project barycentric coords into valid simplex (sum=1, all >= 0)."""
    bary = np.maximum(bary, 0.0)
    s = bary.sum(axis=1, keepdims=True)
    s = np.maximum(s, 1e-15)
    return bary / s


def _min_area_from_bary(bary):
    """Compute minimum triangle area from barycentric coordinates."""
    pts = _bary_to_cart(bary)
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _all_areas_from_bary(bary):
    """Compute all triangle areas from barycentric coordinates."""
    pts = _bary_to_cart(bary)
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return np.abs(cross) * 0.5


def _soft_min(areas, beta):
    """Soft minimum approximation: -(1/beta)*log(sum(exp(-beta*a)))."""
    # For numerical stability
    min_a = np.min(areas)
    shifted = areas - min_a
    return min_a - np.log(np.sum(np.exp(-beta * shifted))) / beta


def _soft_min_grad(areas, beta):
    """Gradient of soft min w.r.t. areas."""
    min_a = np.min(areas)
    shifted = areas - min_a
    weights = np.exp(-beta * shifted)
    weights /= np.sum(weights)
    return -weights  # d/dareas of soft_min = -weights


def _objective_and_grad(bary, beta):
    """
    Compute -soft_min(areas) and its gradient w.r.t. barycentric coords.
    We maximize soft_min, so minimize -soft_min.
    """
    pts = _bary_to_cart(bary)
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    
    # Cross product: (pj-pi) x (pk-pi)
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    soft_min_val = _soft_min(areas, beta)
    
    # Gradient of soft_min w.r.t. areas
    dsoft_da = _soft_min_grad(areas, beta)
    
    # Gradient of areas w.r.t. points:
    # area = 0.5 * |cross|
    # d(area)/d(cross) = 0.5 * sign(cross)
    sign_cross = np.sign(cross)
    darea_dcross = 0.5 * sign_cross
    
    # cross = dx1*dy2 - dy1*dx2
    # d(cross)/d(pj_x) = dy2, d(cross)/d(pj_y) = -dx2
    # d(cross)/d(pk_x) = -dy1, d(cross)/d(pk_y) = dx1
    # d(cross)/d(pi_x) = -(dy2) + dx1... let me be careful
    # cross = (pj_x - pi_x)*(pk_y - pi_y) - (pj_y - pi_y)*(pk_x - pi_x)
    # d(cross)/d(pi_x) = -(pk_y - pi_y) + (pj_y - pi_y) = -dy2 + dy1
    # d(cross)/d(pi_y) = -(pj_x - pi_x) + (pk_x - pi_x) = -dx1 + dx2
    # d(cross)/d(pj_x) = dy2
    # d(cross)/d(pj_y) = -dx2
    # d(cross)/d(pk_x) = -dy1
    # d(cross)/d(pk_y) = dx1
    
    # Gradient w.r.t. Cartesian points
    grad_x = np.zeros((N, 2))
    for t in range(_N_TRIP):
        i, j, k = _TRIPLETS[t]
        w = dsoft_da[t] * darea_dcross[t]
        grad_x[i, 0] += w * (-dy2[t] + dy1[t])
        grad_x[i, 1] += w * (-dx1[t] + dx2[t])
        grad_x[j, 0] += w * dy2[t]
        grad_x[j, 1] += w * (-dx2[t])
        grad_x[k, 0] += w * (-dy1[t])
        grad_x[k, 1] += w * dx1[t]
    
    # Chain rule: bary -> cart
    # x = b1 + 0.5*b2, y = S3H*b2
    # d(x)/d(b1) = 1, d(x)/d(b2) = 0.5, d(y)/d(b2) = S3H
    # grad_bary[i,0] = 0 (b0 doesn't affect x,y directly, but b0=1-b1-b2)
    # Actually, let's think: x = b1 + 0.5*b2, y = S3H*b2
    # d(x)/d(b1) = 1, d(x)/d(b2) = 0.5
    # d(y)/d(b1) = 0, d(y)/d(b2) = S3H
    # So grad_bary[:, 1] = grad_x[:, 0]  (from b1)
    # grad_bary[:, 2] = 0.5*grad_x[:, 0] + S3H*grad_x[:, 1]  (from b2)
    # grad_bary[:, 0] = -(grad_bary[:, 1] + grad_bary[:, 2])  (since b0 = 1-b1-b2, but this is just for consistency)
    
    # Actually for optimization we only need gradients w.r.t. independent variables.
    # Since b0 = 1 - b1 - b2, we optimize over b1 and b2.
    # But we can compute full gradient and project.
    
    grad_bary = np.zeros_like(bary)
    grad_bary[:, 1] = grad_x[:, 0]
    grad_bary[:, 2] = 0.5 * grad_x[:, 0] + S3H * grad_x[:, 1]
    grad_bary[:, 0] = -(grad_bary[:, 1] + grad_bary[:, 2])
    
    # We want to MAXIMIZE soft_min, so the gradient for ascent is grad_bary
    # For minimization of -soft_min, the gradient is -grad_bary
    return -soft_min_val, -grad_bary


def _project_to_triangle_cart(pts):
    """Project Cartesian points into the unit equilateral triangle."""
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])


def _gradient_ascent(bary_start, start_time, max_iters=5000, beta_init=20.0, beta_end=100.0):
    """
    Gradient ascent on soft-min objective in barycentric coordinates.
    """
    bary = _project_bary(bary_start.copy())
    current_val = _min_area_from_bary(bary)
    best_bary = bary.copy()
    best_val = current_val
    
    for it in range(max_iters):
        if (time.time() - start_time) > (DEADLINE - 10):
            break
        
        # Adaptive beta
        progress = it / max_iters
        beta = beta_init * (beta_end / beta_init) ** progress
        
        # Compute gradient
        obj, grad = _objective_and_grad(bary, beta)
        
        # Gradient ascent step with adaptive learning rate
        lr = 0.001 * (1.0 - 0.5 * progress)
        bary_new = bary + lr * grad
        bary_new = _project_bary(bary_new)
        
        new_val = _min_area_from_bary(bary_new)
        
        if new_val > current_val:
            bary = bary_new
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_bary = bary.copy()
        else:
            # Reduce learning rate and try again
            lr *= 0.5
            bary_new = bary + lr * grad
            bary_new = _project_bary(bary_new)
            new_val = _min_area_from_bary(bary_new)
            if new_val > current_val:
                bary = bary_new
                current_val = new_val
                if current_val > best_val:
                    best_val = current_val
                    best_bary = bary.copy()
    
    return best_bary, best_val


def _push_apart_worst_bary(bary, step_size=0.01):
    """Push apart the worst triangle's vertices in barycentric space."""
    pts = _bary_to_cart(bary)
    areas = _all_areas_from_bary(bary)
    min_idx = np.argmin(areas)
    i, j, k = _TRIPLETS[min_idx]
    
    pi = pts[i]
    pj = pts[j]
    pk = pts[k]
    centroid = (pi + pj + pk) / 3.0
    
    bary_new = bary.copy()
    for idx, pt in [(i, pi), (j, pj), (k, pk)]:
        direction = pt - centroid
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
            # Move in Cartesian, then convert back
            new_pt = pt + step_size * direction
            new_pt = _project_to_triangle_cart(new_pt.reshape(1, 2))[0]
            bary_new[idx] = _cart_to_bary(new_pt.reshape(1, 2))[0]
    
    return _project_bary(bary_new)


def _simulated_annealing_bary(bary_start, start_time, deadline):
    """Simulated annealing in barycentric space."""
    bary = _project_bary(bary_start.copy())
    current_val = _min_area_from_bary(bary)
    best_bary = bary.copy()
    best_val = current_val
    
    T0 = 0.005
    Tend = 0.0001
    n_steps = 20000
    
    for step in range(n_steps):
        if (time.time() - start_time) > deadline:
            break
        
        T = T0 * (Tend / T0) ** (step / n_steps)
        
        # Random perturbation in barycentric space
        bary_cand = bary + np.random.normal(0, 0.02 * (T / T0 + 0.1), size=bary.shape)
        bary_cand = _project_bary(bary_cand)
        
        cand_val = _min_area_from_bary(bary_cand)
        
        delta = cand_val - current_val
        if delta > 0 or np.random.random() < np.exp(delta / T):
            bary = bary_cand
            current_val = cand_val
            if current_val > best_val:
                best_val = current_val
                best_bary = bary.copy()
    
    return best_bary, best_val


def _iterative_refine_bary(bary_start, start_time, max_iters=1000):
    """Iteratively push apart worst triangle with small perturbations."""
    bary = _project_bary(bary_start.copy())
    current_val = _min_area_from_bary(bary)
    best_bary = bary.copy()
    best_val = current_val
    
    step_size = 0.005
    
    for it in range(max_iters):
        if (time.time() - start_time) > (DEADLINE - 5):
            break
        
        bary_cand = _push_apart_worst_bary(bary, step_size)
        cand_val = _min_area_from_bary(bary_cand)
        
        if cand_val > current_val:
            bary = bary_cand
            current_val = cand_val
            if current_val > best_val:
                best_val = current_val
                best_bary = bary.copy()
        else:
            # Small random perturbation to the worst triangle vertices
            pts = _bary_to_cart(bary)
            areas = _all_areas_from_bary(bary)
            min_idx = np.argmin(areas)
            i, j, k = _TRIPLETS[min_idx]
            for idx in [i, j, k]:
                pts[idx] = pts[idx] + np.random.normal(0, 0.003, size=2)
            pts = _project_to_triangle_cart(pts)
            bary = _project_bary(_cart_to_bary(pts))
            current_val = _min_area_from_bary(bary)
        
        if it % 50 == 0:
            step_size *= 0.95
    
    return best_bary, best_val


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_bary = None
    best_area = 0.0
    
    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(bary):
        nonlocal best_bary, best_area
        area = _min_area_from_bary(bary)
        if area > best_area:
            best_area = area
            best_bary = _project_bary(bary.copy())
    
    # Generate diverse initial configurations in barycentric space
    inits = []
    
    # Init 1: Symmetric distribution
    init1_cart = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [0.5, S3H / 3],
        [0.333, S3H / 6],
        [0.667, S3H / 6],
        [0.5, 2 * S3H / 3],
        [0.2, 0.2],
    ], dtype=float)
    inits.append(_project_bary(_cart_to_bary(init1_cart)))
    
    # Init 2: Grid-like
    init2_cart = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, S3H / 2],
        [0.75, S3H / 2],
        [0.5, S3H / 2],
        [0.5, S3H / 6],
        [0.5, 2 * S3H / 3],
        [0.5, 0.0],
    ], dtype=float)
    inits.append(_project_bary(_cart_to_bary(init2_cart)))
    
    # Init 3: Random spread
    u = rng.random((N, 2))
    mask = u[:, 0] + u[:, 1] > 1
    u[mask, 0] = 1 - u[mask, 0]
    u[mask, 1] = 1 - u[mask, 1]
    x = u[:, 0] + 0.5 * u[:, 1]
    y = u[:, 1] * S3H
    init3_cart = np.column_stack([x, y])
    inits.append(_project_bary(_cart_to_bary(init3_cart)))
    
    # Evaluate all inits
    for init in inits:
        _update_best(init)
    
    # Phase A: Gradient ascent from best init
    if _elapsed_ok(40):
        ga_deadline = start_time + 60
        ga_best, ga_area = _gradient_ascent(best_bary, start_time, max_iters=3000, 
                                             beta_init=20.0, beta_end=80.0)
        _update_best(ga_best)
    
    # Phase B: Simulated annealing
    if _elapsed_ok(30):
        sa_deadline = start_time + 90
        sa_best, sa_area = _simulated_annealing_bary(best_bary, start_time, sa_deadline)
        _update_best(sa_best)
    
    # Phase C: Iterative refinement
    if _elapsed_ok(20):
        for pass_num in range(3):
            if not _elapsed_ok(10):
                break
            if pass_num == 0:
                start = best_bary.copy()
            else:
                start = best_bary + rng.normal(0, 0.03, size=best_bary.shape)
                start = _project_bary(start)
            
            time_left = DEADLINE - (time.time() - start_time) - 8
            if time_left <= 0:
                break
            
            max_iters = int(min(1500, max(100, time_left * 3000)))
            ref_best, ref_area = _iterative_refine_bary(start, start_time, max_iters=max_iters)
            _update_best(ref_best)
    
    # Phase D: Final fine-grained gradient ascent
    if _elapsed_ok(10):
        final_best, final_area = _gradient_ascent(best_bary, start_time, max_iters=2000,
                                                   beta_init=50.0, beta_end=200.0)
        _update_best(final_best)
    
    # Convert best barycentric to Cartesian
    if best_bary is None:
        # Fallback: random points
        u = rng.random((N, 2))
        mask = u[:, 0] + u[:, 1] > 1
        u[mask, 0] = 1 - u[mask, 0]
        u[mask, 1] = 1 - u[mask, 1]
        x = u[:, 0] + 0.5 * u[:, 1]
        y = u[:, 1] * S3H
        return np.column_stack([x, y])
    
    result = _bary_to_cart(_project_bary(best_bary))
    return _project_to_triangle_cart(result)
