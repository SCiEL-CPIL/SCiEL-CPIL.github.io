"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize
import warnings

N = 11
H = np.sqrt(3) / 2

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)
_I = tri_idx[:, 0]
_J = tri_idx[:, 1]
_K = tri_idx[:, 2]

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area ratios.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[_I]
    vi = v[_I]
    wi = w[_I]
    
    uj = u[_J]
    vj = v[_J]
    wj = w[_J]
    
    uk = u[_K]
    vk = v[_K]
    wk = w[_K]
    
    # Determinant calculation vectorized
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-9):
    """
    Clamp barycentric coordinates to the strictly valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Clip to strict interior
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    # Ensure u + v <= 1 - eps
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    # Re-check bounds after scaling
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def hard_min_obj(bary_flat):
    """
    Objective function using hard min.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    bary: (N, 2) array of (u, v). w = 1 - u - v.
    Point = u*A + v*B + w*C
    A=(0,0), B=(1,0), C=(0.5, H)
    x = v + 0.5*w
    y = w*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def generate_structural_init():
    """
    Generate initial configurations based on structural principles.
    For 11 points in an equilateral triangle, use combinations of:
    - Corners
    - Edge midpoints
    - Interior points in triangular/hexagonal patterns
    """
    candidates = []
    
    # Pattern 1: 3 corners + 3 edge midpoints + 5 interior
    corners = np.array([[1e-9, 1e-9], [1-1e-9, 1e-9], [0.5, H - 1e-9]])
    edge_mid = np.array([[0.5, 1e-9], [0.75, H/2], [0.25, H/2]])
    # Interior: center + 4 around it
    cx, cy = 1/3, H/3
    r = 0.15
    interior = np.array([
        [cx, cy],
        [cx + r, cy],
        [cx - r, cy],
        [cx, cy + r*H],
        [cx, cy - r*H]
    ])
    candidates.append(np.vstack([corners, edge_mid, interior]))
    
    # Pattern 2: 3 corners + 6 edge points + 2 interior
    corners2 = np.array([[1e-9, 1e-9], [1-1e-9, 1e-9], [0.5, H - 1e-9]])
    edge_pts = np.array([
        [0.3, 1e-9], [0.7, 1e-9],
        [0.85, H*0.5], [0.15, H*0.5],
        [0.35, H*0.85], [0.65, H*0.85]
    ])
    interior2 = np.array([[1/3, H/3], [2/3, H/3]])
    candidates.append(np.vstack([corners2, edge_pts, interior2]))
    
    # Pattern 3: Hexagonal lattice
    for s in [0.2, 0.25, 0.3]:
        pts = []
        for i in range(6):
            for j in range(6):
                u = i * s
                v = j * s
                if u + v <= 1 - 1e-6 and len(pts) < N:
                    pts.append([u, v])
        if len(pts) == N:
            candidates.append(np.array(pts))
    
    # Pattern 4: Random with structure
    rng = np.random.default_rng(42)
    for seed in range(10):
        rng = np.random.default_rng(seed)
        pts = []
        # Place some near corners
        for corner in [V0, V1, V2]:
            pts.append(corner + rng.normal(0, 0.02, 2))
        # Place rest randomly
        while len(pts) < N:
            u = rng.random()
            v = rng.random() * (1 - u)
            pts.append([u, v])
        candidates.append(np.array(pts))
    
    # Evaluate all candidates
    best = None
    best_score = -np.inf
    for cand in candidates:
        cand = clamp_bary(cand)
        score = -hard_min_obj(cand.flatten())
        if score > best_score:
            best_score = score
            best = cand.copy()
    
    return best

def _directed_hill_climb(bary, time_limit=30.0, seed=None):
    """
    Directed hill climbing: repeatedly move the point that is part of the worst triangle
    in a direction that improves the minimum area.
    """
    rng = np.random.default_rng(seed)
    bary = clamp_bary(np.array(bary, dtype=float)).copy()
    current_score = -hard_min_obj(bary.flatten())
    best_bary = bary.copy()
    best_score = current_score
    
    start_time = time.time()
    step_size = 0.03
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
        # Find the worst triangle
        areas = calc_areas_bary(bary)
        worst_idx = np.argmin(areas)
        worst_area = areas[worst_idx]
        
        # Get the three points in the worst triangle
        i, j, k = _I[worst_idx], _J[worst_idx], _K[worst_idx]
        
        # Try moving each of the three points
        best_move = None
        best_move_score = current_score
        
        for pt_idx in [i, j, k]:
            # Try multiple directions and step sizes
            for _ in range(20):
                # Random direction with bias
                angle = rng.uniform(0, 2 * np.pi)
                du = step_size * np.cos(angle)
                dv = step_size * np.sin(angle)
                
                cand = bary.copy()
                cand[pt_idx, 0] += du
                cand[pt_idx, 1] += dv
                cand = clamp_bary(cand)
                
                score = -hard_min_obj(cand.flatten())
                
                if score > best_move_score + 1e-10:
                    best_move_score = score
                    best_move = cand.copy()
        
        if best_move is not None:
            bary = best_move
            current_score = best_move_score
            if current_score > best_score:
                best_score = current_score
                best_bary = bary.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            # Adaptive step size
            step_size *= 0.98
        else:
            no_improve_count += 1
            # Try larger random jump to escape
            if no_improve_count % 10 == 0:
                jump_idx = rng.integers(N)
                bary[jump_idx] += rng.normal(0, step_size * 2, 2)
                bary = clamp_bary(bary)
                current_score = -hard_min_obj(bary.flatten())
                if current_score > best_score:
                    best_score = current_score
                    best_bary = bary.copy()
    
    return best_bary, best_score

def _coordinate_descent(bary, time_limit=15.0):
    """
    Coordinate descent: iteratively optimize each point's (u, v) independently.
    """
    bary = clamp_bary(np.array(bary, dtype=float)).copy()
    best_score = -hard_min_obj(bary.flatten())
    best_bary = bary.copy()
    
    start_time = time.time()
    step = 0.01
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            base_u, base_v = bary[i]
            cur_score = best_score
            
            # Try 8 directions
            for du in [step, -step, 0]:
                for dv in [step, -step, 0]:
                    if du == 0 and dv == 0:
                        continue
                    cand = bary.copy()
                    cand[i, 0] = base_u + du
                    cand[i, 1] = base_v + dv
                    cand = clamp_bary(cand)
                    s = -hard_min_obj(cand.flatten())
                    if s > cur_score + 1e-12:
                        cur_score = s
                        bary = cand
                        improved = True
                        if s > best_score:
                            best_score = s
                            best_bary = bary.copy()
        
        # Reduce step size
        step *= 0.85
    
    return best_bary, best_score

def _slsqp_polish(bary, time_limit=10.0):
    """SLSQP polish with explicit constraints."""
    bary = clamp_bary(np.array(bary, dtype=float)).copy()
    
    def obj(x):
        return hard_min_obj(x)
    
    # Constraints: each point must be inside the triangle
    # In barycentric: u >= 0, v >= 0, u+v <= 1
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i] - x[2*i+1]})
    
    bounds = [(0, 1)] * (N * 2)
    
    start_time = time.time()
    try:
        result = minimize(
            obj,
            bary.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-12}
        )
        if result.success:
            return clamp_bary(result.x.reshape(N, 2))
    except Exception:
        pass
    return bary

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Phase 1: Generate diverse initial configurations
    initial_configs = []
    
    # Structural initialization
    struct_init = generate_structural_init()
    if struct_init is not None:
        initial_configs.append(struct_init)
    
    # Multiple random initializations
    for seed in range(20):
        rng = np.random.default_rng(seed)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        initial_configs.append(np.column_stack([u, v]))
    
    # Evaluate all initial configs
    for cfg in initial_configs:
        update_best(cfg)
    
    # Phase 2: Directed hill climbing on best config
    if best_bary is not None and time.time() < deadline - 60:
        remaining = deadline - time.time() - 60
        if remaining > 20:
            dhc_bary, dhc_score = _directed_hill_climb(best_bary, time_limit=min(remaining, 40), seed=42)
            update_best(dhc_bary)
    
    # Phase 3: Coordinate descent polish
    if best_bary is not None and time.time() < deadline - 30:
        remaining = deadline - time.time() - 30
        if remaining > 10:
            cd_bary, cd_score = _coordinate_descent(best_bary, time_limit=min(remaining, 20))
            update_best(cd_bary)
    
    # Phase 4: Try hill climbing from other initial configs
    for cfg in initial_configs[1:6]:  # Try a few more
        if time.time() > deadline - 40:
            break
        dhc_bary, dhc_score = _directed_hill_climb(cfg, time_limit=8, seed=100)
        update_best(dhc_bary)
    
    # Phase 5: SLSQP polish
    if best_bary is not None and time.time() < deadline - 15:
        slsqp_bary = _slsqp_polish(best_bary, time_limit=8)
        update_best(slsqp_bary)
    
    # Phase 6: Final coordinate descent
    if best_bary is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        if remaining > 3:
            cd_bary, cd_score = _coordinate_descent(best_bary, time_limit=remaining)
            update_best(cd_bary)
    
    # Convert best barycentric to Cartesian
    if best_bary is not None:
        result = bary_to_cart(best_bary)
    else:
        # Fallback
        rng = np.random.default_rng(0)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        result = bary_to_cart(np.column_stack([u, v]))
    
    return result

# EVOLVE-BLOCK-END
