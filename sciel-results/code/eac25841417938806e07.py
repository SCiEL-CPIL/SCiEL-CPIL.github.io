import math
import time
import numpy as np
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9

def _axial_to_cartesian(axial_list):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

def _hex_normals(beta):
    return np.array([
        (math.cos(beta + math.pi/6.0 + k*math.pi/3.0),
         math.sin(beta + math.pi/6.0 + k*math.pi/3.0))
        for k in range(6)
    ])

def _min_enclosing_side(pts, beta):
    """LP: minimize D s.t. |dot(p-c, n_k)| <= D for all k. Returns (D, cx, cy)."""
    normals = _hex_normals(beta)
    n_pts = len(pts)
    n_con = 12 * n_pts
    A = np.zeros((n_con, 3))
    b = np.zeros(n_con)
    idx = 0
    for k in range(6):
        nx, ny = normals[k]
        for i in range(n_pts):
            px, py = pts[i]
            A[idx, 0] = -nx; A[idx, 1] = -ny; A[idx, 2] = -1.0
            b[idx] = -(px*nx + py*ny)
            idx += 1
            A[idx, 0] = nx; A[idx, 1] = ny; A[idx, 2] = -1.0
            b[idx] = px*nx + py*ny
            idx += 1
    c = np.array([0.0, 0.0, 1.0])
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=[(None,None),(None,None),(0,None)], method='highs')
        if res.status == 0:
            return res.x[2], float(res.x[0]), float(res.x[1])
    except Exception:
        pass
    return float('inf'), 0.0, 0.0

def _exact_side(pts, beta, cx, cy):
    normals = _hex_normals(beta)
    max_proj = 0.0
    for nx, ny in normals:
        projs = (pts[:,0]-cx)*nx + (pts[:,1]-cy)*ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
    return (max_proj + APOTHEM) / APOTHEM

def _smooth_side(pts, beta, cx, cy, temp=0.05):
    normals = _hex_normals(beta)
    centered = pts - np.array([cx, cy])
    projs = centered @ normals.T  # (n,6)
    all_p = np.concatenate([projs.ravel(), (-projs).ravel()])
    m = np.max(all_p)
    if m > 50:
        return (m + APOTHEM) / APOTHEM
    shifted = np.clip((all_p - m)/temp, -50, 50)
    D = temp * (np.log(np.sum(np.exp(shifted))) + m/temp)
    return (D + APOTHEM) / APOTHEM

def _laplacian_grad(pts, beta, cx, cy, temp=0.05):
    """Smooth gradient of side length w.r.t. (positions, cx, cy)."""
    normals = _hex_normals(beta)
    centered = pts - np.array([cx, cy])
    projs = centered @ normals.T  # (n,6)
    all_p = np.concatenate([projs.ravel(), (-projs).ravel()])
    m = np.max(all_p)
    shifted = np.clip((all_p - m)/temp, -50, 50)
    w = np.exp(shifted) / np.sum(np.exp(shifted))
    # d all_p / d pts[i] and d all_p / d c
    # all_p first half: projs.ravel() -> for each (i,k): (pts[i]-c) . n_k
    # second half: -projs.ravel()
    n = len(pts)
    grad_pts = np.zeros_like(pts)
    grad_c = np.zeros(2)
    for i in range(n):
        for k in range(6):
            w_pos = w[i*6+k]
            w_neg = w[n*6 + i*6+k]
            nx, ny = normals[k]
            # d/d pts[i] of pos term: w_pos * n_k ; of neg term: -w_neg * n_k
            grad_pts[i,0] += (w_pos - w_neg)*nx
            grad_pts[i,1] += (w_pos - w_neg)*ny
            # d/d c of pos term: -w_pos * n_k ; of neg term: w_neg * n_k
            grad_c[0] += (-w_pos + w_neg)*nx
            grad_c[1] += (-w_pos + w_neg)*ny
    return grad_pts, grad_c

def _sa_optimize(pts0, beta0, n_iter, temp0, temp_min, time_budget):
    """Simulated annealing on positions + center + beta. Returns best (S, pts, beta, cx, cy)."""
    pts = pts0.copy()
    cx, cy = float(np.mean(pts[:,0])), float(np.mean(pts[:,1]))
    beta = beta0 % (math.pi/6)
    best_S = _exact_side(pts, beta, cx, cy)
    best_state = (pts.copy(), beta, cx, cy)
    t = temp0
    cooling = (temp_min / temp0) ** (1.0 / max(1, n_iter))
    t_start = time.time()
    for it in range(n_iter):
        if time.time() - t_start > time_budget or time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        # Propose perturbation
        noise = np.random.normal(0, t*0.1, size=(N,2))
        new_pts = pts + noise
        new_beta = (beta + np.random.normal(0, t*0.02)) % (math.pi/6)
        new_cx = cx + np.random.normal(0, t*0.05)
        new_cy = cy + np.random.normal(0, t*0.05)
        new_S = _smooth_side(new_pts, new_beta, new_cx, new_cy, temp=max(t*0.3, 0.01))
        cur_S = _smooth_side(pts, beta, cx, cy, temp=max(t*0.3, 0.01))
        if new_S < cur_S or np.random.random() < math.exp(-(new_S - cur_S) / max(t*0.1, 1e-6)):
            pts, beta, cx, cy = new_pts, new_beta, new_cx, new_cy
            exact_S = _exact_side(pts, beta, cx, cy)
            if exact_S < best_S:
                best_S = exact_S
                best_state = (pts.copy(), beta, cx, cy)
        t *= cooling
    return best_S, best_state[0], best_state[1], best_state[2], best_state[3]

def _polish(pts, beta, cx, cy):
    """High-precision Nelder-Mead polish on (cx, cy, beta) with exact objective."""
    def obj(p):
        cx2, cy2, b2 = p
        b2 = ((b2 % (math.pi/6)) + math.pi/6) % (math.pi/6)
        return _exact_side(pts, b2, cx2, cy2)
    best_S = obj([cx, cy, beta])
    best_x = [cx, cy, beta]
    starts = [
        [cx, cy, beta],
        [cx, cy, (beta + 0.01) % (math.pi/6)],
        [cx, cy, (beta - 0.01) % (math.pi/6)],
        [cx + 0.01, cy, beta],
        [cx - 0.01, cy, beta],
        [cx, cy + 0.01, beta],
        [cx, cy - 0.01, beta],
    ]
    for s in starts:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        try:
            res = minimize(obj, s, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_S:
                best_S = res.fun
                best_x = res.x
        except Exception:
            pass
    # Final LP center for the best beta
    b_final = ((best_x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
    D, cx_f, cy_f = _min_enclosing_side(pts, b_final)
    if D < float('inf'):
        S_final = (D + APOTHEM) / APOTHEM
        if S_final < best_S:
            best_S = S_final
            best_x = [cx_f, cy_f, b_final]
    return best_S, best_x[0], best_x[1], best_x[2]

def _generate_lattice_seeds():
    """Generate diverse lattice-based seed configurations."""
    seeds = []
    # Classic flower + extensions
    axial_sets = [
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)],
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2),(1,3),(2,3)],
    ]
    for axial in axial_sets:
        if len(axial) == N:
            seeds.append(_axial_to_cartesian(axial))
    return seeds

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    np.random.seed(42)
    
    # Generate seeds
    seeds = _generate_lattice_seeds()
    # Add random seeds
    for _ in range(5):
        # Random positions in a hexagonal-ish region
        angles = np.random.uniform(0, 2*math.pi, N)
        radii = np.random.uniform(0, 2.5, N)
        pts = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])
        seeds.append(pts)
    # Add ring seeds
    for r in [1.5, 2.0, 2.5]:
        angles = np.linspace(0, 2*math.pi, N, endpoint=False) + np.random.uniform(0, 0.5)
        pts = np.column_stack([r*np.cos(angles), r*np.sin(angles)])
        seeds.append(pts)

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_cx = 0.0
    best_global_cy = 0.0

    # Time budget per seed
    n_seeds = len(seeds)
    time_per_seed = (TIME_LIMIT - SAFETY_MARGIN - 20.0) / max(1, n_seeds)
    time_per_seed = max(2.0, min(15.0, time_per_seed))

    for seed_idx, seed_pts in enumerate(seeds):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        # Initial center
        cx0 = float(np.mean(seed_pts[:,0]))
        cy0 = float(np.mean(seed_pts[:,1]))
        # Try a few initial betas
        for beta0 in [0.0, math.pi/12, math.pi/6*0.25]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            S, pts, beta, cx, cy = _sa_optimize(seed_pts, beta0, n_iter=800, temp0=0.5, temp_min=0.01, time_budget=time_per_seed)
            if S < best_global_S:
                best_global_S = S
                best_global_pts = pts
                best_global_beta = beta
                best_global_cx = cx
                best_global_cy = cy

    if best_global_pts is None:
        # Fallback
        seed_pts = _axial_to_cartesian([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
        best_global_pts = seed_pts
        best_global_beta = 0.0
        best_global_cx = float(np.mean(seed_pts[:,0]))
        best_global_cy = float(np.mean(seed_pts[:,1]))
        best_global_S = _exact_side(best_global_pts, best_global_beta, best_global_cx, best_global_cy)

    # Polish the best result
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        S_polished, cx_p, cy_p, beta_p = _polish(best_global_pts, best_global_beta, best_global_cx, best_global_cy)
        if S_polished < best_global_S:
            best_global_S = S_polished
            best_global_beta = beta_p
            best_global_cx = cx_p
            best_global_cy = cy_p

    # Final LP center
    D, cx_f, cy_f = _min_enclosing_side(best_global_pts, best_global_beta)
    if D < float('inf'):
        S_final = (D + APOTHEM) / APOTHEM
        if S_final < best_global_S:
            best_global_S = S_final
            best_global_beta = best_global_beta
            best_global_cx = cx_f
            best_global_cy = cy_f

    cx, cy = best_global_cx, best_global_cy
    angle_deg = math.degrees(best_global_beta)
    
    # Scale very slightly to ensure no floating point overlap issues
    SCALE = 1.0 + 1e-10
    pts_scaled = best_global_pts.copy()
    pts_scaled[:,0] = cx + (pts_scaled[:,0] - cx) * SCALE
    pts_scaled[:,1] = cy + (pts_scaled[:,1] - cy) * SCALE

    # Recompute exact side with scaled points
    max_proj = 0.0
    normals = _hex_normals(best_global_beta)
    for nx, ny in normals:
        projs = (pts_scaled[:,0]-cx)*nx + (pts_scaled[:,1]-cy)*ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
    side = (max_proj + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    return inner, center, float(side), float(angle_deg)
