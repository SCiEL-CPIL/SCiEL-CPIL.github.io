"""Heilbronn triangles: place 11 points to maximize the minimum triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4
DEADLINE = 145.0

# Precompute all triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

def _bary_to_cart(abc):
    """(a,b,c) -> (x,y)."""
    x = abc[:, 1] + 0.5 * abc[:, 2]
    y = abc[:, 2] * S3H
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """(x,y) -> (a,b,c)."""
    c = pts[:, 1] / S3H
    b = pts[:, 0] - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])

def _project_bary(abc):
    """Project barycentric coords to valid simplex (a,b,c >= 0, sum=1)."""
    abc = np.maximum(abc, 0)
    s = abc.sum(axis=1, keepdims=True)
    s = np.maximum(s, 1e-15)
    return abc / s

def _all_areas(pts):
    """All 165 triangle areas."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return np.abs(cross) * 0.5

def _min_area(pts):
    return np.min(_all_areas(pts))

def _score(pts):
    return _min_area(pts) / TRI_AREA

def _soft_min(areas, beta):
    m = np.min(areas)
    shifted = beta * (areas - m)
    shifted = np.clip(shifted, 0, 700)
    logsum = m * beta + np.log(np.sum(np.exp(shifted)))
    return logsum / beta

def _bary_obj(flat, beta=200.0):
    """Objective: minimize -soft_min_area in barycentric (b,c) space."""
    bc = flat.reshape(N, 2)
    a = 1.0 - bc[:, 0] - bc[:, 1]
    abc = np.column_stack([a, bc[:, 0], bc[:, 1]])
    abc = _project_bary(abc)
    pts = _bary_to_cart(abc)
    areas = _all_areas(pts)
    sm = _soft_min(areas, beta)
    return -sm

def _hard_obj(flat):
    """Objective: minimize -min_area in barycentric (b,c) space."""
    bc = flat.reshape(N, 2)
    a = 1.0 - bc[:, 0] - bc[:, 1]
    abc = np.column_stack([a, bc[:, 0], bc[:, 1]])
    abc = _project_bary(abc)
    pts = _bary_to_cart(abc)
    return -_min_area(pts)

def _random_bary(rng):
    """Random point in barycentric simplex."""
    u = rng.random(N)
    v = rng.random(N)
    w = 1.0 - u - v
    abc = np.column_stack([u, v, w])
    abc = _project_bary(abc)
    return abc

def _random_cart(rng):
    """Random point in Cartesian triangle."""
    abc = _random_bary(rng)
    return _bary_to_cart(abc)

def _nm_obj(flat):
    """Nelder-Mead objective in Cartesian space."""
    pts = flat.reshape(N, 2)
    bary = _cart_to_bary(pts)
    penalty = 100.0 * np.sum(np.maximum(0, -bary)**2)
    return -_score(pts) + penalty

def _push_apart(pts, n_steps=500, rng=None):
    """
    Local search: repeatedly find the minimum-area triangle and push
    its vertices apart along the direction that increases the area.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    
    cur = pts.copy()
    cur_score = _score(cur)
    best = cur.copy()
    best_score = cur_score
    
    for step in range(n_steps):
        areas = _all_areas(cur)
        min_idx = np.argmin(areas)
        min_val = areas[min_idx]
        
        i, j, k = _TRIPLETS[min_idx]
        
        # Direction to increase area of triangle (i,j,k)
        # Area = 0.5 * |cross(pj-pi, pk-pi)|
        # Push i away from jk, j away from ik, k away from ij
        pi, pj, pk = cur[i], cur[j], cur[k]
        
        # Centroid of the triangle
        cent = (pi + pj + pk) / 3.0
        
        # For each vertex, push away from the other two
        # Direction for i: away from midpoint of j,k
        mid_jk = (pj + pk) / 2.0
        dir_i = pi - mid_jk
        norm_i = np.linalg.norm(dir_i)
        if norm_i > 1e-12:
            dir_i /= norm_i
        
        mid_ik = (pi + pk) / 2.0
        dir_j = pj - mid_ik
        norm_j = np.linalg.norm(dir_j)
        if norm_j > 1e-12:
            dir_j /= norm_j
        
        mid_ij = (pi + pj) / 2.0
        dir_k = pk - mid_ij
        norm_k = np.linalg.norm(dir_k)
        if norm_k > 1e-12:
            dir_k /= norm_k
        
        # Try several step sizes
        improved = False
        for alpha in [0.02, 0.01, 0.005, 0.002, 0.001]:
            cand = cur.copy()
            cand[i] = pi + alpha * dir_i
            cand[j] = pj + alpha * dir_j
            cand[k] = pk + alpha * dir_k
            
            # Project to triangle
            bary = _cart_to_bary(cand)
            bary = _project_bary(bary)
            cand = _bary_to_cart(bary)
            
            sc = _score(cand)
            if sc > cur_score + 1e-12:
                cur = cand
                cur_score = sc
                improved = True
                if sc > best_score:
                    best_score = sc
                    best = cur.copy()
                break
        
        if not improved:
            # Try random perturbation
            idx = rng.randint(N)
            cand = cur.copy()
            dx = rng.normal(0, 0.01)
            dy = rng.normal(0, 0.01)
            cand[idx, 0] += dx
            cand[idx, 1] += dy
            bary = _cart_to_bary(cand)
            bary = _project_bary(bary)
            cand = _bary_to_cart(bary)
            sc = _score(cand)
            if sc > cur_score:
                cur = cand
                cur_score = sc
                if sc > best_score:
                    best_score = sc
                    best = cur.copy()
    
    return best, best_score

def _sa_bary(abc, n_iter=2000, T0=0.03, T1=0.001, rng=None):
    """Simulated annealing in barycentric space."""
    if rng is None:
        rng = np.random.RandomState(1)
    
    cur = abc.copy()
    cur_score = _score(_bary_to_cart(cur))
    best = cur.copy()
    best_score = cur_score
    
    for i in range(n_iter):
        T = T0 * (T1 / T0) ** (i / n_iter)
        
        idx = rng.randint(N)
        cand = cur.copy()
        step = 0.03 * (1.0 + 0.5 * (T / T0))
        # Perturb two coordinates (a is dependent)
        db = rng.normal(0, step)
        dc = rng.normal(0, step)
        cand[idx, 1] += db
        cand[idx, 2] += dc
        cand = _project_bary(cand)
        
        sc = _score(_bary_to_cart(cand))
        
        if sc >= cur_score:
            cur = cand
            cur_score = sc
            if sc > best_score:
                best_score = sc
                best = cur.copy()
        else:
            delta = cur_score - sc
            if delta < 1e-12:
                cur = cand
                cur_score = sc
            else:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    cur = cand
                    cur_score = sc
    
    return best, best_score

def _nm_optimize(pts, maxiter=2000):
    """Nelder-Mead in Cartesian space."""
    try:
        res = minimize(
            _nm_obj,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        polished = res.x.reshape(N, 2)
        bary = _cart_to_bary(polished)
        if np.all(bary >= -1e-9):
            sc = _score(polished)
            return polished, sc
    except Exception:
        pass
    return pts, _score(pts)

def _powell_optimize(abc, maxiter=300):
    """Powell in barycentric (b,c) space."""
    bc = np.column_stack([abc[:, 1], abc[:, 2]]).flatten()
    try:
        res = minimize(
            _hard_obj,
            bc,
            method='Powell',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        bc_res = res.x.reshape(N, 2)
        a = 1.0 - bc_res[:, 0] - bc_res[:, 1]
        abc_res = np.column_stack([a, bc_res[:, 0], bc_res[:, 1]])
        abc_res = _project_bary(abc_res)
        pts = _bary_to_cart(abc_res)
        sc = _score(pts)
        return pts, sc
    except Exception:
        pass
    return _bary_to_cart(abc), _score(_bary_to_cart(abc))

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    best_pts = None
    best_score = -float('inf')
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Centroid + perturbation
    abc0 = np.zeros((N, 3))
    abc0[:, 0] = 1.0 / N
    abc0[:, 1] = 1.0 / N
    abc0[:, 2] = 1.0 - 2.0 / N
    # Perturb
    for i in range(N):
        abc0[i, 1] += rng.normal(0, 0.1)
        abc0[i, 2] += rng.normal(0, 0.1)
    abc0 = _project_bary(abc0)
    inits.append(abc0)
    
    # 2. Grid-based
    grid_pts = []
    for i in range(5):
        for j in range(5):
            t = i / 4.0
            s = j / 4.0
            x = t * (1.0 - s) + s * 0.5
            y = s * S3H
            if x >= -1e-9 and y >= -1e-9 and (1.0 - x - y/S3H) >= -1e-9:
                grid_pts.append([x, y])
    grid_pts = np.array(grid_pts)
    if len(grid_pts) >= N:
        idx = rng.choice(len(grid_pts), N, replace=False)
        inits.append(_cart_to_bary(grid_pts[idx]))
    
    # 3. Random
    for _ in range(3):
        inits.append(_random_bary(rng))
    
    # 4. Vertices + centroid + edges
    verts = np.array([[0, 0], [1, 0], [0.5, S3H]])
    cent = np.array([0.5, S3H / 3])
    edge_mids = np.array([[0.25, S3H * 0.25], [0.75, S3H * 0.25], [0.5, S3H * 0.5]])
    special = np.vstack([verts, cent, edge_mids])
    if len(special) >= N:
        inits.append(_cart_to_bary(special))
    
    # 5. FPS-like
    fps_pts = [_random_cart(rng)]
    for _ in range(N - 1):
        best_d = -1
        best_p = None
        for _ in range(200):
            cand = _random_cart(rng)
            d = np.min(np.linalg.norm(np.array(fps_pts) - cand, axis=1))
            if d > best_d:
                best_d = d
                best_p = cand
        fps_pts.append(best_p)
    inits.append(_cart_to_bary(np.array(fps_pts)))
    
    # For each initialization, run a pipeline: SA -> push_apart -> Powell -> NM
    for abc_init in inits:
        if elapsed() > DEADLINE - 25:
            break
        
        try:
            # Step 1: Simulated Annealing
            abc_sa, sa_score = _sa_bary(abc_init, n_iter=1500, T0=0.03, T1=0.001, rng=rng)
            pts_sa = _bary_to_cart(abc_sa)
            
            if sa_score > best_score:
                best_score = sa_score
                best_pts = pts_sa.copy()
            
            if elapsed() > DEADLINE - 20:
                break
            
            # Step 2: Push-apart local search
            pts_pa, pa_score = _push_apart(pts_sa, n_steps=300, rng=rng)
            
            if pa_score > best_score:
                best_score = pa_score
                best_pts = pts_pa.copy()
            
            if elapsed() > DEADLINE - 15:
                break
            
            # Step 3: Powell in barycentric
            abc_pa = _cart_to_bary(pts_pa)
            pts_pow, pow_score = _powell_optimize(abc_pa, maxiter=200)
            
            if pow_score > best_score:
                best_score = pow_score
                best_pts = pts_pow.copy()
            
            if elapsed() > DEADLINE - 10:
                break
            
            # Step 4: Nelder-Mead polish
            pts_nm, nm_score = _nm_optimize(pts_pow, maxiter=1500)
            
            if nm_score > best_score:
                best_score = nm_score
                best_pts = pts_nm.copy()
                
        except Exception:
            continue
    
    # Final polish on best
    if best_pts is not None and elapsed() < DEADLINE - 5:
        try:
            pts_final, sc_final = _nm_optimize(best_pts, maxiter=2000)
            if sc_final > best_score:
                best_score = sc_final
                best_pts = pts_final.copy()
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _bary_to_cart(_random_bary(rng))
    
    return best_pts
