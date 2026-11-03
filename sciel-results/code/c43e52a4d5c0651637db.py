import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7

# Precompute all C(14,3) = 364 triangle index triples
_idx = np.arange(N)
_II, _JJ, _KK = np.meshgrid(_idx, _idx, _idx, indexing='ij')
_mask = (_II < _JJ) & (_JJ < _KK)
_I = _II[_mask].ravel()
_J = _JJ[_mask].ravel()
_K = _KK[_mask].ravel()

def _min_area(pts):
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return np.min(areas)

def _hull_area(pts):
    try:
        if np.allclose(pts[:, 0], pts[0, 0]) or np.allclose(pts[:, 1], pts[0, 1]):
            return 0.0
        h = ConvexHull(pts)
        return h.volume
    except Exception:
        return 0.0

def _score(pts):
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    return _min_area(pts) / ha

def _sym_pts(half):
    return np.vstack([half, -half])

def _sym_score(half):
    return _score(_sym_pts(half))

def _objective_sym(flat):
    half = flat.reshape(N_HALF, 2)
    if np.any(~np.isfinite(half)):
        return 1e10
    s = _sym_score(half)
    return -s

def _soft_objective_sym(flat, beta=30.0):
    half = flat.reshape(N_HALF, 2)
    if np.any(~np.isfinite(half)):
        return 1e10
    pts = _sym_pts(half)
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    mz = np.max(z)
    if mz < -500:
        log_sum = mz
    else:
        log_sum = mz + np.log(np.sum(np.exp(z - mz)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e10
    return ha / soft_min

def _push_pull(half, max_iter=500, tol=1e-9):
    """Push-pull local search: identify worst triangle, push its vertices outward."""
    best = half.copy()
    best_s = _sym_score(best)
    for it in range(max_iter):
        pts = _sym_pts(best)
        pi = pts[_I]
        pj = pts[_J]
        pk = pts[_K]
        v1x = pj[:, 0] - pi[:, 0]
        v1y = pj[:, 1] - pi[:, 1]
        v2x = pk[:, 0] - pi[:, 0]
        v2y = pk[:, 1] - pi[:, 1]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        worst_idx = np.argmin(areas)
        wi, wj, wk = _I[worst_idx], _J[worst_idx], _K[worst_idx]
        
        # Push the three vertices of the worst triangle outward (away from centroid)
        centroid = np.mean(pts, axis=0)
        for idx in [wi, wj, wk]:
            # Map global index to half index
            if idx < N_HALF:
                hi = idx
            else:
                hi = idx - N_HALF
            # Direction: away from centroid
            p = best[hi]
            c = centroid if idx < N_HALF else -centroid
            direction = p - c
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction /= norm
            else:
                direction = np.array([1.0, 0.0])
            
            # Try several step sizes
            improved = False
            for step in [0.05, 0.02, 0.01, 0.005, 0.002]:
                new_half = best.copy()
                new_half[hi] = best[hi] + step * direction
                new_s = _sym_score(new_half)
                if new_s > best_s + 1e-12:
                    best = new_half
                    best_s = new_s
                    improved = True
                    break
            if not improved:
                # Try pulling inward slightly (sometimes helps with hull area)
                for step in [0.02, 0.01, 0.005]:
                    new_half = best.copy()
                    new_half[hi] = best[hi] - step * direction
                    new_s = _sym_score(new_half)
                    if new_s > best_s + 1e-12:
                        best = new_half
                        best_s = new_s
                        improved = True
                        break
        
        if not improved:
            # Try small random perturbations
            rng = np.random.default_rng(it)
            improved = False
            for _ in range(20):
                new_half = best + rng.normal(0, 0.01, size=(N_HALF, 2))
                new_s = _sym_score(new_half)
                if new_s > best_s + 1e-12:
                    best = new_half
                    best_s = new_s
                    improved = True
                    break
        
        if not improved:
            break
    
    return best, best_s

def _nm_optimize(flat_init, maxiter=2000):
    try:
        res = minimize(_objective_sym, flat_init, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        if res.fun < 0:
            half = res.x.reshape(N_HALF, 2)
            s = -res.fun
            return half, s
    except Exception:
        pass
    return None, -np.inf

def _powell_optimize(flat_init, maxiter=1000):
    try:
        res = minimize(_objective_sym, flat_init, method='Powell',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'ftol': 1e-8})
        if res.fun < 0:
            half = res.x.reshape(N_HALF, 2)
            s = -res.fun
            return half, s
    except Exception:
        pass
    return None, -np.inf

def _lbfgs_soft_optimize(flat_init, maxiter=3000, beta=30.0):
    try:
        res = minimize(lambda x: _soft_objective_sym(x, beta=beta), flat_init, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
        half = res.x.reshape(N_HALF, 2)
        s = _sym_score(half)
        return half, s
    except Exception:
        pass
    return None, -np.inf

def construct():
    start = time.time()
    deadline = start + 135.0  # 15s safety margin before 150s kill
    
    best_half = None
    best_score = -np.inf
    
    def update(half, score):
        nonlocal best_half, best_score
        if score > best_score:
            best_score = score
            best_half = half.copy()
    
    def remaining():
        return deadline - time.time()
    
    # Generate diverse initializations for the 7-point half
    inits = []
    
    # Regular heptagon
    a7 = np.linspace(0, 2*np.pi, N_HALF, endpoint=False)
    inits.append(np.column_stack([np.cos(a7), np.sin(a7)]))
    
    # Perturbed regular heptagon
    rng = np.random.default_rng(42)
    for _ in range(5):
        pert = np.column_stack([np.cos(a7), np.sin(a7)]) + rng.normal(0, 0.15, size=(N_HALF, 2))
        inits.append(pert)
    
    # Elliptical heptagons
    for rx, ry in [(1.0, 0.7), (1.1, 0.8), (0.9, 0.85), (1.2, 0.6), (1.0, 0.5)]:
        inits.append(np.column_stack([rx*np.cos(a7), ry*np.sin(a7)]))
    
    # Two-cluster: 3 points near one side, 4 near the other
    for r1, r2, off in [(0.3, 0.8, 0.0), (0.4, 0.7, 0.3), (0.5, 0.6, 0.5)]:
        a1 = np.linspace(-0.5, 0.5, 3) + off
        a2 = np.linspace(np.pi-0.5, np.pi+0.5, 4) + off
        p1 = r1 * np.column_stack([np.cos(a1), np.sin(a1)])
        p2 = r2 * np.column_stack([np.cos(a2), np.sin(a2)])
        inits.append(np.vstack([p1, p2]))
    
    # Random points in disk
    for _ in range(4):
        r = np.sqrt(rng.uniform(0.1, 1.0, N_HALF))
        theta = rng.uniform(0, 2*np.pi, N_HALF)
        inits.append(np.column_stack([r*np.cos(theta), r*np.sin(theta)]))
    
    # Halton-like in 7D mapped to 2D
    def halton_1d(i, base):
        f = 1.0
        r = 0.0
        while i:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    hp = np.array([[halton_1d(i, 2), halton_1d(i, 3)] for i in range(1, N_HALF+1)])
    inits.append(hp)
    
    # Phase 1: Nelder-Mead + Push-Pull on each init
    for init in inits:
        if remaining() < 20.0:
            break
        flat = init.flatten()
        
        # Nelder-Mead
        half_nm, s_nm = _nm_optimize(flat, maxiter=1500)
        if half_nm is not None:
            update(half_nm, s_nm)
        
        # Push-Pull from best so far
        if best_half is not None and remaining() > 15.0:
            half_pp, s_pp = _push_pull(best_half, max_iter=300)
            update(half_pp, s_pp)
        
        # Powell refinement
        if remaining() > 15.0:
            half_pw, s_pw = _powell_optimize(flat, maxiter=500)
            if half_pw is not None:
                update(half_pw, s_pw)
    
    # Phase 2: Soft-min L-BFGS-B from multiple starting points
    if remaining() > 15.0:
        start_points = []
        if best_half is not None:
            start_points.append(best_half.flatten())
        start_points.append(inits[0].flatten())
        start_points.append(inits[1].flatten())
        
        for sp in start_points:
            if remaining() < 10.0:
                break
            half_soft, s_soft = _lbfgs_soft_optimize(sp, maxiter=2000, beta=30.0)
            if half_soft is not None:
                update(half_soft, s_soft)
            
            # Follow up with hard Nelder-Mead
            if remaining() > 10.0:
                half_nm2, s_nm2 = _nm_optimize(half_soft.flatten() if half_soft is not None else sp, maxiter=1000)
                if half_nm2 is not None:
                    update(half_nm2, s_nm2)
    
    # Phase 3: Iterative Push-Pull with increasing aggressiveness
    if best_half is not None and remaining() > 10.0:
        for _ in range(3):
            if remaining() < 8.0:
                break
            half_pp, s_pp = _push_pull(best_half, max_iter=400)
            update(half_pp, s_pp)
            
            # NM refinement after push-pull
            if remaining() > 8.0:
                half_nm3, s_nm3 = _nm_optimize(best_half.flatten(), maxiter=1000)
                if half_nm3 is not None:
                    update(half_nm3, s_nm3)
    
    # Phase 4: Final polish
    if best_half is not None and remaining() > 5.0:
        half_final, s_final = _push_pull(best_half, max_iter=500)
        update(half_final, s_final)
        
        if remaining() > 3.0:
            half_nm4, s_nm4 = _nm_optimize(best_half.flatten(), maxiter=800)
            if half_nm4 is not None:
                update(half_nm4, s_nm4)
    
    if best_half is None:
        best_half = inits[0]
    
    return _sym_pts(best_half)
