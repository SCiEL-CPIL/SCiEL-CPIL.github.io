import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 16
D = 2
DEADLINE = 140.0

def _pairwise_dists(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    return d2[iu]

def _soft_objective(params, temp=0.01):
    pts = params.reshape(N, D)
    d2 = _pairwise_dists(pts)
    d2 = np.maximum(d2, 1e-12)
    m = np.max(d2)
    soft_max = m + temp * np.log(np.sum(np.exp((d2 - m) / temp)))
    m2 = np.min(d2)
    soft_min = m2 - temp * np.log(np.sum(np.exp(-(d2 - m2) / temp)))
    return soft_max / soft_min

def _exact_score(pts):
    d2 = _pairwise_dists(pts)
    d2 = np.maximum(d2, 1e-15)
    return np.max(d2) / np.min(d2)

def _ring_init(k, radius, offset=0.0):
    """Place k points on a circle of given radius with angular offset."""
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + offset
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _hex_init():
    """Hexagonal packing initialization for 16 points."""
    pts = []
    dx = 1.0
    dy = np.sqrt(3) / 2
    for i in range(4):
        for j in range(4):
            x = i * dx + (j % 2) * dx / 2
            y = j * dy
            pts.append([x, y])
    pts = np.array(pts[:16])
    # Center and normalize
    pts -= pts.mean(axis=0)
    max_r = np.max(np.sqrt(np.sum(pts ** 2, axis=1)))
    if max_r > 0:
        pts /= max_r
    return pts

def _random_init():
    """Random initialization within unit disk."""
    rng = np.random.RandomState(42)
    angles = rng.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(rng.uniform(0.2, 1.0, N))
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _optimize_from(pts, time_limit=10.0):
    """Multi-stage optimization starting from initial points."""
    params = pts.flatten()
    start = time.time()

    # Stage 1: L-BFGS with soft objective
    try:
        res = minimize(_soft_objective, params, args=(0.1,), method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
        if res.fun < 1e10:
            params = res.x
    except Exception:
        pass

    if time.time() - start > time_limit * 0.5:
        return params.reshape(N, 2)

    # Stage 2: L-BFGS with tighter soft objective
    try:
        res = minimize(_soft_objective, params, args=(0.01,), method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.fun < 1e10:
            params = res.x
    except Exception:
        pass

    if time.time() - start > time_limit * 0.8:
        return params.reshape(N, 2)

    # Stage 3: Final L-BFGS polish
    try:
        res = minimize(_soft_objective, params, args=(0.001,), method='L-BFGS-B',
                       options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12})
        if res.fun < 1e10:
            params = res.x
    except Exception:
        pass

    return params.reshape(N, 2)

def construct():
    t0 = time.time()
    
    # Initial candidates
    candidates = []
    
    # 1. Standard 4x4 Grid
    gx = np.array([0, 1, 2, 3], dtype=float)
    gy = np.array([0, 1, 2, 3], dtype=float)
    X, Y = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([X.flatten(), Y.flatten()])
    candidates.append(grid_pts)
    
    # 2. Compressed 4x4 Grid (corners moved in)
    comp_pts = grid_pts.copy()
    center = np.array([1.5, 1.5])
    for idx in [0, 3, 12, 15]:
        comp_pts[idx] = comp_pts[idx] + 0.3 * (center - comp_pts[idx])
    candidates.append(comp_pts)
    
    # 3. Triangular Lattice Rhombus
    tri_pts = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0.5, np.sqrt(3)/2], [1.5, np.sqrt(3)/2], [2.5, np.sqrt(3)/2], [3.5, np.sqrt(3)/2],
        [1, np.sqrt(3)], [2, np.sqrt(3)], [3, np.sqrt(3)], [4, np.sqrt(3)],
        [1.5, 1.5*np.sqrt(3)], [2.5, 1.5*np.sqrt(3)], [3.5, 1.5*np.sqrt(3)], [4.5, 1.5*np.sqrt(3)]
    ])
    candidates.append(tri_pts)
    
    # 4. Random perturbations of Grid
    rng = np.random.RandomState(42)
    for i in range(10):
        perturbed = grid_pts + rng.randn(N, D) * 0.1
        candidates.append(perturbed)
    
    # 5. Ring-based initializations
    for r1, r2 in [(0.4, 0.9), (0.3, 0.85), (0.5, 0.95), (0.35, 0.8)]:
        inner = _ring_init(8, r1)
        outer = _ring_init(8, r2, offset=np.pi/8)
        pts = np.vstack([inner, outer])
        candidates.append(pts)
    
    # 6. Hexagonal
    candidates.append(_hex_init())
    
    # 7. Random starts
    for _ in range(5):
        candidates.append(_random_init())
    
    # 8. Single ring (all 16 on one circle)
    candidates.append(_ring_init(16, 1.0))
    
    # 9. Two rings with different splits
    for n1, n2 in [(4, 12), (6, 10), (8, 8), (10, 6), (12, 4)]:
        inner = _ring_init(n1, 0.4)
        outer = _ring_init(n2, 0.9, offset=np.pi/n2)
        pts = np.vstack([inner, outer])
        candidates.append(pts)

    best_pts = None
    best_score = np.inf

    for i, init_pts in enumerate(candidates):
        if time.time() - t0 > DEADLINE - 10:
            break
        
        try:
            # Use multi-stage optimization
            time_per_init = min(12.0, (DEADLINE - (time.time() - t0)) / max(1, len(candidates) - i))
            optimized = _optimize_from(init_pts, time_limit=time_per_init)
            score = _exact_score(optimized)
            
            if score < best_score:
                best_score = score
                best_pts = optimized.copy()
                
        except Exception:
            continue

    # Final polish on the best
    if best_pts is not None and time.time() - t0 < DEADLINE - 2:
        try:
            res_final = minimize(
                lambda p: _soft_objective(p, temp=0.001),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            pts_final = res_final.x.reshape(N, D)
            score_final = _exact_score(pts_final)
            if score_final < best_score:
                best_pts = pts_final
        except Exception:
            pass

    if best_pts is None:
        best_pts = grid_pts

    # Verify distinctness
    dists = _pairwise_dists(best_pts)
    if np.min(dists) < 1e-8:
        # Add tiny perturbation to separate any coincident points
        best_pts = best_pts + np.random.normal(0, 1e-6, best_pts.shape)

    return best_pts

if __name__ == '__main__':
    pts = construct()
    print("Score:", _exact_score(pts))
