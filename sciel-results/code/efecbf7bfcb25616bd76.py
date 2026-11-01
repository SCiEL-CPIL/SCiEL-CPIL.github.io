"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fib_sample(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _hull_data(p):
    """Facets, outward circumcenters, and circumradii (radians) of the convex hull."""
    try:
        hull = ConvexHull(p)
        facets = hull.simplices
    except Exception:
        return None, None, None
    n_f = len(facets)
    if n_f == 0:
        return None, None, None
    a = p[facets[:, 0]]
    b = p[facets[:, 1]]
    c = p[facets[:, 2]]
    n = np.cross(a - b, b - c)
    nn = np.linalg.norm(n, axis=1)
    valid = nn > 1e-12
    cc = np.zeros((n_f, 3))
    rad = np.zeros(n_f)
    if np.any(valid):
        ccv = n[valid] / nn[valid, None]
        d = np.sum(ccv * a[valid], axis=1)
        ccv[d < 0] *= -1
        cc[valid] = ccv
        rad[valid] = np.arccos(np.clip(np.sum(ccv * a[valid], axis=1), -1.0, 1.0))
    return facets, cc, rad

def _covering_radius_deg(p):
    """Exact covering radius in degrees via hull facets; sampling fallback."""
    _, _, rad = _hull_data(p)
    if rad is None or not np.any(rad > 0):
        g = _fib_sample(20000)
        dots = g @ p.T
        return float(np.degrees(np.arccos(np.clip(np.min(dots, axis=1), -1.0, 1.0)).max()))
    return float(np.degrees(rad.max()))

def _soft_max(p, beta=100.0):
    """Smooth surrogate for the max circumradius; 0 if hull degenerate."""
    X = _normalize(p.reshape(N, 3))
    _, _, rad = _hull_data(X)
    if rad is None or not np.any(rad > 0):
        return 0.0
    rmax = rad.max()
    return rmax + np.log(np.sum(np.exp(beta * (rad - rmax)))) / beta

def _repulsion(points, alpha=0.05, decay=0.995, max_iter=200, deadline=None):
    points = points.copy()
    mask = np.eye(N, dtype=bool)
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        diff = points[:, None, :] - points[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        inv = np.zeros_like(d2)
        np.divide(1.0, d2, out=inv, where=~mask)
        forces = np.sum(diff * inv[:, :, None], axis=1)
        forces = forces - np.sum(forces * points, axis=1)[:, None] * points
        points = _normalize(points + alpha * forces)
        alpha *= decay
    return points

def _lloyd(config):
    try:
        hull = ConvexHull(config)
        facets = hull.simplices
    except Exception:
        return config
    a = config[facets[:, 0]]
    b = config[facets[:, 1]]
    c = config[facets[:, 2]]
    n = np.cross(a - b, b - c)
    nn = np.linalg.norm(n, axis=1)
    valid = nn > 1e-12
    if not np.any(valid):
        return config
    cc = n[valid] / nn[valid, None]
    cc[np.sum(cc * a[valid], axis=1) < 0] *= -1
    nearest = np.argmax(cc @ config.T, axis=1)
    sums = np.zeros((N, 3))
    counts = np.zeros(N)
    np.add.at(sums, nearest, cc)
    np.add.at(counts, nearest, 1.0)
    new = config.copy()
    for i in range(N):
        if counts[i] > 0 and np.linalg.norm(sums[i]) > 1e-10:
            new[i] = sums[i] / np.linalg.norm(sums[i])
    return new

def _lbfqgs(config, deadline):
    """L-BFGS on the soft-max surrogate (beta annealed), then a low-beta polish."""
    best = config.copy()
    best_score = _covering_radius_deg(config)
    
    # More granular beta annealing to navigate landscape better
    for beta in (50.0, 80.0, 120.0, 200.0, 300.0):
        if time.time() > deadline - 8.0:
            break
        res = minimize(_soft_max, config.ravel(), args=(beta,), method='L-BFGS-B',
                       options={'maxiter': 400, 'maxls': 15, 'ftol': 1e-12,
                                'gtol': 1e-5, 'disp': False})
        cand = _normalize(res.x.reshape(N, 3))
        s = _covering_radius_deg(cand)
        if s < best_score - 1e-9:
            best, best_score = cand.copy(), s
            config = cand.copy()
    
    if time.time() < deadline - 8.0:
        res = minimize(_soft_max, config.ravel(), args=(30.0,), method='L-BFGS-B',
                       options={'maxiter': 300, 'maxls': 15, 'ftol': 1e-12,
                                'gtol': 1e-6, 'disp': False})
        cand = _normalize(res.x.reshape(N, 3))
        s = _covering_radius_deg(cand)
        if s < best_score - 1e-9:
            best, best_score = cand.copy(), s
    return best, best_score

def _greedy_polish(config, deadline):
    """Move the vertices of the worst facet toward its circumcenter (line search)."""
    config = config.copy()
    best_score = _covering_radius_deg(config)
    for _ in range(400):
        if time.time() > deadline - 3.0:
            break
        facets, cc, rad = _hull_data(config)
        if facets is None:
            break
        k = int(np.argmax(rad))
        target = cc[k]
        if np.linalg.norm(target) < 1e-8:
            break
        improved = False
        for t in (0.05, 0.02, 0.01, 0.005):
            trial = config.copy()
            for v in facets[k]:
                nv = config[v] + t * (target - config[v])
                trial[v] = nv / (np.linalg.norm(nv) + 1e-15)
            s = _covering_radius_deg(trial)
            if s < best_score - 1e-9:
                config, best_score, improved = trial, s, True
                break
        if not improved:
            break
    return config, best_score

def _spectral_init(rng, deadline):
    """
    Initialize points by minimizing a spectral energy:
    Sum over random directions k of |Sum exp(i k . x_i)|^2.
    This penalizes clumping at the scale of 1/|k|.
    """
    # Start with Fibonacci as a good baseline
    pts = _fib_sample(N).copy()
    
    # Generate random directions for the spectral modes
    # High frequency -> local uniformity. Low frequency -> global uniformity.
    # We want to avoid large gaps, so we focus on medium/high frequencies.
    K = 20 
    # Random unit vectors
    k_vecs = rng.normal(size=(K, 3))
    k_vecs /= np.linalg.norm(k_vecs, axis=1, keepdims=True)
    
    def spectral_energy(params):
        x = params.reshape(N, 3)
        x = _normalize(x)
        # Compute phases: (K, N)
        phases = k_vecs @ x.T
        # Complex exponentials: (K, N)
        exps = np.exp(1j * phases)
        # Sum over points for each mode: (K,)
        sums = np.sum(exps, axis=1)
        # Energy is sum of squared magnitudes
        return np.sum(np.abs(sums)**2)
    
    # Optimize the spectral energy
    # Use L-BFGS-B
    res = minimize(spectral_energy, pts.ravel(), method='L-BFGS-B',
                   options={'maxiter': 200, 'maxls': 50, 'ftol': 1e-8, 'gtol': 1e-5, 'disp': False})
    
    if time.time() > deadline:
        return _normalize(pts)
        
    return _normalize(res.x.reshape(N, 3))

def _geometric_refine(config, deadline):
    """Apply the full geometric optimization pipeline."""
    current = config.copy()
    cur_score = _covering_radius_deg(current)
    
    # Phase 1: Repulsion
    if time.time() < deadline - 40.0:
        current = _repulsion(current, max_iter=100, deadline=deadline)
        s = _covering_radius_deg(current)
        if s < cur_score:
            cur_score = s

    # Phase 2: Lloyd
    if time.time() < deadline - 20.0:
        for _ in range(20):
            if time.time() > deadline - 20.0:
                break
            new = _lloyd(current)
            s = _covering_radius_deg(new)
            if s < cur_score:
                cur_score = s
            current = new

    # Phase 3: L-BFGS
    if time.time() < deadline - 10.0:
        current, s = _lbfqgs(current, deadline)
        if s < cur_score:
            cur_score = s

    # Phase 4: Greedy Polish
    if time.time() < deadline - 2.0:
        current, s = _greedy_polish(current, deadline)
        if s < cur_score:
            cur_score = s
            
    return current, cur_score

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before the 120s kill
    
    best = None
    best_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # We will try a few different spectral initializations and one standard fib start.
    # The spectral init is a structural change: it uses a smooth proxy to find a "uniform" start.
    
    starts = []
    
    # Start 0: Standard Fibonacci (baseline)
    starts.append(_fib_sample(N))
    
    # Starts 1-3: Spectral Initializations with different random seeds/directions
    for i in range(3):
        if time.time() > deadline - 60.0:
            break
        # Each spectral init takes a few seconds
        spec_pts = _spectral_init(rng, deadline)
        starts.append(spec_pts)
        
    # Also add a random start perturbed
    if time.time() < deadline - 60.0:
        rand_pts = rng.normal(size=(N, 3))
        starts.append(_normalize(rand_pts))

    for start_pts in starts:
        if time.time() > deadline - 20.0:
            break
        
        # Refine each start
        cand, score = _geometric_refine(start_pts, deadline)
        
        if score < best_score:
            best_score = score
            best = cand.copy()
            
    if best is None:
        best = _normalize(_fib_sample(N))
        
    return _normalize(best)
# EVOLVE-BLOCK-END
