"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.transform import Rotation

N = 14
D = 3

# EVOLVE-BLOCK-START

def get_pairwise_sq_dists(points):
    """Compute squared pairwise distances as flat array of length N*(N-1)/2."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Actual score: (max_dist / min_dist)^2 = max_sq_dist / min_sq_dist."""
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return 1e10
    return np.max(sq_dists) / np.min(sq_dists)

def _normalize_points(points):
    """Center and normalize so mean squared distance = 1."""
    pts = points - np.mean(points, axis=0, keepdims=True)
    sq_dists = get_pairwise_sq_dists(pts)
    mean_sq = np.mean(sq_dists)
    if mean_sq < 1e-12:
        return pts
    return pts / np.sqrt(mean_sq)

def objective_ratio(x, beta=10.0):
    """
    Smooth objective: minimize (soft_max_sq_dist / soft_min_sq_dist) using 
    a power-based soft min/max. This is more numerically stable than log-sum-exp
    for the ratio directly.
    """
    points = x.reshape(N, D)
    points = _normalize_points(points)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Power-based soft max: (sum(d^p))^(1/p)
    p = beta
    soft_max = (np.sum(sq_dists ** p)) ** (1.0 / p)
    
    # Power-based soft min: (sum(d^-p))^(-1/p)
    soft_min = (np.sum(sq_dists ** (-p))) ** (-1.0 / p)
    
    return soft_max / soft_min

def objective_raw(x):
    """Raw objective for final polish."""
    points = x.reshape(N, D)
    points = _normalize_points(points)
    return raw_score(points)

def _fibonacci_sphere(n, rng, rotation=None):
    """Generate n points on a sphere using Fibonacci lattice."""
    golden = (1 + np.sqrt(5)) / 2
    pts = []
    for k in range(n):
        z = 1 - 2 * k / (n - 1)
        r = np.sqrt(1 - z ** 2)
        theta = 2 * np.pi * k / golden
        pts.append([r * np.cos(theta), r * np.sin(theta), z])
    pts = np.array(pts)
    if rotation is not None:
        pts = rotation.apply(pts)
    return pts

def _icosahedron_plus(rng):
    """12 icosahedron vertices + 2 random points."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    p1 = rng.normal(size=D); p1 /= np.linalg.norm(p1)
    p2 = rng.normal(size=D); p2 /= np.linalg.norm(p2)
    pts = np.vstack([ico, p1, p2])
    rot = Rotation.from_quat(rng.normal(size=4))
    pts = rot.apply(pts)
    return pts

def _generate_initializations(rng, n_candidates=12):
    """Generate diverse initial configurations."""
    candidates = []
    
    # Fibonacci spheres with random rotations
    for i in range(4):
        rot = Rotation.from_quat(rng.normal(size=4))
        candidates.append(_fibonacci_sphere(N, rng, rot))
    
    # Icosahedron + 2
    for i in range(3):
        candidates.append(_icosahedron_plus(rng))
    
    # Random spherical
    for i in range(3):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        rot = Rotation.from_quat(rng.normal(size=4))
        candidates.append(rot.apply(pts))
    
    # Two rings
    angles1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles2 = angles1 + np.pi / 7
    ring1 = np.column_stack([np.cos(angles1), np.sin(angles1), np.zeros(7)])
    ring2 = np.column_stack([np.cos(angles2), np.sin(angles2), np.full(7, 0.3)])
    pts = np.vstack([ring1, ring2])
    rot = Rotation.from_quat(rng.normal(size=4))
    candidates.append(rot.apply(pts))
    
    # Tetrahedron-based
    a = 1.0
    verts = np.array([
        [a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]
    ], dtype=float)
    verts -= np.mean(verts, axis=0)
    edge_len = np.linalg.norm(verts[0] - verts[1])
    verts /= edge_len
    edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    mids = np.array([(verts[i]+verts[j])/2 for i,j in edges])
    faces = [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
    fcs = np.array([(verts[i]+verts[j]+verts[k])/3 for i,j,k in faces])
    pts = np.vstack([verts, mids, fcs])
    pts -= np.mean(pts, axis=0)
    candidates.append(pts)
    
    return candidates[:n_candidates]

def construct():
    """Construct 14 points in 3D to minimize squared max/min distance ratio."""
    start_time = time.time()
    deadline = start_time + 125.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    best_points = None
    best_score = np.inf
    
    def update_best(pts):
        nonlocal best_points, best_score
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, D):
            return
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-10:
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()
    
    # Generate diverse initializations
    inits = _generate_initializations(rng, n_candidates=10)
    
    # Phase 1: Quick local refinement of each initialization with moderate beta
    beta_schedule = [5.0, 15.0, 50.0, 200.0]
    
    for i in range(len(inits)):
        if time.time() > deadline - 30.0:
            break
        x0 = inits[i].flatten()
        try:
            res = minimize(
                objective_ratio, x0,
                method='L-BFGS-B',
                args=(beta_schedule[0],),
                options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                inits[i] = pts
        except Exception:
            pass
    
    # Phase 2: Progressive refinement with increasing beta (sharper)
    for beta_idx in range(1, len(beta_schedule)):
        beta = beta_schedule[beta_idx]
        max_iter = 60 + beta_idx * 40
        for i in range(len(inits)):
            if time.time() > deadline - 20.0:
                break
            x0 = inits[i].flatten()
            try:
                res = minimize(
                    objective_ratio, x0,
                    method='L-BFGS-B',
                    args=(beta,),
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    inits[i] = pts
            except Exception:
                pass
    
    # Phase 3: Differential evolution global search (structural change)
    if time.time() < deadline - 25.0:
        # Define bounds for DE: points in a box, normalization handles scale
        bounds = [(-2.0, 2.0)] * (N * D)
        
        def de_objective(x):
            pts = x.reshape(N, D)
            pts = _normalize_points(pts)
            return raw_score(pts)
        
        try:
            de_res = differential_evolution(
                de_objective,
                bounds=bounds,
                seed=42,
                maxiter=30,
                popsize=8,
                tol=1e-6,
                init='sobol',
                polish=False,
                updating='immediate',
                workers=1
            )
            if de_res.x is not None:
                pts = de_res.x.reshape(N, D)
                update_best(pts)
        except Exception:
            pass
    
    # Phase 4: Final polish with raw objective using Nelder-Mead
    if best_points is not None and time.time() < deadline - 10.0:
        x0 = best_points.flatten()
        try:
            res = minimize(
                objective_raw, x0,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
        except Exception:
            pass
        
        # Second polish from perturbed version
        if time.time() < deadline - 5.0:
            x0 = best_points.flatten() + rng.normal(size=N*D) * 1e-4
            try:
                res = minimize(
                    objective_raw, x0,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
            except Exception:
                pass
    
    # Fallback
    if best_points is None:
        best_points = inits[0].copy()
    
    # Final validation: ensure distinctness
    sq_dists = get_pairwise_sq_dists(best_points)
    if np.min(sq_dists) < 1e-8:
        iu = np.triu_indices(N, k=1)
        min_idx = np.argmin(sq_dists)
        i, j = iu[0][min_idx], iu[1][min_idx]
        best_points[j] += 1e-5 * rng.normal(size=D)
    
    return best_points

# EVOLVE-BLOCK-END
