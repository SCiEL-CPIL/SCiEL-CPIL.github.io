import time
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.spatial.distance import cdist
import itertools

N = 14
D = 3
DEADLINE = 115.0  # Safety margin from 120s

def _pairwise_distances(pts):
    """Compute pairwise distances between all points."""
    d = cdist(pts, pts)
    mask = ~np.eye(N, dtype=bool)
    return d[mask]

def _true_score(pts):
    """Calculate the actual score: (max/min)^2."""
    d = _pairwise_distances(pts)
    if len(d) == 0:
        return np.inf
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _soft_log_ratio(distances, beta=20.0):
    """
    Smooth approximation of log(max/min) using soft max/min.
    Returns a scalar that approximates log(max) - log(min).
    """
    log_d = np.log(distances + 1e-12)
    # Soft max of log_d
    soft_max = (1.0 / beta) * np.log(np.sum(np.exp(beta * log_d)))
    # Soft min of log_d
    soft_min = -(1.0 / beta) * np.log(np.sum(np.exp(-beta * log_d)))
    return soft_max - soft_min

def _objective_epigraph(params, t):
    """
    Epigraph formulation: minimize t subject to all log(distances) in [0, t].
    This is a smooth approximation using soft constraints.
    """
    pts = params.reshape(N, D)
    pts = pts - np.mean(pts, axis=0)
    total_norm = np.sqrt(np.sum(pts ** 2))
    if total_norm < 1e-9:
        return 1e10
    pts = pts / total_norm
    
    d = _pairwise_distances(pts)
    if np.any(d < 1e-12):
        return 1e10
    
    log_d = np.log(d)
    
    # Soft constraint: log_d <= t + epsilon
    # Use a smooth max: soft_max(log_d - t) should be close to 0
    beta = 30.0
    soft_violation_max = (1.0 / beta) * np.log(np.sum(np.exp(beta * (log_d - t))))
    # Soft constraint: log_d >= 0 (i.e., -log_d <= 0)
    soft_violation_min = -(1.0 / beta) * np.log(np.sum(np.exp(-beta * log_d)))
    
    # Minimize t + penalty for violations
    penalty_weight = 100.0
    return t + penalty_weight * (soft_violation_max + soft_violation_min)

def _objective_gram(params):
    """
    Objective in Gram matrix parameterization.
    params: flattened (N, D) coordinates (we'll build Gram from these)
    
    Actually, let's use a direct coordinate approach but with a better objective:
    minimize the soft log ratio of distances.
    """
    pts = params.reshape(N, D)
    
    # Center to remove translation invariance
    pts = pts - np.mean(pts, axis=0)
    
    # Scale to unit norm (remove scale invariance)
    total_norm = np.sqrt(np.sum(pts ** 2))
    if total_norm < 1e-9:
        return 1e10
    pts = pts / total_norm
    
    d = _pairwise_distances(pts)
    
    if np.any(d < 1e-12):
        return 1e10
    
    # Soft log ratio - smooth and differentiable
    return _soft_log_ratio(d, beta=20.0)

def _generate_tetrahedron_config():
    """
    Generate a tetrahedron-based configuration: 4 vertices of a regular tetrahedron
    plus 10 points at edge midpoints.
    """
    # Regular tetrahedron vertices
    s = np.sqrt(2)
    tetra = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]) / s
    
    # Edge midpoints (6 edges)
    edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    midpoints = []
    for i, j in edges:
        mid = (tetra[i] + tetra[j]) / 2
        midpoints.append(mid)
    
    # We have 4 + 6 = 10 points, need 4 more
    # Add face centers (4 faces)
    faces = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
    face_centers = []
    for f in faces:
        center = (tetra[f[0]] + tetra[f[1]] + tetra[f[2]]) / 3
        face_centers.append(center)
    
    pts = np.vstack([tetra, midpoints, face_centers])
    return pts[:N]

def _generate_icosahedron_plus_2():
    """
    Generate points based on an icosahedron (12 vertices) plus 2 additional points.
    """
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize to unit sphere
    norms = np.linalg.norm(v, axis=1)
    v = v / norms[:, np.newaxis]
    
    p1 = np.array([0, 0, 1.0])
    p2 = np.array([0, 0, -1.0])
    
    pts = np.vstack([v, p1, p2])
    return pts

def _generate_fib_sphere_points():
    """Generate points on a sphere using Fibonacci distribution."""
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))

def _generate_ring_config():
    """
    Generate a ring-like configuration: points distributed on two rings
    in the x-y plane at different z levels.
    """
    n_per_ring = N // 2
    remaining = N - 2 * n_per_ring
    
    pts = []
    # Ring 1 at z = 0.3
    angles1 = np.linspace(0, 2*np.pi, n_per_ring, endpoint=False)
    for a in angles1:
        pts.append([np.cos(a), np.sin(a), 0.3])
    
    # Ring 2 at z = -0.3
    angles2 = np.linspace(0, 2*np.pi, n_per_ring, endpoint=False) + np.pi / n_per_ring
    for a in angles2:
        pts.append([np.cos(a), np.sin(a), -0.3])
    
    # Remaining points
    if remaining > 0:
        pts.append([0, 0, 0.7])
    if remaining > 1:
        pts.append([0, 0, -0.7])
    
    return np.array(pts[:N])

def _generate_random_points():
    """Generate random points."""
    return np.random.randn(N, D)

def _generate_spherical_shell():
    """
    Generate points on two concentric spherical shells.
    7 points on inner shell, 7 on outer shell.
    """
    n_inner = 7
    n_outer = N - n_inner
    
    # Inner shell: radius 0.5
    phi_in = np.arccos(1 - 2 * (np.arange(n_inner) + 0.5) / n_inner)
    theta_in = np.pi * (1 + 5 ** 0.5) * np.arange(n_inner)
    inner = np.column_stack([
        0.5 * np.sin(phi_in) * np.cos(theta_in),
        0.5 * np.sin(phi_in) * np.sin(theta_in),
        0.5 * np.cos(phi_in)
    ])
    
    # Outer shell: radius 1.0
    phi_out = np.arccos(1 - 2 * (np.arange(n_outer) + 0.5) / n_outer)
    theta_out = np.pi * (1 + 5 ** 0.5) * np.arange(n_outer)
    outer = np.column_stack([
        np.sin(phi_out) * np.cos(theta_out),
        np.sin(phi_out) * np.sin(theta_out),
        np.cos(phi_out)
    ])
    
    return np.vstack([inner, outer])

def construct():
    t_start = time.time()
    
    best_score = np.inf
    best_pts = None
    
    # Initial candidates - diverse starting points
    candidates = []
    candidates.append(_generate_fib_sphere_points())
    candidates.append(_generate_icosahedron_plus_2())
    candidates.append(_generate_ring_config())
    candidates.append(_generate_tetrahedron_config())
    candidates.append(_generate_spherical_shell())
    
    # Add random starts
    for _ in range(6):
        candidates.append(_generate_random_points())
    
    # Multi-stage optimization:
    # Stage 1: Coarse optimization with L-BFGS from each candidate
    # Stage 2: Basin hopping from best Stage 1 results
    # Stage 3: Fine polish with higher beta
    
    stage1_results = []
    
    for i, init_pts in enumerate(candidates):
        if time.time() - t_start > DEADLINE - 30:
            break
        
        try:
            init_flat = init_pts.flatten()
            
            # Stage 1: L-BFGS with moderate beta
            res = minimize(
                _objective_gram,
                init_flat,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            
            opt_pts = res.x.reshape(N, D)
            opt_pts = opt_pts - np.mean(opt_pts, axis=0)
            
            # Check for distinct points
            d = _pairwise_distances(opt_pts)
            if d.min() < 1e-6:
                continue
            
            score = _true_score(opt_pts)
            stage1_results.append((score, opt_pts))
            
            if score < best_score:
                best_score = score
                best_pts = opt_pts.copy()
                
        except Exception:
            continue
    
    # Stage 2: Basin hopping from top Stage 1 results
    if time.time() - t_start < DEADLINE - 25 and len(stage1_results) > 0:
        # Sort by score and take top 4
        stage1_results.sort(key=lambda x: x[0])
        top_results = stage1_results[:min(4, len(stage1_results))]
        
        for i, (init_score, init_pts) in enumerate(top_results):
            if time.time() - t_start > DEADLINE - 18:
                break
            
            try:
                # Use larger stepsize for better exploration
                res = basinhopping(
                    _objective_gram,
                    init_pts.flatten(),
                    minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 150}},
                    niter=40,
                    stepsize=0.5,
                    seed=100 + i
                )
                
                opt_pts = res.x.reshape(N, D)
                opt_pts = opt_pts - np.mean(opt_pts, axis=0)
                
                d = _pairwise_distances(opt_pts)
                if d.min() < 1e-6:
                    continue
                
                score = _true_score(opt_pts)
                
                if score < best_score:
                    best_score = score
                    best_pts = opt_pts.copy()
                    
            except Exception:
                continue
    
    # Stage 3: Fine polish with higher beta for more precise optimization
    if time.time() - t_start < DEADLINE - 8 and best_pts is not None:
        try:
            # Use a higher beta for the final polish
            def _objective_fine(params):
                pts = params.reshape(N, D)
                pts = pts - np.mean(pts, axis=0)
                total_norm = np.sqrt(np.sum(pts ** 2))
                if total_norm < 1e-9:
                    return 1e10
                pts = pts / total_norm
                d = _pairwise_distances(pts)
                if np.any(d < 1e-12):
                    return 1e10
                return _soft_log_ratio(d, beta=50.0)
            
            res = minimize(
                _objective_fine,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-14}
            )
            
            opt_pts = res.x.reshape(N, D)
            opt_pts = opt_pts - np.mean(opt_pts, axis=0)
            
            d = _pairwise_distances(opt_pts)
            if d.min() < 1e-6:
                return best_pts
            
            score = _true_score(opt_pts)
            if score < best_score:
                best_score = score
                best_pts = opt_pts.copy()
                
        except Exception:
            pass
    
    # Stage 4: Additional local refinement with Nelder-Mead
    if time.time() - t_start < DEADLINE - 3 and best_pts is not None:
        try:
            def _objective_nm(params):
                pts = params.reshape(N, D)
                pts = pts - np.mean(pts, axis=0)
                total_norm = np.sqrt(np.sum(pts ** 2))
                if total_norm < 1e-9:
                    return 1e10
                pts = pts / total_norm
                d = _pairwise_distances(pts)
                if np.any(d < 1e-12):
                    return 1e10
                return _soft_log_ratio(d, beta=30.0)
            
            res = minimize(
                _objective_nm,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            
            opt_pts = res.x.reshape(N, D)
            opt_pts = opt_pts - np.mean(opt_pts, axis=0)
            
            d = _pairwise_distances(opt_pts)
            if d.min() < 1e-6:
                return best_pts
            
            score = _true_score(opt_pts)
            if score < best_score:
                best_score = score
                best_pts = opt_pts.copy()
                
        except Exception:
            pass
    
    # Fallback
    if best_pts is None:
        best_pts = _generate_fib_sphere_points()
        
    return best_pts
