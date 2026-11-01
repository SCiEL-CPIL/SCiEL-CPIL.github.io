import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 14
D = 3
DEADLINE = 115.0  # Safety margin from 120s

def _pairwise_sq_dists(points):
    """
    Compute squared pairwise distances for N points in D dimensions.
    Returns a flat array of length N*(N-1)/2.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def _true_score(points):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    """
    sq_dists = _pairwise_sq_dists(points)
    min_d_sq = np.min(sq_dists)
    if min_d_sq < 1e-12:
        return np.inf
    return (np.max(sq_dists) / min_d_sq)

def _normalize_points(points):
    """
    Center and normalize points to unit mean squared norm.
    This removes translation and scale invariance.
    """
    points = points - np.mean(points, axis=0)
    # Compute total norm squared
    total_norm_sq = np.sum(points**2)
    if total_norm_sq < 1e-12:
        return points
    # Normalize such that sum of squared norms is 1 (or similar fixed scale)
    # Using sum of squared norms = 1 is equivalent to mean squared norm = 1/N
    points = points / np.sqrt(total_norm_sq)
    return points

def _soft_log_ratio_sq(sq_dists, beta=25.0):
    """
    Smooth approximation of log(max/min) using soft max/min on squared distances.
    Minimizing this is equivalent to minimizing the log of the distance ratio.
    Score is (max/min)^2 = exp(2 * log(max/min)).
    So minimizing log(max/min) minimizes the score.
    """
    # Clip to avoid log(0)
    sq_dists = np.maximum(sq_dists, 1e-12)
    log_sq_d = np.log(sq_dists)
    
    # Soft max of log_sq_d
    m_max = np.max(log_sq_d)
    soft_max = m_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (log_sq_d - m_max))))
    
    # Soft min of log_sq_d
    m_min = np.min(log_sq_d)
    soft_min = m_min - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (log_sq_d - m_min))))
    
    return soft_max - soft_min

def _objective_soft(params, beta=25.0):
    """
    Objective function for optimization.
    params: flattened (N, D) coordinates.
    """
    pts = params.reshape(N, D)
    pts = _normalize_points(pts)
    
    sq_dists = _pairwise_sq_dists(pts)
    
    if np.any(sq_dists < 1e-12):
        return 1e10
    
    return _soft_log_ratio_sq(sq_dists, beta=beta)

def _generate_fib_sphere_points():
    """Generate points on a sphere using Fibonacci distribution."""
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))

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

def _generate_tetrahedron_config():
    """
    Generate a tetrahedron-based configuration.
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
    
    # Face centers (4 faces)
    faces = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
    face_centers = []
    for f in faces:
        center = (tetra[f[0]] + tetra[f[1]] + tetra[f[2]]) / 3
        face_centers.append(center)
    
    pts = np.vstack([tetra, midpoints, face_centers])
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, np.newaxis]
    return pts[:N]

def _generate_spherical_shell():
    """
    Generate points on two concentric spherical shells.
    """
    n_inner = 7
    n_outer = N - n_inner
    
    # Inner shell: radius 0.6
    phi_in = np.arccos(1 - 2 * (np.arange(n_inner) + 0.5) / n_inner)
    theta_in = np.pi * (1 + 5 ** 0.5) * np.arange(n_inner)
    inner = np.column_stack([
        0.6 * np.sin(phi_in) * np.cos(theta_in),
        0.6 * np.sin(phi_in) * np.sin(theta_in),
        0.6 * np.cos(phi_in)
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

def _generate_random_points():
    """Generate random points."""
    return np.random.randn(N, D)

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
    for _ in range(8):
        candidates.append(_generate_random_points())
    
    # Multi-stage optimization:
    # Stage 1: Coarse optimization with L-BFGS from each candidate
    # Stage 2: Basin hopping-like perturbation and re-optimization
    # Stage 3: Fine polish with higher beta
    
    stage1_results = []
    
    for i, init_pts in enumerate(candidates):
        if time.time() - t_start > DEADLINE - 35:
            break
        
        try:
            init_flat = init_pts.flatten()
            
            # Stage 1: L-BFGS with moderate beta
            res = minimize(
                _objective_soft,
                init_flat,
                method='L-BFGS-B',
                args=(25.0,),
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            opt_pts = res.x.reshape(N, D)
            opt_pts = _normalize_points(opt_pts)
            
            # Check for distinct points
            d_sq = _pairwise_sq_dists(opt_pts)
            if d_sq.min() < 1e-8:
                continue
            
            score = _true_score(opt_pts)
            stage1_results.append((score, opt_pts))
            
            if score < best_score:
                best_score = score
                best_pts = opt_pts.copy()
                
        except Exception:
            continue
    
    # Stage 2: Perturbation and re-optimization (Basin Hopping style)
    if time.time() - t_start < DEADLINE - 25 and len(stage1_results) > 0:
        # Sort by score and take top 5
        stage1_results.sort(key=lambda x: x[0])
        top_results = stage1_results[:min(5, len(stage1_results))]
        
        for i, (init_score, init_pts) in enumerate(top_results):
            if time.time() - t_start > DEADLINE - 20:
                break
            
            # Perform multiple perturbations
            for pert in range(3):
                if time.time() - t_start > DEADLINE - 20:
                    break
                
                try:
                    # Add noise
                    noise = np.random.randn(N, D) * 0.1
                    perturbed_pts = init_pts + noise
                    perturbed_flat = perturbed_pts.flatten()
                    
                    res = minimize(
                        _objective_soft,
                        perturbed_flat,
                        method='L-BFGS-B',
                        args=(25.0,),
                        options={'maxiter': 300, 'ftol': 1e-12}
                    )
                    
                    opt_pts = res.x.reshape(N, D)
                    opt_pts = _normalize_points(opt_pts)
                    
                    d_sq = _pairwise_sq_dists(opt_pts)
                    if d_sq.min() < 1e-8:
                        continue
                    
                    score = _true_score(opt_pts)
                    
                    if score < best_score:
                        best_score = score
                        best_pts = opt_pts.copy()
                        
                except Exception:
                    continue
    
    # Stage 3: Fine polish with higher beta for more precise optimization
    if time.time() - t_start < DEADLINE - 10 and best_pts is not None:
        try:
            # Use a higher beta for the final polish
            res = minimize(
                _objective_soft,
                best_pts.flatten(),
                method='L-BFGS-B',
                args=(60.0,),
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            
            opt_pts = res.x.reshape(N, D)
            opt_pts = _normalize_points(opt_pts)
            
            d_sq = _pairwise_sq_dists(opt_pts)
            if d_sq.min() < 1e-8:
                return best_pts
            
            score = _true_score(opt_pts)
            if score < best_score:
                best_score = score
                best_pts = opt_pts.copy()
                
        except Exception:
            pass
    
    # Stage 4: Additional local refinement with Nelder-Mead
    if time.time() - t_start < DEADLINE - 5 and best_pts is not None:
        try:
            res = minimize(
                _objective_soft,
                best_pts.flatten(),
                method='Nelder-Mead',
                args=(40.0,),
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            
            opt_pts = res.x.reshape(N, D)
            opt_pts = _normalize_points(opt_pts)
            
            d_sq = _pairwise_sq_dists(opt_pts)
            if d_sq.min() < 1e-8:
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
        
    # Final check: ensure points are distinct
    d_final_sq = _pairwise_sq_dists(best_pts)
    if d_final_sq.min() < 1e-8:
        best_pts = _generate_fib_sphere_points()
    
    return best_pts
