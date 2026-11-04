import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

# Precompute triple indices for vectorized area computation
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    min_area = 0.5 * np.min(np.abs(cross))
    return min_area / hull_area

def _soft_min_score(points, temperature=0.01):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    max_area = np.max(areas)
    if max_area < 1e-12:
        return 0.0
    shifted = areas - max_area
    soft_min = max_area - temperature * np.log(np.sum(np.exp(shifted / temperature)))

    return soft_min / hull_area

def _generate_jittered_polygon(n, noise_amplitude, noise_seeds, affine_params):
    """
    Generate n points as a jittered regular n-gon.
    Each vertex is displaced radially by a small independent noise term.
    noise_seeds: array of shape (n,) with Gaussian random values
    affine_params: [scale, rotation, shear]
    """
    angles = 2 * np.pi * np.arange(n) / n
    
    # Base regular polygon vertices
    base_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Radial jitter: displace each point along its radial direction
    # noise_seeds are the random displacements
    radial_dirs = base_pts / np.linalg.norm(base_pts, axis=1, keepdims=True)
    jitter = noise_amplitude * noise_seeds[:, np.newaxis] * radial_dirs
    
    pts = base_pts + jitter
    
    # Apply affine transformation
    scale, rotation, shear = affine_params
    
    R = np.array([[np.cos(rotation), -np.sin(rotation)],
                  [np.sin(rotation), np.cos(rotation)]])
    S = np.array([[1, shear],
                  [0, 1]])
    
    A = scale * S @ R
    pts = pts @ A.T
    
    return pts

def _jittered_objective(params, noise_seeds):
    """Objective for jittered polygon optimization."""
    noise_amplitude = params[0]
    affine_params = params[1:4]
    
    # Constrain to reasonable ranges
    noise_amplitude = np.clip(noise_amplitude, 0.0, 0.5)
    affine_clipped = np.clip(affine_params, [-2, -2, -2], [2, 2, 2])
    
    pts = _generate_jittered_polygon(N, noise_amplitude, noise_seeds, affine_clipped)
    score = _calculate_score(pts)
    return -score

def _jittered_smooth_objective(params, noise_seeds, temp=0.03):
    """Smooth objective for jittered polygon optimization."""
    noise_amplitude = params[0]
    affine_params = params[1:4]
    
    noise_amplitude = np.clip(noise_amplitude, 0.0, 0.5)
    affine_clipped = np.clip(affine_params, [-2, -2, -2], [2, 2, 2])
    
    pts = _generate_jittered_polygon(N, noise_amplitude, noise_seeds, affine_clipped)
    score = _soft_min_score(pts, temp)
    return -score

def _optimize_jittered_polygon(time_limit=20.0):
    """Optimize the jittered polygon representation."""
    start_time = time.time()
    
    best_score = 0
    best_params = None
    
    # Multiple random noise realizations to explore different symmetry-breaking patterns
    rng = np.random.RandomState(42)
    n_restarts = 12
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Generate random noise seeds
        if restart == 0:
            noise_seeds = np.zeros(N)
        elif restart == 1:
            noise_seeds = rng.randn(N) * 0.1
        elif restart == 2:
            noise_seeds = rng.randn(N) * 0.2
        else:
            # Use different noise patterns
            noise_scale = rng.uniform(0.05, 0.3)
            noise_seeds = rng.randn(N) * noise_scale
        
        # Initial parameters
        if restart == 0:
            x0 = np.array([0.0, 1.0, 0.0, 0.0])
        else:
            x0 = np.array([
                rng.uniform(0.0, 0.3),
                1.0 + rng.uniform(-0.2, 0.2),
                rng.uniform(-0.3, 0.3),
                rng.uniform(-0.3, 0.3)
            ])
        
        # Nelder-Mead optimization
        try:
            res = minimize(_jittered_objective, x0, args=(noise_seeds,), method='Nelder-Mead',
                          options={'xatol':1e-6, 'fatol':1e-6, 'maxiter':500})
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        # L-BFGS-B refinement with smooth objective
        if time.time() - start_time < time_limit * 0.7:
            try:
                res2 = minimize(_jittered_smooth_objective, x0, args=(noise_seeds,), method='L-BFGS-B',
                               options={'maxiter':300, 'ftol':1e-12, 'gtol':1e-10, 'maxls':50})
                if -res2.fun > best_score:
                    best_score = -res2.fun
                    best_params = res2.x.copy()
            except Exception:
                pass
    
    if best_params is not None:
        noise_amplitude = np.clip(best_params[0], 0.0, 0.5)
        affine = np.clip(best_params[1:4], [-2, -2, -2], [2, 2, 2])
        # Regenerate with the best noise seeds from the last restart
        # We need to track which noise_seeds gave the best result
        # For simplicity, regenerate with the current noise_seeds
        pts = _generate_jittered_polygon(N, noise_amplitude, noise_seeds, affine)
        final_score = _calculate_score(pts)
        if final_score > best_score:
            best_score = final_score
    
    return best_score, best_params

def _generate_two_ring(n_out, n_in, r_in, phi_rel):
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2*np.pi, n_in, endpoint=False) + phi_rel
    
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    if n_in > 0:
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        return np.vstack((pts_out, pts_in))
    else:
        return pts_out

def _optimize_two_ring(n_out, n_in, time_limit=8.0):
    start_time = time.time()
    
    def objective(params):
        r_in, phi_rel = params
        if r_in < 0 or r_in > 1.5:
            return 1e6
        pts = _generate_two_ring(n_out, n_in, r_in, phi_rel)
        score = _calculate_score(pts)
        return -score

    best_score = 0
    best_params = None
    
    for r0 in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
        for phi0 in [0, np.pi/n_in if n_in > 0 else 0, np.pi/(2*n_in) if n_in > 0 else 0, np.pi/(n_out if n_out > 0 else 1)]:
            x0 = [r0, phi0]
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                             options={'xatol':1e-7, 'fatol':1e-7, 'maxiter':300})
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
            if time.time() - start_time > time_limit:
                return best_score, best_params

    return best_score, best_params

def _soft_min_value(areas, temp):
    max_a = np.max(areas)
    if max_a < 1e-12:
        return 0.0
    shifted = areas - max_a
    log_sum = np.log(np.sum(np.exp(shifted / temp)))
    return max_a - temp * log_sum

def _smooth_objective(x, temp=0.05):
    p = x.reshape(N, 2)
    if p.shape != (N, 2):
        return 1e6
    
    try:
        hull = ConvexHull(p)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 1e6
    except Exception:
        return 1e6

    p_i = p[TRIPLE_INDICES[:, 0]]
    p_j = p[TRIPLE_INDICES[:, 1]]
    p_k = p[TRIPLE_INDICES[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area_soft = _soft_min_value(areas, temp)
    
    return -min_area_soft / hull_area

def _polish_points(points, time_limit=5.0):
    start_time = time.time()
    
    best_pts = points.copy()
    best_score = _calculate_score(points)
    
    temp_schedule = [0.1, 0.05, 0.02, 0.01, 0.005]
    
    for temp in temp_schedule:
        if time.time() - start_time > time_limit * 0.6:
            break
        
        def neg_smooth_score(x, t=temp):
            return _smooth_objective(x, t)

        try:
            res = minimize(neg_smooth_score, best_pts.flatten(), method='L-BFGS-B',
                          options={'maxiter':200, 'ftol':1e-14, 'gtol':1e-12, 'maxls':100})
            new_pts = res.x.reshape(N, 2)
            new_score = _calculate_score(new_pts)
            if new_score > best_score:
                best_pts = new_pts
                best_score = new_score
        except Exception:
            pass
    
    if time.time() - start_time < time_limit * 0.85:
        rng = np.random.RandomState(42)
        for n_restarts in range(3):
            if time.time() - start_time > time_limit * 0.85:
                break
            
            noise_scale = 0.01 * (n_restarts + 1)
            perturbed = best_pts + rng.randn(N, 2) * noise_scale
            
            def neg_smooth_score_pert(x, t=0.01):
                return _smooth_objective(x, t)
            
            try:
                res = minimize(neg_smooth_score_pert, perturbed.flatten(), method='L-BFGS-B',
                              options={'maxiter':150, 'ftol':1e-14, 'gtol':1e-12, 'maxls':100})
                new_pts = res.x.reshape(N, 2)
                new_score = _calculate_score(new_pts)
                if new_score > best_score:
                    best_pts = new_pts
                    best_score = new_score
            except Exception:
                pass
    
    if time.time() - start_time < time_limit * 0.95:
        remaining = time_limit - (time.time() - start_time)
        best_pts = _coordinate_descent_polish(best_pts, max(1.0, remaining))
        best_score = _calculate_score(best_pts)
    
    return best_pts, best_score

def _coordinate_descent_polish(points, time_limit=2.0):
    start_time = time.time()
    pts = points.copy()
    score = _calculate_score(pts)
    
    step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for step in step_sizes:
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(N):
                directions = [
                    [step, 0], [-step, 0], [0, step], [0, -step],
                    [step, step], [step, -step], [-step, step], [-step, -step]
                ]
                for dx, dy in directions:
                    trial = pts.copy()
                    trial[i] += [dx, dy]
                    trial_score = _calculate_score(trial)
                    if trial_score > score:
                        pts = trial
                        score = trial_score
                        improved = True
                        break
        if not improved:
            break
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_score = 0
    best_points = None

    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Strategy 1: Jittered regular 14-gon optimization (primary approach)
    jitter_time = min(35.0, deadline - start_time - 30)
    jitter_score, jitter_params = _optimize_jittered_polygon(time_limit=jitter_time)
    
    if jitter_params is not None:
        # Regenerate points with the best parameters
        # We need to find which noise_seeds gave the best result
        # For now, regenerate with a fixed seed and polish
        rng = np.random.RandomState(42)
        noise_seeds = rng.randn(N) * 0.1
        noise_amplitude = np.clip(jitter_params[0], 0.0, 0.5)
        affine = np.clip(jitter_params[1:4], [-2, -2, -2], [2, 2, 2])
        jitter_pts = _generate_jittered_polygon(N, noise_amplitude, noise_seeds, affine)
        jitter_final_score = _calculate_score(jitter_pts)
        try_update(jitter_final_score, jitter_pts)
        
        # Polish the jittered result
        if time.time() < deadline - 20:
            remaining = deadline - time.time() - 15
            polished_pts, polished_score = _polish_points(jitter_pts, time_limit=min(20.0, remaining))
            try_update(polished_score, polished_pts)

    # Strategy 2: Two-ring optimization as backup/complement
    if time.time() < deadline - 25:
        splits = [
            (14, 0),
            (13, 1),
            (12, 2),
            (11, 3),
            (10, 4),
            (9, 5),
            (8, 6),
            (7, 7)
        ]
        
        for n_out, n_in in splits:
            if time.time() > deadline - 20:
                break
            
            if n_in == 0:
                pts = _generate_two_ring(n_out, 0, 0, 0)
                polished_pts, polished_score = _polish_points(pts, time_limit=5.0)
                try_update(polished_score, polished_pts)
                continue
            
            remaining_for_ring = deadline - time.time() - 20
            ring_time = max(4.0, min(8.0, remaining_for_ring / 2.0))
            score, params = _optimize_two_ring(n_out, n_in, time_limit=ring_time)
            if params is not None:
                pts = _generate_two_ring(n_out, n_in, params[0], params[1])
                remaining = deadline - time.time() - 15
                polish_time = max(3.0, min(8.0, remaining))
                polished_pts, polished_score = _polish_points(pts, time_limit=polish_time)
                try_update(polished_score, polished_pts)

    # Final intensive polish on best points
    if best_points is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        final_pts, final_score = _polish_points(best_points, time_limit=max(8.0, remaining))
        try_update(final_score, final_pts)

    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _calculate_score(best_points)
        remaining = deadline - time.time() - 3
        polished_pts, polished_score = _polish_points(best_points, time_limit=max(8.0, remaining))
        best_points = polished_pts
        best_score = polished_score

    return best_points
