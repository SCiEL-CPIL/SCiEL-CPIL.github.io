"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

N = 13
DEADLINE = 135.0  # safety margin before 150s kill

# Pre-compute all triples for fast vectorized scoring
# There are C(13, 3) = 286 triples
_triple_i = []
_triple_j = []
_triple_k = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _triple_i.append(_i)
            _triple_j.append(_j)
            _triple_k.append(_k)

_triple_i = np.array(_triple_i, dtype=np.int32)
_triple_j = np.array(_triple_j, dtype=np.int32)
_triple_k = np.array(_triple_k, dtype=np.int32)

def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    # Vectorized computation of all triangle areas
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area < 1e-15:
        return 0.0
    
    return min_area / hull_area

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return 0.5 * np.abs(cross)

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """Generate a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])

def _make_two_layer(inner_n, outer_n, r_inner=0.5, r_outer=1.0):
    """
    Create a two-layer configuration.
    inner_n points on a circle of radius r_inner.
    outer_n points on a circle of radius r_outer.
    Total points must be N.
    """
    if inner_n + outer_n != N:
        raise ValueError("inner_n + outer_n must equal N")
    
    pts = []
    if inner_n > 0:
        angles_in = np.linspace(0, 2 * np.pi, inner_n, endpoint=False)
        x_in = r_inner * np.cos(angles_in)
        y_in = r_inner * np.sin(angles_in)
        pts.append(np.column_stack([x_in, y_in]))
        
    if outer_n > 0:
        angles_out = np.linspace(0, 2 * np.pi, outer_n, endpoint=False)
        x_out = r_outer * np.cos(angles_out)
        y_out = r_outer * np.sin(angles_out)
        pts.append(np.column_stack([x_out, y_out]))
        
    return np.vstack(pts)

def _make_three_layer(center_n, mid_n, outer_n, r_mid=0.5, r_outer=1.0):
    """
    Create a three-layer configuration.
    center_n points at center (or small ring).
    mid_n points on circle radius r_mid.
    outer_n points on circle radius r_outer.
    """
    if center_n + mid_n + outer_n != N:
        raise ValueError("center_n + mid_n + outer_n must equal N")
        
    pts = []
    if center_n == 1:
        pts.append(np.array([[0.0, 0.0]]))
    elif center_n > 1:
        angles_c = np.linspace(0, 2 * np.pi, center_n, endpoint=False)
        r_c = 0.1
        x_c = r_c * np.cos(angles_c)
        y_c = r_c * np.sin(angles_c)
        pts.append(np.column_stack([x_c, y_c]))
        
    if mid_n > 0:
        angles_m = np.linspace(0, 2 * np.pi, mid_n, endpoint=False)
        x_m = r_mid * np.cos(angles_m)
        y_m = r_mid * np.sin(angles_m)
        pts.append(np.column_stack([x_m, y_m]))
        
    if outer_n > 0:
        angles_o = np.linspace(0, 2 * np.pi, outer_n, endpoint=False)
        x_o = r_outer * np.cos(angles_o)
        y_o = r_outer * np.sin(angles_o)
        pts.append(np.column_stack([x_o, y_o]))
        
    return np.vstack(pts)

def _logsumexp_surrogate(pts, beta=50.0):
    """
    Smooth surrogate: -logsumexp(-beta * areas) approximates min(areas) for large beta.
    This gives a differentiable objective that we can optimize with gradient methods.
    """
    areas = _triangle_areas(pts)
    if np.any(areas < 1e-15):
        return 0.0
    # -logsumexp(-beta * areas) / beta approximates min(areas)
    # Use stable logsumexp
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:  # overflow guard
        return areas[np.argmin(areas)]
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum / beta

def _neg_score(params):
    """Negative score for minimization. params is 1D array of length 2*N."""
    pts = params.reshape(N, 2)
    return -_score_points(pts)

def _neg_surrogate(params, beta=50.0):
    """Negative log-sum-exp surrogate for minimization."""
    pts = params.reshape(N, 2)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1.0
    surrogate_val = _logsumexp_surrogate(pts, beta)
    return -surrogate_val / hull_a

def _soft_min_score(params, beta=30.0):
    """
    Soft-min objective using log-sum-exp:
    f(x) = -(1/beta) * log(sum(exp(-beta * a_i / hull)))
    approximates -min(a_i) / hull for large beta.
    """
    pts = params.reshape(N, 2)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1.0
    areas = _triangle_areas(pts)
    if np.any(areas < 1e-15):
        return 1.0
    scaled = -beta * areas / hull_a
    max_s = np.max(scaled)
    if max_s > 700:
        min_a = np.min(areas)
        return -min_a / hull_a
    log_sum = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return -(1.0 / beta) * log_sum

def _neg_soft_score(params, beta=30.0):
    return _soft_min_score(params, beta)

def _neg_hard_score(params):
    return _neg_score(params)

def _simulated_annealing(pts, t_start, max_iters=5000, t0=1.0, cooling=0.999):
    """
    Simulated annealing targeting the worst triangle.
    At each step, identify the minimum-area triangle and propose a move
    that increases its area while monitoring the global minimum.
    """
    best_pts = pts.copy()
    best_score = _score_points(best_pts)
    current_pts = pts.copy()
    current_score = _score_points(current_pts)
    temp = t0
    
    rng = np.random.RandomState(42)
    
    for it in range(max_iters):
        elapsed = time.time() - t_start
        if elapsed > DEADLINE - 10.0:
            break
        
        # Get current areas
        areas = _triangle_areas(current_pts)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        # Identify the triple of the worst triangle
        i1 = _triple_i[min_idx]
        i2 = _triple_j[min_idx]
        i3 = _triple_k[min_idx]
        
        # Propose a move: perturb one of the three points slightly
        # Direction: away from the centroid of the triangle to increase area
        p1 = current_pts[i1]
        p2 = current_pts[i2]
        p3 = current_pts[i3]
        centroid = (p1 + p2 + p3) / 3.0
        
        # Pick a random vertex from the worst triangle to move
        choice = rng.randint(3)
        if choice == 0:
            idx = i1
            others = [i2, i3]
        elif choice == 1:
            idx = i2
            others = [i1, i3]
        else:
            idx = i3
            others = [i1, i2]
        
        # Direction away from the midpoint of the other two
        mid_other = (current_pts[others[0]] + current_pts[others[1]]) / 2.0
        direction = current_pts[idx] - mid_other
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10:
            direction = rng.normal(0, 1, 2)
        else:
            direction = direction / dir_norm
        
        # Add some random component
        noise = rng.normal(0, 0.05 * temp, 2)
        step_size = 0.02 * temp
        move = step_size * direction + noise
        
        proposed_pts = current_pts.copy()
        proposed_pts[idx] = proposed_pts[idx] + move
        proposed_pts = np.clip(proposed_pts, -2.0, 2.0)
        
        proposed_score = _score_points(proposed_pts)
        
        delta = proposed_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-10)):
            current_pts = proposed_pts
            current_score = proposed_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        
        temp *= cooling
    
    return best_pts, best_score

def _optimize_ring_1c12o(t_start, deadline_margin=10.0):
    """
    Optimize the 1-center + 12-outer ring configuration.
    Parameters: r_outer (radius), 12 angle offsets.
    Total: 13 parameters.
    """
    t_start = time.time()
    best_score = -np.inf
    best_params = None
    
    # Initial guess: regular 12-gon on outer ring
    base_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    
    def _ring_params_to_points(r_center, r_outer, angle_offsets):
        """
        Convert ring parameters to 13 points: 1 center + 12 outer.
        angle_offsets: 12 angles (radians) for outer ring points.
        Returns (13, 2) array.
        """
        pts = np.zeros((N, 2))
        # Center point at origin
        pts[0] = [0.0, 0.0]
        # Outer ring
        for i in range(12):
            angle = angle_offsets[i]
            pts[i + 1, 0] = r_outer * np.cos(angle)
            pts[i + 1, 1] = r_outer * np.sin(angle)
        return pts
    
    def neg_obj_ring(params):
        """Objective for ring optimization: params = [r_outer, angle_0, ..., angle_11]"""
        r_outer = params[0]
        angles = params[1:]
        pts = _ring_params_to_points(0.0, r_outer, angles)
        hull_a = _hull_area(pts)
        if hull_a <= 1e-12:
            return 1.0
        areas = _triangle_areas(pts)
        min_a = np.min(areas)
        if min_a < 1e-15:
            return 1.0
        return -min_a / hull_a
    
    def neg_obj_ring_surrogate(params, beta=30.0):
        """Surrogate version for smoother optimization."""
        r_outer = params[0]
        angles = params[1:]
        pts = _ring_params_to_points(0.0, r_outer, angles)
        hull_a = _hull_area(pts)
        if hull_a <= 1e-12:
            return 1.0
        areas = _triangle_areas(pts)
        if np.any(areas < 1e-15):
            return 1.0
        # Log-sum-exp surrogate for min
        scaled = -beta * areas
        max_s = np.max(scaled)
        if max_s > 700:
            min_a = np.min(areas)
            return -min_a / hull_a
        log_sum = max_s + np.log(np.sum(np.exp(scaled - max_s)))
        surrogate = -log_sum / beta
        return -surrogate / hull_a
    
    # Bounds: r_outer in [0.3, 1.5], angles in [0, 2*pi]
    bounds_ring = [(0.3, 1.5)] + [(0.0, 2 * np.pi)] * 12
    
    # Phase 1: Differential evolution on ring parameters
    try:
        de_result = differential_evolution(
            neg_obj_ring,
            bounds_ring,
            seed=42,
            maxiter=60,
            popsize=20,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=True,
            init='sobol',
            updating='deferred',
            workers=1
        )
        
        if de_result.fun < -best_score:
            candidate_params = de_result.x
            r_out = candidate_params[0]
            angles = candidate_params[1:]
            pts = _ring_params_to_points(0.0, r_out, angles)
            score = _score_points(pts)
            if score > best_score:
                best_score = score
                best_params = candidate_params.copy()
    except Exception:
        pass
    
    # Phase 2: Local refinement with L-BFGS-B from multiple starts
    if best_params is None:
        best_params = np.array([1.0] + list(base_angles))
    
    starts = [best_params.copy()]
    
    # Add some perturbed starts
    rng = np.random.RandomState(123)
    for _ in range(6):
        pert = rng.normal(0, 0.1, size=13)
        pert[0] = rng.uniform(-0.2, 0.2)  # smaller perturbation for radius
        start = best_params + pert
        start[0] = np.clip(start[0], 0.3, 1.5)
        start[1:] = np.mod(start[1:], 2 * np.pi)
        starts.append(start)
    
    # Also try regular polygon start
    starts.append(np.array([1.0] + list(base_angles)))
    
    for start in starts:
        if time.time() - t_start > deadline_margin:
            break
        try:
            res = minimize(
                neg_obj_ring,
                start,
                method='L-BFGS-B',
                bounds=bounds_ring,
                options={'maxiter': 500, 'maxfun': 10000}
            )
            candidate_params = res.x
            r_out = candidate_params[0]
            angles = candidate_params[1:]
            pts = _ring_params_to_points(0.0, r_out, angles)
            score = _score_points(pts)
            if score > best_score:
                best_score = score
                best_params = candidate_params.copy()
        except Exception:
            pass
    
    # Phase 3: Surrogate-based refinement for smoother landscape
    if best_params is not None and time.time() - t_start < deadline_margin:
        for beta in [15.0, 30.0]:
            if time.time() - t_start > deadline_margin:
                break
            try:
                res_surr = minimize(
                    lambda p: neg_obj_ring_surrogate(p, beta=beta),
                    best_params,
                    method='L-BFGS-B',
                    bounds=bounds_ring,
                    options={'maxiter': 300, 'maxfun': 5000}
                )
                candidate_params = res_surr.x
                r_out = candidate_params[0]
                angles = candidate_params[1:]
                pts = _ring_params_to_points(0.0, r_out, angles)
                score = _score_points(pts)
                if score > best_score:
                    best_score = score
                    best_params = candidate_params.copy()
            except Exception:
                pass
    
    # Convert back to full 26-dim parameter vector
    if best_params is not None:
        r_out = best_params[0]
        angles = best_params[1:]
        pts = _ring_params_to_points(0.0, r_out, angles)
        return pts, best_score
    
    return None, best_score

def _equalize_active_triangles(pts, t_start, max_iters=200, tolerance=1e-8):
    """
    Equalize the areas of the 'active' (near-minimum) triangles.
    This is a max-min fairness approach: we try to increase the minimum
    by moving points to equalize the worst triangles.
    """
    best_pts = pts.copy()
    best_score = _score_points(best_pts)
    current_pts = pts.copy()
    
    for it in range(max_iters):
        elapsed = time.time() - t_start
        if elapsed > DEADLINE - 15.0:
            break
        
        areas = _triangle_areas(current_pts)
        hull_a = _hull_area(current_pts)
        if hull_a < 1e-12:
            break
        
        min_area = np.min(areas)
        if min_area < 1e-15:
            break
        
        # Find all "active" triangles (within 5% of minimum)
        threshold = min_area * 1.05
        active_mask = areas <= threshold
        active_idx = np.where(active_mask)[0]
        
        if len(active_idx) == 0:
            break
        
        # Compute gradient-like direction: for each point, sum contributions
        # from active triangles that involve that point
        # We want to move points to increase the areas of active triangles
        
        # Use a simple approach: for each active triangle, compute the
        # direction that would increase its area, and average over all
        # active triangles for each point
        displacement = np.zeros_like(current_pts)
        
        for idx in active_idx:
            i1 = _triple_i[idx]
            i2 = _triple_j[idx]
            i3 = _triple_k[idx]
            
            p1 = current_pts[i1]
            p2 = current_pts[i2]
            p3 = current_pts[i3]
            
            # Area = 0.5 * |cross(p2-p1, p3-p1)|
            # To increase area, move points in direction that increases cross product
            v12 = p2 - p1
            v13 = p3 - p1
            cross_val = v12[0] * v13[1] - v12[1] * v13[0]
            
            if abs(cross_val) < 1e-15:
                continue
            
            sign = np.sign(cross_val)
            
            # Gradient of cross product w.r.t. each vertex
            # d/dp1: -(v12 x v13 direction)
            # d/dp2: direction to increase cross
            # d/dp3: direction to increase cross
            
            # For a triangle, moving p1 opposite to the direction from p1 to midpoint of p2p3
            # increases area. Similarly for other vertices.
            mid_23 = (p2 + p3) / 2.0
            dir_1 = p1 - mid_23
            
            mid_13 = (p1 + p3) / 2.0
            dir_2 = p2 - mid_13
            
            mid_12 = (p1 + p2) / 2.0
            dir_3 = p3 - mid_12
            
            # Normalize
            n1 = np.linalg.norm(dir_1)
            n2 = np.linalg.norm(dir_2)
            n3 = np.linalg.norm(dir_3)
            
            if n1 > 1e-10:
                displacement[i1] += sign * dir_1 / n1
            if n2 > 1e-10:
                displacement[i2] += sign * dir_2 / n2
            if n3 > 1e-10:
                displacement[i3] += sign * dir_3 / n3
        
        # Normalize total displacement
        disp_norm = np.linalg.norm(displacement)
        if disp_norm < 1e-15:
            break
        
        # Take a small step
        step_size = 0.01
        proposed_pts = current_pts + step_size * displacement / disp_norm
        proposed_pts = np.clip(proposed_pts, -3.0, 3.0)
        
        proposed_score = _score_points(proposed_pts)
        
        if proposed_score > _score_points(current_pts) - 1e-10:
            current_pts = proposed_pts
            if proposed_score > best_score:
                best_score = proposed_score
                best_pts = current_pts.copy()
        else:
            # Reduce step size
            step_size *= 0.5
            if step_size < 1e-8:
                break
    
    return best_pts, best_score

def construct():
    t_start = time.time()
    
    best_score = -np.inf
    best_pts = None
    
    def time_remaining():
        return (DEADLINE - 5.0) - (time.time() - t_start)
    
    # === Phase 1: Symmetric ring optimization (1 center + 12 outer) ===
    # This is the primary structural approach, exploiting likely optimal geometry
    ring_pts, ring_score = _optimize_ring_1c12o(t_start, deadline_margin=DEADLINE - 40.0)
    
    if ring_pts is not None:
        best_pts = ring_pts
        best_score = ring_score
    
    # === Phase 2: General optimization from ring solution and other inits ===
    bounds = [(-1.5, 1.5)] * (2 * N)
    rng = np.random.RandomState(42)
    
    # Build initializations
    inits = []
    
    # Regular 13-gon
    inits.append(_regular_polygon(N))
    
    # 1 center + 12 outer (regular)
    inits.append(_make_three_layer(1, 0, 12))
    
    # 1 center + 6 mid + 6 outer
    inits.append(_make_three_layer(1, 6, 6))
    
    # 7 mid + 6 outer
    inits.append(_make_two_layer(7, 6))
    
    # 1 center + 4 mid + 8 outer
    inits.append(_make_three_layer(1, 4, 8, r_mid=0.4, r_outer=1.0))
    
    # Random
    inits.append(rng.uniform(-1, 1, size=(N, 2)))
    
    # Add the ring solution if we have one
    if best_pts is not None:
        inits.append(best_pts.copy())
    
    # Evaluate all inits
    for p in inits:
        s = _score_points(p)
        if s > best_score:
            best_score = s
            best_pts = p.copy()
    
    # === Phase 3: Simulated annealing from best solution ===
    if time_remaining() > 25.0 and best_pts is not None:
        sa_pts, sa_score = _simulated_annealing(best_pts, t_start, max_iters=8000, t0=0.5, cooling=0.9995)
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts
    
    # === Phase 4: Equalize active triangles ===
    if time_remaining() > 15.0 and best_pts is not None:
        eq_pts, eq_score = _equalize_active_triangles(best_pts, t_start, max_iters=300)
        if eq_score > best_score:
            best_score = eq_score
            best_pts = eq_pts
    
    # === Phase 5: More simulated annealing with different parameters ===
    if time_remaining() > 20.0 and best_pts is not None:
        sa2_pts, sa2_score = _simulated_annealing(best_pts, t_start, max_iters=6000, t0=0.3, cooling=0.9997)
        if sa2_score > best_score:
            best_score = sa2_score
            best_pts = sa2_pts
    
    # === Phase 6: Local refinement ===
    if time_remaining() > 15.0 and best_pts is not None:
        starts_to_refine = [best_pts.flatten()]
        
        # Add a few other good inits
        for p in inits[:4]:
            s = _score_points(p)
            if s > best_score * 0.8:
                starts_to_refine.append(p.flatten())
        
        # Deduplicate
        unique_starts = []
        for s in starts_to_refine:
            is_unique = True
            for u in unique_starts:
                if np.allclose(s, u, atol=1e-3):
                    is_unique = False
                    break
            if is_unique:
                unique_starts.append(s)
        
        for start in unique_starts:
            if time_remaining() < 10.0:
                break
            try:
                res = minimize(
                    _neg_score,
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'maxfun': 8000}
                )
                score = _score_points(res.x.reshape(N, 2))
                if score > best_score:
                    best_score = score
                    best_pts = res.x.reshape(N, 2)
            except Exception:
                pass
    
    # === Phase 7: Surrogate-based refinement ===
    if time_remaining() > 12.0 and best_pts is not None:
        for beta in [15.0, 25.0]:
            if time_remaining() < 8.0:
                break
            try:
                res_surr = minimize(
                    lambda p: _neg_surrogate(p, beta=beta),
                    best_pts.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'maxfun': 4000}
                )
                score = _score_points(res_surr.x.reshape(N, 2))
                if score > best_score:
                    best_score = score
                    best_pts = res_surr.x.reshape(N, 2)
            except Exception:
                pass
    
    # === Phase 8: Final local polish ===
    if time_remaining() > 5.0 and best_pts is not None:
        try:
            res_final = minimize(
                _neg_score,
                best_pts.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'maxfun': 3000}
            )
            score = _score_points(res_final.x.reshape(N, 2))
            if score > best_score:
                best_score = score
                best_pts = res_final.x.reshape(N, 2)
        except Exception:
            pass
    
    # === Phase 9: Final equalization pass ===
    if time_remaining() > 3.0 and best_pts is not None:
        eq2_pts, eq2_score = _equalize_active_triangles(best_pts, t_start, max_iters=100)
        if eq2_score > best_score:
            best_score = eq2_score
            best_pts = eq2_pts
    
    # Safety: ensure we return a valid point set
    if best_pts is None:
        best_pts = _regular_polygon(N)
    
    return best_pts
