import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

N = 13

# Pre-compute triangle indices for N=13
_TRI_INDICES = np.array(list(combinations(range(N), 3)))
_I_IDX = _TRI_INDICES[:, 0]
_J_IDX = _TRI_INDICES[:, 1]
_K_IDX = _TRI_INDICES[:, 2]
_NUM_TRI = len(_TRI_INDICES)

# Support line angles (evenly spaced)
_LINE_ANGLES = 2 * np.pi * np.arange(N) / N
_COS_A = np.cos(_LINE_ANGLES)
_SIN_A = np.sin(_LINE_ANGLES)

def _get_triangle_areas(points):
    """
    Vectorized calculation of areas for all triples.
    """
    pi = points[_I_IDX]
    pj = points[_J_IDX]
    pk = points[_K_IDX]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the raw score: min_triangle_area / hull_area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    areas = _get_triangle_areas(points)
    if len(areas) == 0:
        return 0.0
    min_area = np.min(areas)
    return min_area / hull_area

def _offsets_to_points(d):
    """
    Convert 13 support-line offsets to 13 polygon vertices.
    Each line k: x*cos(θ_k) + y*sin(θ_k) = d_k
    Vertex k is intersection of lines k and k+1 (mod N).
    """
    points = np.zeros((N, 2))
    for k in range(N):
        k1 = (k + 1) % N
        # Solve:
        #   x*cos(a_k) + y*sin(a_k) = d_k
        #   x*cos(a_k1) + y*sin(a_k1) = d_k1
        a0 = _LINE_ANGLES[k]
        a1 = _LINE_ANGLES[k1]
        c0, s0 = np.cos(a0), np.sin(a0)
        c1, s1 = np.cos(a1), np.sin(a1)
        det = c0 * s1 - s0 * c1
        if abs(det) < 1e-12:
            points[k] = [d[k] * c0, d[k] * s0]
        else:
            x = (d[k] * s1 - d[k1] * s0) / det
            y = (c0 * d[k1] - c1 * d[k]) / det
            points[k] = [x, y]
    return points

def _validate_polygon(d):
    """Check if the support-line polygon is valid (all offsets positive enough)."""
    # For a valid convex polygon containing origin, all d_k should be positive
    # and the polygon should have non-zero area
    if np.any(d < 1e-6):
        return False
    pts = _offsets_to_points(d)
    # Check area via shoelace
    x = pts[:, 0]
    y = pts[:, 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return area > 1e-8

def _sa_minimize_max_violation(d_init, deadline, rng, n_iterations=50000):
    """
    Simulated annealing to maximize the minimum triangle area / hull area.
    We minimize the negative score, but use SA with direct evaluation.
    """
    best_d = d_init.copy()
    best_score = _score(_offsets_to_points(d_init))
    current_d = d_init.copy()
    current_score = best_score
    
    T = 0.1  # Initial temperature
    cooling = 0.9997
    T_min = 1e-8
    
    iter_count = 0
    while time.time() < deadline and T > T_min and iter_count < n_iterations:
        iter_count += 1
        
        # Perturb: randomly change 1-3 offsets
        n_perturb = rng.randint(1, 4)
        indices = rng.choice(N, size=n_perturb, replace=False)
        new_d = current_d.copy()
        for idx in indices:
            new_d[idx] += rng.normal(0, 0.05)
        
        # Validate
        if not _validate_polygon(new_d):
            T *= cooling
            continue
        
        new_pts = _offsets_to_points(new_d)
        new_score = _score(new_pts)
        
        # Accept if better or with probability exp((new-old)/T)
        if new_score >= current_score:
            current_d = new_d
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_d = new_d.copy()
        else:
            delta = new_score - current_score
            prob = np.exp(delta / T)
            if rng.random() < prob:
                current_d = new_d
                current_score = new_score
        
        T *= cooling
    
    return best_d, best_score

def _construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s kill
    
    best_score = 0.0
    best_points = None
    rng = np.random.RandomState(42)
    
    # Initial offsets: regular polygon approximation
    # For a regular N-gon with circumradius R, the inradius (apothem) is R*cos(π/N)
    # The support line offset for angle θ is the inradius
    base_offset = 1.0 * np.cos(np.pi / N)
    
    # Multiple restarts with different initial conditions
    restarts = []
    
    # Regular polygon
    restarts.append(np.full(N, base_offset))
    
    # Slightly perturbed regular
    for _ in range(5):
        d = np.full(N, base_offset) + rng.normal(0, 0.02, N)
        restarts.append(d)
    
    # Varying offsets (elliptic-like)
    for amp in [0.1, 0.2, 0.3]:
        d = base_offset * (1 + amp * np.cos(2 * _LINE_ANGLES))
        restarts.append(np.maximum(d, 0.1))
    
    for amp in [0.1, 0.2]:
        d = base_offset * (1 + amp * np.cos(3 * _LINE_ANGLES))
        restarts.append(np.maximum(d, 0.1))
    
    # Asymmetric perturbations
    for _ in range(8):
        d = base_offset * (1 + rng.normal(0, 0.1, N))
        d = np.maximum(d, 0.1)
        restarts.append(d)
    
    # Run simulated annealing on each restart
    for i, d_init in enumerate(restarts):
        if time.time() > deadline - 10:
            break
        
        try:
            d_best, score = _sa_minimize_max_violation(d_init, deadline, rng)
            if score > best_score:
                best_score = score
                best_points = _offsets_to_points(d_best)
        except Exception:
            continue
    
    # If SA didn't find anything good, fallback to regular polygon
    if best_points is None or best_score < 0.01:
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_points)
    
    # Final polish with local Nelder-Mead on the best points
    if best_points is not None and time.time() < deadline - 5:
        def _neg_score_local(params):
            pts = params.reshape(N, 2)
            return -_score(pts)
        
        try:
            res = minimize(
                _neg_score_local,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            cand_pts = res.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            if cand_score > best_score:
                best_score = cand_score
                best_points = cand_pts
        except Exception:
            pass
    
    return best_points

def construct():
    return _construct()
