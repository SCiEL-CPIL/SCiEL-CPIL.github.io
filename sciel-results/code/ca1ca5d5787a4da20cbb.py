"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize

H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)
_n_triangles = len(_idx_i)


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas.min()


def _is_inside(p):
    """Check if point p=(x,y) is inside the equilateral triangle."""
    x, y = p
    if y < -1e-9 or x < -1e-9 or x > 1 + 1e-9:
        return False
    if y > np.sqrt(3) * x + 1e-9:
        return False
    if y > np.sqrt(3) * (1 - x) + 1e-9:
        return False
    return True


def project_to_triangle(p):
    """
    p: array of shape (N, 2)
    Returns projected points inside the triangle (0,0), (1,0), (0.5, h)
    """
    p = p.copy()
    # Clip to bounding box first
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, H)
    
    # Iterate a few times to handle corners properly
    for _ in range(3):
        # Edge AB: y >= 0
        mask_ab = p[:, 1] < 0.0
        if np.any(mask_ab):
            p[mask_ab, 1] = 0.0
            
        # Edge AC: y <= sqrt(3) x  =>  sqrt(3)x - y >= 0
        # Line: sqrt(3)x - y = 0. Normal pointing inward: (sqrt(3), -1).
        # Actually, for point (x,y), violation is y > sqrt(3)x.
        # Project to line y = sqrt(3)x.
        mask_ac = p[:, 1] > np.sqrt(3) * p[:, 0]
        if np.any(mask_ac):
            px = p[mask_ac, 0]
            py = p[mask_ac, 1]
            # Projection of (px, py) onto line through origin with direction (1, sqrt(3))
            # v = (1, sqrt(3)). v.v = 4.
            # dot = px + py*sqrt(3)
            # proj = dot/4 * v
            dot = px + py * np.sqrt(3)
            px_proj = dot / 4.0
            py_proj = dot * np.sqrt(3) / 4.0
            # Clamp to segment AC: x in [0, 0.5]
            px_proj = np.clip(px_proj, 0.0, 0.5)
            py_proj = px_proj * np.sqrt(3)
            p[mask_ac, 0] = px_proj
            p[mask_ac, 1] = py_proj
            
        # Edge BC: y <= sqrt(3)(1-x) => y + sqrt(3)x <= sqrt(3)
        # Violation: y > sqrt(3)(1-x)
        mask_bc = p[:, 1] > np.sqrt(3) * (1.0 - p[:, 0])
        if np.any(mask_bc):
            px = p[mask_bc, 0]
            py = p[mask_bc, 1]
            # Line passes through (1,0) and (0.5, h).
            # Equation: y = -sqrt(3)(x-1) = sqrt(3)(1-x)
            # Normal vector pointing inward (towards origin side): (sqrt(3), 1).
            # Let's check: At (0.5, 0), y=0 < sqrt(3)(0.5)=0.866. Inward.
            # Vector from point (1,0) to (0.5, h) is (-0.5, h) = (-0.5, sqrt(3)/2).
            # Normal to this is (sqrt(3)/2, 0.5) or (sqrt(3), 1).
            # Let's project onto the line.
            # Line eq: sqrt(3)x + y - sqrt(3) = 0.
            # Distance d = (sqrt(3)px + py - sqrt(3)) / (3+1) = (sqrt(3)px + py - sqrt(3)) / 4
            # Since violation is y > sqrt(3)(1-x) => sqrt(3)x + y > sqrt(3) => d > 0.
            # Projection: P - d * n, where n=(sqrt(3), 1).
            d = (np.sqrt(3) * px + py - np.sqrt(3)) / 4.0
            px_proj = px - d * np.sqrt(3)
            py_proj = py - d * 1.0
            # Clamp to segment BC: x in [0.5, 1.0]
            px_proj = np.clip(px_proj, 0.5, 1.0)
            py_proj = np.sqrt(3) * (1.0 - px_proj)
            p[mask_bc, 0] = px_proj
            p[mask_bc, 1] = py_proj
            
    return p

def _objective_exact(points):
    """Minimize negative exact min-area."""
    for i in range(N):
        if not _is_inside(points[i]):
            return 1e6
    return -_min_area(points)

def construct():
    t_start = time.time()
    deadline = t_start + 140.0

    rng = np.random.RandomState(42)
    
    best_points = None
    best_score = -1e9

    # --- Phase 1: Generate diverse initial guesses ---
    
    # Guess 1: Symmetric rows (5, 3, 2, 1)
    y_levels = [0.08*H, 0.35*H, 0.62*H, 0.88*H]
    counts = [5, 3, 2, 1]
    pts = []
    for y, c in zip(y_levels, counts):
        x_min = y / np.sqrt(3)
        x_max = 1.0 - y / np.sqrt(3)
        if c > 1:
            margin = 0.03 * (x_max - x_min)
            xs = np.linspace(x_min + margin, x_max - margin, c)
        else:
            xs = np.array([(x_min + x_max) / 2])
        for x in xs:
            pts.append([x, y])
    guess1 = project_to_triangle(np.array(pts))
    
    # Guess 2: Hexagonal-like
    pts = []
    y0 = 0.05 * H
    for i in range(5):
        x = 0.1 + 0.2 * i
        pts.append([x, y0])
    y1 = 0.3 * H
    for i in range(3):
        x = 0.25 + 0.25 * i
        pts.append([x, y1])
    y2 = 0.55 * H
    for i in range(2):
        x = 0.35 + 0.3 * i
        pts.append([x, y2])
    pts.append([0.5, 0.8 * H])
    guess2 = project_to_triangle(np.array(pts))
    
    # Guess 3: Random
    guess3 = project_to_triangle(rng.random((N, 2)) * np.array([1.0, H]))
    
    # Guess 4: Random
    guess4 = project_to_triangle(rng.random((N, 2)) * np.array([1.0, H]))
    
    initial_guesses = [guess1, guess2, guess3, guess4]
    
    # --- Phase 2: Local Optimization (Nelder-Mead) on each guess ---
    
    def _obj_flat(x):
        pts = x.reshape(N, 2)
        pts = project_to_triangle(pts)
        return _objective_exact(pts)
        
    for i, start_points in enumerate(initial_guesses):
        if time.time() > deadline - 10.0:
            break
            
        x0 = start_points.flatten()
        
        try:
            res = minimize(
                _obj_flat,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            curr_points = project_to_triangle(res.x.reshape(N, 2))
            curr_score = -_min_area(curr_points) # objective returns -area, so best is most negative
            # Wait, _objective_exact returns -area. So lower is better.
            # Let's stick to the convention: score = min_area. We want max score.
            # _objective_exact returns -min_area.
            # So if res.fun < best_score (which is -min_area), then min_area > best_min_area.
            
            # Let's just compute the actual min area for comparison
            actual_area = _min_area(curr_points)
            if actual_area > best_score:
                best_score = actual_area
                best_points = curr_points
        except Exception:
            pass

    # --- Phase 3: Simulated Annealing Refinement ---
    # Use the best point found so far as the starting point for SA
    if best_points is None:
        best_points = initial_guesses[0]
        best_score = _min_area(best_points)
        
    current_points = best_points.copy()
    current_score = best_score
    
    T = 0.005
    T_min = 1e-8
    alpha = 0.999
    max_iter = 50000
    
    for i in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        idx = rng.randint(N)
        scale = 0.05 * (T + 0.01)
        delta = rng.normal(0, scale, 2)
        
        new_points = current_points.copy()
        new_points[idx] += delta
        new_points = project_to_triangle(new_points)
        
        new_area = _min_area(new_points)
        new_score = -new_area
        
        # Acceptance: we are minimizing new_score (which is -area)
        # So if new_score < current_score, it's better.
        if new_score < current_score:
            current_score = new_score
            current_points = new_points
            if -current_score > best_score:
                best_score = -current_score
                best_points = current_points.copy()
        else:
            delta_e = current_score - new_score # positive if new is worse (higher -area)
            if delta_e < 0:
                accept = True
            else:
                accept_prob = np.exp(-delta_e / T)
                accept = rng.random() < accept_prob
            
            if accept:
                current_score = new_score
                current_points = new_points
        
        T *= alpha
        if T < T_min:
            T = 0.005

    # --- Phase 4: Final Polish ---
    if time.time() < deadline - 2.0:
        try:
            x0 = best_points.flatten()
            res = minimize(
                _obj_flat,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            final_points = project_to_triangle(res.x.reshape(N, 2))
            final_area = _min_area(final_points)
            if final_area > best_score:
                best_points = final_points
                best_score = final_area
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
