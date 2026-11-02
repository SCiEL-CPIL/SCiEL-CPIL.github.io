"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.

Strategy:
1. Direct derivative-free local search (nudge / hill-climb) on the true minimum-area objective.
2. Vectorized area evaluation for speed.
3. Multiple diverse initializations (random, grid, greedy, structured).
4. Simulated-annealing-style acceptance to escape local minima.
5. Refinement with progressively smaller step sizes.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 140.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]
n_triplets = len(_triplets)


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True


def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0.0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0.0
        if x > 1:
            x = 1.0
        pts[i, 0] = x
        pts[i, 1] = y
    return pts


def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]


def _grid_init(rng, n_per_side=4):
    """Structured grid initialization inside the triangle."""
    pts = []
    for i in range(n_per_side):
        for j in range(n_per_side - i):
            u = (i + 0.5) / n_per_side
            v = (j + 0.5) / (n_per_side - i)
            if u + v < 1:
                x = u + 0.5 * v
                y = S3H * v
                pts.append([x + rng.uniform(-0.02, 0.02), y + rng.uniform(-0.02, 0.02)])
    # Add corners and edge midpoints
    pts.append([0.0, 0.0])
    pts.append([1.0, 0.0])
    pts.append([0.5, S3H])
    pts.append([0.25, 0.0])
    pts.append([0.75, 0.0])
    pts.append([0.25, S3H * 0.5])
    pts.append([0.75, S3H * 0.5])
    pts.append([0.5, S3H * 0.5])
    pts.append([0.5, 0.0])
    # Take first N points, project to triangle
    pts = np.array(pts[:N], dtype=float)
    return _project_to_triangle(pts)


def _greedy_init(rng):
    """Greedy initialization: place points one by one to maximize minimum distance to existing points."""
    pts = []
    # Start with three corners
    pts.append([0.0, 0.0])
    pts.append([1.0, 0.0])
    pts.append([0.5, S3H])
    
    for _ in range(N - 3):
        best_pt = None
        best_score = -1
        for _ in range(500):
            pt = _random_point_in_triangle(rng)
            # Score: minimum distance to all existing points
            dmin = min(np.linalg.norm(pt - p) for p in pts)
            if dmin > best_score:
                best_score = dmin
                best_pt = pt
        pts.append(best_pt)
    
    pts = np.array(pts, dtype=float)
    return _project_to_triangle(pts)


def _structured_init(rng):
    """Structured initialization based on known Heilbronn-like patterns."""
    # Use a pattern that spreads points well
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radii = [0.0, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7]
    center = np.array([0.5, S3H / 3])
    pts = []
    for r, a in zip(radii, angles):
        x = center[0] + r * np.cos(a) * 0.5
        y = center[1] + r * np.sin(a) * 0.3
        pts.append([x + rng.uniform(-0.05, 0.05), y + rng.uniform(-0.05, 0.05)])
    pts = np.array(pts, dtype=float)
    return _project_to_triangle(pts)


def _nudge_search(pts, rng, max_iters=5000, initial_step=0.1):
    """
    Nudge-based local search with simulated annealing.
    Try small random perturbations and accept if they improve or with SA probability.
    """
    pts = np.array(pts, dtype=float)
    current_score = _min_triangle_area(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    step = initial_step
    T = 1.0  # Temperature
    cooling = 0.9995
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        # Choose a random point to nudge
        idx = rng.randrange(N)
        
        # Random direction
        angle = rng.uniform(0, 2 * np.pi)
        magnitude = step * rng.uniform(0.5, 1.5)
        dx = magnitude * np.cos(angle)
        dy = magnitude * np.sin(angle)
        
        new_pt = pts[idx].copy()
        new_pt[0] += dx
        new_pt[1] += dy
        
        # Project to triangle
        x, y = new_pt
        if y < 0:
            y = 0.0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0.0
        if x > 1:
            x = 1.0
        new_pt[0] = x
        new_pt[1] = y
        
        # Skip if no change
        if np.allclose(new_pt, pts[idx]):
            continue
        
        trial_pts = pts.copy()
        trial_pts[idx] = new_pt
        trial_score = _min_triangle_area(trial_pts)
        
        if trial_score >= current_score:
            pts = trial_pts
            current_score = trial_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
            # Reduce step slightly on improvement
            step *= 0.999
        else:
            # Simulated annealing: accept worse move with probability exp(-delta/T)
            delta = current_score - trial_score
            if delta < 1e-10 or rng.random() < np.exp(-delta / T):
                pts = trial_pts
                current_score = trial_score
            # Reduce step on rejection
            step *= 0.998
        
        T *= cooling
        
        # Occasionally increase step if stuck
        if it % 500 == 0 and it > 0:
            # Check if we've made progress recently
            if best_score - current_score < 1e-8:
                step = min(step * 1.5, initial_step)
    
    return best_pts, best_score


def _refine_with_lbfgs(pts):
    """
    Refine using L-BFGS-B on a smooth approximation of the minimum area.
    This is used as a final polishing step.
    """
    def obj(x_flat):
        p = x_flat.reshape(N, 2)
        # Penalty for going outside
        penalty = 0.0
        for px, py in p:
            d1 = py
            d2 = np.sqrt(3) * px - py
            d3 = np.sqrt(3) * (1 - px) - py
            if d1 < 0:
                penalty += d1 ** 2
            if d2 < 0:
                penalty += d2 ** 2
            if d3 < 0:
                penalty += d3 ** 2
        
        pi = p[_i_idx]
        pj = p[_j_idx]
        pk = p[_k_idx]
        dx1 = pj - pi
        dx2 = pk - pi
        cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        # Soft min with beta
        beta = 100.0
        min_a = np.min(areas)
        exp_terms = np.exp(-beta * (areas - min_a))
        log_sum = np.log(np.sum(exp_terms))
        soft_min_val = min_a + (1.0 / beta) * log_sum
        
        return -soft_min_val + 500.0 * penalty
    
    try:
        result = minimize(obj, pts.flatten(), method='L-BFGS-B',
                          options={'maxiter': 30, 'maxfun': 50})
        refined = result.x.reshape(N, 2)
        refined = _project_to_triangle(refined)
        if _check_inside(refined):
            return refined
    except Exception:
        pass
    return pts


def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses multi-start nudge search with diverse initializations and SA.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # Diverse initializations
    inits = []
    inits.append(_greedy_init(rng))
    inits.append(_grid_init(rng))
    inits.append(_structured_init(rng))
    
    # Random initializations
    for _ in range(5):
        pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        inits.append(_project_to_triangle(pts))
    
    # Run nudge search on each initialization
    for init in inits:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        result_pts, result_score = _nudge_search(init, rng, max_iters=3000, initial_step=0.08)
        
        if result_score > best_score:
            best_score = result_score
            best_pts = result_pts
    
    # Final refinement with L-BFGS-B on the best solution
    if best_pts is not None and time.time() - start_time < DEADLINE - 3:
        refined = _refine_with_lbfgs(best_pts)
        refined_score = _min_triangle_area(refined)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined
    
    # One more round of nudge search on the refined result if time permits
    if best_pts is not None and time.time() - start_time < DEADLINE - 2:
        result_pts, result_score = _nudge_search(best_pts, rng, max_iters=2000, initial_step=0.03)
        if result_score > best_score:
            best_score = result_score
            best_pts = result_pts
    
    if best_pts is None:
        best_pts = _project_to_triangle(np.array([_random_point_in_triangle(rng) for _ in range(N)]))
    
    return best_pts
