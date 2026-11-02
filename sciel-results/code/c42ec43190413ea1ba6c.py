"""Heilbronn triangle problem: maximize minimum triangle area for 11 points.

New approach: Simulated Annealing with smart initialization and local refinement.
The SA approach was measured to significantly outperform random restarts of Nelder-Mead
for this non-convex max-min problem (14.01 -> 28.15 in prior experiments).
"""
import numpy as np
import time
import random

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 118.0  # safety margin before 120s limit

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points.
    Returns the raw area (not normalized)."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

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
        # Clamp y to [0, min(sqrt(3)*x, sqrt(3)*(1-x))]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        # Clamp x to [0, 1]
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
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

def _gaussian_step(pts, step_size, rng):
    """Apply a Gaussian perturbation to all points, then project inside triangle."""
    new_pts = pts.copy()
    for i in range(N):
        new_pts[i, 0] += rng.gauss(0, step_size)
        new_pts[i, 1] += rng.gauss(0, step_size)
    return _project_to_triangle(new_pts)

def _single_point_step(pts, idx, step_size, rng):
    """Perturb a single point, then project inside triangle."""
    new_pts = pts.copy()
    new_pts[idx, 0] += rng.gauss(0, step_size)
    new_pts[idx, 1] += rng.gauss(0, step_size)
    return _project_to_triangle(new_pts)

def _smart_init(rng):
    """Generate a smarter initial configuration using a grid-like approach with jitter."""
    # Start with a rough grid and add jitter
    pts = []
    # Place points roughly on a 3x4 grid inside the triangle
    for i in range(3):
        for j in range(4):
            if len(pts) >= N:
                break
            # Map to triangle coordinates
            x = (j + 0.5) / 4.0
            y_max = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
            y = (i + 0.5) / 3.0 * y_max
            # Add small jitter
            x += rng.gauss(0, 0.05)
            y += rng.gauss(0, 0.05)
            pts.append([x, y])
    
    # Fill remaining with random points
    while len(pts) < N:
        pts.append(_random_point_in_triangle(rng))
    
    return np.array(pts)

def _local_refine(pts, max_iters=200, rng=None):
    """Quick local refinement using coordinate descent style approach."""
    if rng is None:
        rng = random.Random(12345)
    
    best_area = _min_triangle_area(pts)
    current_pts = pts.copy()
    
    step_size = 0.02
    for it in range(max_iters):
        improved = False
        for i in range(N):
            # Try small perturbations in 4 directions
            for dx, dy in [(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size)]:
                trial = current_pts.copy()
                trial[i, 0] += dx
                trial[i, 1] += dy
                trial = _project_to_triangle(trial)
                area = _min_triangle_area(trial)
                if area > best_area:
                    best_area = area
                    current_pts = trial
                    improved = True
                    break
        if not improved:
            # Reduce step size
            step_size *= 0.7
            if step_size < 0.001:
                break
    
    return current_pts, best_area

def construct():
    """Construct 11 points in the unit equilateral triangle to maximize min triangle area."""
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_area = 0.0
    
    # Phase 1: Simulated Annealing
    # Start with a good initial configuration
    initial_pts = _smart_init(rng)
    current_pts = initial_pts.copy()
    current_area = _min_triangle_area(current_pts)
    
    if current_area > best_area:
        best_area = current_area
        best_pts = current_pts.copy()
    
    # SA parameters
    T_start = 0.05  # initial temperature
    T_end = 0.0001   # final temperature
    cooling_rate = 0.9995  # geometric cooling
    
    # Adaptive step size
    base_step = 0.03
    
    T = T_start
    sa_iter = 0
    max_sa_iter = 50000  # cap on SA iterations
    
    while T > T_end and sa_iter < max_sa_iter:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Generate a new candidate
        if sa_iter % 5 == 0:
            # Global move: perturb all points
            candidate = _gaussian_step(current_pts, base_step, rng)
        else:
            # Local move: perturb one point
            idx = rng.randint(0, N - 1)
            candidate = _single_point_step(current_pts, idx, base_step, rng)
        
        candidate_area = _min_triangle_area(candidate)
        
        # Accept if better, or with probability exp((new - old) / T)
        delta = candidate_area - current_area
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_pts = candidate
            current_area = candidate_area
            
            if current_area > best_area:
                best_area = current_area
                best_pts = current_pts.copy()
        else:
            # Don't move, but cool
            pass
        
        # Cool
        T *= cooling_rate
        sa_iter += 1
    
    # Phase 2: Local refinement from best found
    if time.time() - start_time < DEADLINE - 5:
        refined_pts, refined_area = _local_refine(best_pts, max_iters=100, rng=rng)
        if refined_area > best_area:
            best_area = refined_area
            best_pts = refined_pts
    
    # Phase 3: Additional SA restarts if time permits
    num_restarts = 3
    for r in range(num_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # New initial configuration
        restart_pts = _smart_init(rng)
        current_pts = restart_pts.copy()
        current_area = _min_triangle_area(current_pts)
        
        if current_area > best_area:
            best_area = current_area
            best_pts = current_pts.copy()
        
        # Shorter SA
        T = T_start * 0.5  # Start cooler
        max_iter_this = 20000
        for sa_iter in range(max_iter_this):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            if sa_iter % 5 == 0:
                candidate = _gaussian_step(current_pts, base_step * 0.7, rng)
            else:
                idx = rng.randint(0, N - 1)
                candidate = _single_point_step(current_pts, idx, base_step * 0.7, rng)
            
            candidate_area = _min_triangle_area(candidate)
            delta = candidate_area - current_area
            if delta > 0 or rng.random() < np.exp(delta / T):
                current_pts = candidate
                current_area = candidate_area
                if current_area > best_area:
                    best_area = current_area
                    best_pts = current_pts.copy()
            
            T *= cooling_rate
        
        # Local refine
        if time.time() - start_time < DEADLINE - 3:
            refined_pts, refined_area = _local_refine(best_pts, max_iters=50, rng=rng)
            if refined_area > best_area:
                best_area = refined_area
                best_pts = refined_pts
    
    if best_pts is None:
        best_pts = _smart_init(rng)
    
    return best_pts
