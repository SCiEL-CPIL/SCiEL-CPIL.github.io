"""Heilbronn triangles: maximize min triangle area for 11 points in equilateral triangle."""
import numpy as np
import time
import random

N = 11
H = np.sqrt(3) / 2.0
AREA_EQUI = np.sqrt(3) / 4.0

# Pre-compute all C(11,3)=165 triangle index triples
combos = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            combos.append((i, j, k))
combos = np.array(combos)
idx_i = combos[:, 0]
idx_j = combos[:, 1]
idx_k = combos[:, 2]


def _min_triangle_area(points):
    """Compute minimum triangle area (raw, not normalized) for N points."""
    pi = points[idx_i]
    pj = points[idx_j]
    pk = points[idx_k]
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj - pi)[:, 0] * (pk - pi)[:, 1] - (pj - pi)[:, 1] * (pk - pi)[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas.min()


def _is_inside_triangle(p):
    """Check if a 2D point is inside or on the boundary of the triangle."""
    x, y = p
    if y < -1e-12:
        return False
    if x < -1e-12:
        return False
    if x > 1 + 1e-12:
        return False
    # Check below the top edges
    # Left edge: from (0,0) to (0.5, H): y = 2*sqrt(3)*x  => x >= y/(2*sqrt(3))
    # Right edge: from (1,0) to (0.5, H): y = -2*sqrt(3)*(x-1) => x <= 1 - y/(2*sqrt(3))
    if y > 2 * np.sqrt(3) * x + 1e-9:
        return False
    if y > -2 * np.sqrt(3) * (x - 1) + 1e-9:
        return False
    return True


def _project_to_triangle(p):
    """Project a point onto the triangle (clip to feasible region)."""
    x, y = p
    # Clip to bounding box first
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(H, y))
    # Check left edge: y <= 2*sqrt(3)*x
    max_y_left = 2 * np.sqrt(3) * x
    if y > max_y_left:
        y = max_y_left
    # Check right edge: y <= -2*sqrt(3)*(x-1)
    max_y_right = -2 * np.sqrt(3) * (x - 1)
    if y > max_y_right:
        y = max_y_right
    # Re-check after clipping y, x might need adjustment
    # Actually the above should be sufficient for a convex triangle
    return np.array([x, y])


def _structured_init():
    """Create a structured initial configuration."""
    # Vertices
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, H])
    # Midpoints of edges
    m01 = (v0 + v1) / 2
    m12 = (v1 + v2) / 2
    m20 = (v2 + v0) / 2
    # Centroid
    c = (v0 + v1 + v2) / 3
    # Additional points: use points that help spread out
    # Points at 1/4 and 3/4 along edges, and some interior points
    p4 = np.array([0.25, 0.0])
    p5 = np.array([0.75, 0.0])
    p6 = (v2 + v1) / 4 * 3 + v1 / 4  # 3/4 from v2 to v1
    p7 = (v2 + v0) / 4 * 3 + v0 / 4  # 3/4 from v2 to v0
    p8 = np.array([0.25, 2 * np.sqrt(3) * 0.25 / 4])  # on left edge, 1/4 up
    p9 = np.array([0.75, 2 * np.sqrt(3) * 0.25 / 4])  # on right edge, 1/4 up
    
    # Let me use a better structured set
    # 3 vertices, 3 edge midpoints, 1 centroid, and 4 more
    points = [v0, v1, v2, m01, m12, m20, c]
    
    # Add points at 1/3 and 2/3 along each edge
    e01_13 = v0 + (v1 - v0) / 3
    e01_23 = v0 + 2 * (v1 - v0) / 3
    e12_13 = v1 + (v2 - v1) / 3
    e20_13 = v2 + (v0 - v2) / 3
    
    # Pick 4 more that spread well
    extra = [e01_13, e01_23, e12_13, e20_13]
    points.extend(extra)
    
    return np.array(points[:N])


def _local_optimize(points, max_iters=5000, step_init=0.05, step_min=0.0001, 
                     cool_rate=0.9995, deadline=None):
    """Simulated annealing with coordinate descent to maximize min triangle area."""
    best = points.copy()
    best_score = _min_triangle_area(best)
    current = points.copy()
    current_score = best_score
    step = step_init
    
    rng = random.Random(42)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = rng.randint(0, N - 1)
        
        # Random perturbation
        dx = (rng.random() - 0.5) * 2 * step
        dy = (rng.random() - 0.5) * 2 * step
        new_p = current[idx] + np.array([dx, dy])
        
        # Project to triangle
        new_p = _project_to_triangle(new_p)
        
        # Evaluate
        trial = current.copy()
        trial[idx] = new_p
        trial_score = _min_triangle_area(trial)
        
        # Accept if better, or with probability exp(-(score_diff)/T) for simulated annealing
        # We want to maximize, so delta = trial_score - current_score
        delta = trial_score - current_score
        if delta > 0:
            current = trial
            current_score = trial_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            # Simulated annealing acceptance
            # Temperature decreases
            T = step  # use step as temperature proxy
            if T > 1e-8:
                prob = np.exp(delta / (T + 1e-12))
                if rng.random() < prob:
                    current = trial
                    current_score = trial_score
        
        # Cool down
        step *= cool_rate
        if step < step_min:
            step = step_min
    
    return best, best_score


def _coordinate_descent(points, max_iters=2000, deadline=None):
    """Coordinate descent: for each point, try to move it to improve min area."""
    best = points.copy()
    best_score = _min_triangle_area(best)
    current = points.copy()
    current_score = best_score
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            # Try small moves in x and y directions
            for dx in [-0.01, -0.005, 0.005, 0.01]:
                trial = current.copy()
                trial[idx, 0] += dx
                trial[idx] = _project_to_triangle(trial[idx])
                s = _min_triangle_area(trial)
                if s > current_score:
                    current = trial
                    current_score = s
                    improved = True
            for dy in [-0.01, -0.005, 0.005, 0.01]:
                trial = current.copy()
                trial[idx, 1] += dy
                trial[idx] = _project_to_triangle(trial[idx])
                s = _min_triangle_area(trial)
                if s > current_score:
                    current = trial
                    current_score = s
                    improved = True
        
        if not improved:
            break
        
        if current_score > best_score:
            best = current.copy()
            best_score = current_score
    
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    # Start with structured initialization
    points = _structured_init()
    best_points, best_score = _local_optimize(points, deadline=deadline)
    
    # Try multiple random restarts with different initial configurations
    rng = random.Random(123)
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization with some structure
        pts = []
        # Include vertices and centroid for stability
        pts.append([0.0, 0.0])
        pts.append([1.0, 0.0])
        pts.append([0.5, H])
        pts.append([0.5, H / 3])  # centroid-ish
        # Add random points
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.95:
                p = [u + 0.5 * v, H * v]
                pts.append(p)
        
        pts = np.array(pts[:N])
        
        # Local optimize
        local_pts, local_score = _local_optimize(pts, deadline=deadline)
        
        if local_score > best_score:
            best_points = local_pts
            best_score = local_score
    
    # Final coordinate descent polish
    if time.time() < deadline - 5:
        best_points, best_score = _coordinate_descent(best_points, deadline=deadline)
    
    # Normalize: the task says the notebook divides by the area of the equilateral triangle
    # The evaluator measures the raw value (higher is better)
    # We return the points; the evaluator computes the score
    return best_points
