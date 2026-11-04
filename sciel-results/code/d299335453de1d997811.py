import numpy as np
import itertools
from scipy.optimize import minimize
import time

N = 11
# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2.0])
TRI_AREA = np.sqrt(3) / 4.0

def _in_triangle(p):
    """Check if point p is inside the triangle ABC."""
    # Using barycentric coordinates or cross product sign
    v0 = C - A
    v1 = B - A
    v2 = p - A
    # Compute dot products
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    # Compute barycentric coordinates
    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    return (u >= 0) and (v >= 0) and (u + v <= 1)

def _project_to_triangle(p):
    """Project point p onto the triangle ABC if it's outside."""
    if _in_triangle(p):
        return p
    # Find closest point on boundary
    # This is a simplification; for optimization, we can just penalize or clip
    # For robustness, let's just return the point but the optimizer will push it in
    # Or we can use a penalty method in the objective.
    return p

def _min_triangle_area(points):
    """Calculate the minimum area of all triangles formed by the points."""
    min_area = float('inf')
    # points is an array of shape (N, 2)
    for i, j, k in itertools.combinations(range(N), 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        if area < min_area:
            min_area = area
    return min_area

def _objective(x):
    """Objective function for optimization: negative of min area."""
    points = x.reshape(N, 2)
    # Penalty for points outside the triangle
    penalty = 0.0
    for p in points:
        if not _in_triangle(p):
            # Calculate distance to boundary and add penalty
            # Simple penalty: large value
            penalty += 1000.0
    if penalty > 0:
        return penalty
    min_area = _min_triangle_area(points)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = -float('inf')
    best_points = None
    
    # Initial guesses
    initial_guesses = []
    
    # Guess 1: Triangular lattice approximation
    # For n=11, a common structure is related to a 3x3 grid plus corners or similar.
    # Let's try a grid of points scaled to fit.
    # A simple heuristic: place points on a triangular grid.
    # For n=11, maybe a 3x3 grid (9 points) + 2 extra?
    # Or just random starts.
    
    rng = np.random.default_rng(42)
    
    # Generate a few random valid starts
    for _ in range(5):
        points = []
        while len(points) < N:
            u = rng.random()
            v = rng.random()
            if u + v < 1:
                p = A + u * (B - A) + v * (C - A)
                # Add small jitter to avoid exact overlaps in early stages if needed, 
                # but Nelder-Mead handles duplicates poorly, so ensure distinctness.
                if all(np.linalg.norm(p - q) > 0.01 for q in points):
                    points.append(p)
        if len(points) == N:
            initial_guesses.append(np.array(points).flatten())

    # Add a structured guess: Triangular lattice
    # Let's try to map a 3x3 grid to the triangle
    # This is tricky, so let's stick to random + one specific symmetric guess.
    # Symmetric guess: Center, Midpoints of sides, Vertices? That's 7 points.
    # For 11, maybe add midpoints of medians?
    
    # Let's just run optimization on the random starts.
    for i, x0 in enumerate(initial_guesses):
        if time.time() > deadline:
            break
            
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            
            if res.success or res.fun < best_score:
                min_area = -res.fun
                if min_area > 0:
                    points = res.x.reshape(N, 2)
                    # Verify points are inside
                    if all(_in_triangle(p) for p in points):
                        if min_area > best_score:
                            best_score = min_area
                            best_points = points
        except Exception:
            continue

    if best_points is None:
        # Fallback: return the first random guess
        return np.array(initial_guesses[0]).reshape(N, 2)
        
    return best_points
