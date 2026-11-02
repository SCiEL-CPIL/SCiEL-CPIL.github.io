"""Heilbronn triangle problem: maximize minimum triangle area for 11 points."""
import numpy as np
import time
import random
import math

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # seconds, safety margin before 150s limit

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_area = float('inf')
    # Vectorized calculation for speed
    # We calculate area for all triplets
    # To be safe and simple in Python without too much overhead for N=11 (165 triples)
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
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _structured_initial_guess():
    """
    Generate a highly symmetric initial guess.
    The optimal solution for Heilbronn triangles often exhibits symmetry.
    For N=11, a likely configuration is:
    - 3 points on each side (excluding vertices which might be better used internally or on edges)
    - Or a central cluster.
    
    Let's try a symmetric distribution:
    1. Points on the edges: Divide each edge into segments.
       If we put 3 points on each edge, that's 9 points.
       Plus 2 central points.
       
       Edge (0,0) to (1,0): (1/4, 0), (1/2, 0), (3/4, 0)
       Edge (1,0) to (0.5, S3H): 
           Vector is (-0.5, S3H). 
           Points: (1,0) + t*(-0.5, S3H) for t in {0.25, 0.5, 0.75}
       Edge (0.5, S3H) to (0,0):
           Vector is (-0.5, -S3H).
           Points: (0.5, S3H) + t*(-0.5, -S3H) for t in {0.25, 0.5, 0.75}
       
       Central points: Centroid (0.5, S3H/3) and maybe a point slightly offset? 
       Or two points near the centroid.
    """
    pts = []
    
    # Bottom edge
    pts.append([0.25, 0.0])
    pts.append([0.50, 0.0])
    pts.append([0.75, 0.0])
    
    # Right edge
    # Start (1,0), End (0.5, S3H)
    # P(t) = (1 - 0.5t, S3H*t)
    pts.append([1 - 0.5*0.25, S3H*0.25])
    pts.append([1 - 0.5*0.50, S3H*0.50])
    pts.append([1 - 0.5*0.75, S3H*0.75])
    
    # Left edge
    # Start (0.5, S3H), End (0,0)
    # P(t) = (0.5 - 0.5t, S3H - S3H*t)
    pts.append([0.5 - 0.5*0.25, S3H - S3H*0.25])
    pts.append([0.5 - 0.5*0.50, S3H - S3H*0.50])
    pts.append([0.5 - 0.5*0.75, S3H - S3H*0.75])
    
    # Central points
    # Centroid is (0.5, S3H/3)
    # Let's place two points near the centroid, symmetric with respect to the vertical axis x=0.5
    # Maybe slightly above and below? Or left and right?
    # Given the symmetry of the triangle, a vertical pair or horizontal pair?
    # The "center" of the triangle is (0.5, S3H/3).
    # Let's try two points very close to the centroid, slightly separated.
    # Or perhaps one at centroid and one elsewhere? 
    # Actually, for N=11, a known good configuration might have points at the midpoints of the medians?
    # Let's just start with two points near the centroid.
    pts.append([0.5 - 0.05, S3H/3])
    pts.append([0.5 + 0.05, S3H/3])
    
    return np.array(pts)

def construct():
    """Construct 11 points in the unit equilateral triangle to maximize min triangle area."""
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = _structured_initial_guess()
    best_pts = _project_to_triangle(best_pts)
    best_area = _min_triangle_area(best_pts)
    
    # Simulated Annealing
    current_pts = best_pts.copy()
    current_area = best_area
    
    T = 1e-4  # Initial temperature (scale of area change)
    alpha = 0.99  # Cooling rate
    min_T = 1e-8
    
    step_size = 0.05
    
    while time.time() - start_time < DEADLINE:
        # Check if we need to cool down
        # We do a fixed number of steps per "epoch" or just continuous
        # Let's do a batch of steps to reduce overhead checks
        
        for _ in range(100):
            if time.time() - start_time > DEADLINE:
                break
                
            # Generate a new candidate by perturbing one random point
            idx = rng.randint(0, N - 1)
            new_pts = current_pts.copy()
            
            # Perturb point idx
            # Use a Gaussian step, scaled by temperature
            # But we must stay inside the triangle.
            # Strategy: Perturb, then project. If projection changes it significantly, it's on boundary.
            
            dx = rng.gauss(0, step_size * T * 100) # Scale T to be meaningful for coords
            dy = rng.gauss(0, step_size * T * 100)
            
            new_pts[idx, 0] += dx
            new_pts[idx, 1] += dy
            
            # Project to triangle
            new_pts = _project_to_triangle(new_pts)
            
            # Check if it's actually different (avoid no-ops if on boundary)
            # For now, just evaluate.
            
            new_area = _min_triangle_area(new_pts)
            
            # Acceptance criterion
            if new_area > current_area:
                # Always accept improvement
                current_pts = new_pts
                current_area = new_area
                
                if current_area > best_area:
                    best_area = current_area
                    best_pts = current_pts.copy()
            else:
                # Accept worse with probability exp(-delta / T)
                delta = current_area - new_area
                if delta > 0:
                    prob = math.exp(-delta / T)
                    if rng.random() < prob:
                        current_pts = new_pts
                        current_area = new_area
                # else reject
                
        # Cool down
        T *= alpha
        if T < min_T:
            # Reheat slightly to escape local minima?
            # Or just stop if time is running out
            if time.time() - start_time < DEADLINE - 5:
                T = 1e-5 # Reheat
                # Random restart from best?
                # Let's perturb best significantly
                current_pts = best_pts.copy()
                for i in range(N):
                    current_pts[i, 0] += rng.gauss(0, 0.1)
                    current_pts[i, 1] += rng.gauss(0, 0.1)
                current_pts = _project_to_triangle(current_pts)
                current_area = _min_triangle_area(current_pts)

    return best_pts

if __name__ == "__main__":
    # Quick test
    pts = construct()
    print("Best Area:", _min_triangle_area(pts))
    print("Points:", pts)
