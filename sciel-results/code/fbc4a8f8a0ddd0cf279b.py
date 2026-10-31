import numpy as np
import time
from itertools import combinations

N = 11
# Triangle vertices
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, np.sqrt(3)/2])

def triangle_area(p1, p2, p3):
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def min_triangle_area(points):
    min_a = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = triangle_area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def is_inside_triangle(p):
    # Barycentric coordinate method
    x, y = p
    # Area of triangle
    area = 0.5 * np.sqrt(3)
    # Area of sub-triangles
    a1 = triangle_area(V1, V2, p)
    a2 = triangle_area(V2, V3, p)
    a3 = triangle_area(V3, V1, p)
    # Check if sum of areas equals total area and no area is negative (which indicates outside for this convex shape)
    # Actually, for a point inside, the sum of areas is the total area.
    # A more robust check:
    # Barycentric coords
    # x = b1*x1 + b2*x2 + b3*x3
    # y = b1*y1 + b2*y2 + b3*y3
    # b1+b2+b3 = 1
    # b_i >= 0
    
    # Using the area method is fine if we check:
    # a1 + a2 + a3 == area (within tolerance)
    # and a1, a2, a3 >= -epsilon
    
    # Let's use a simpler check based on half-planes
    # Edge V1-V2: y >= 0
    # Edge V2-V3: y <= -sqrt(3)*(x-1)  => sqrt(3)*x + y - sqrt(3) <= 0
    # Edge V3-V1: y <= sqrt(3)*x      => -sqrt(3)*x + y <= 0
    
    eps = 1e-9
    if y < -eps: return False
    if np.sqrt(3)*x + y - np.sqrt(3) > eps: return False
    if -np.sqrt(3)*x + y > eps: return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_points = None
    best_val = -1
    
    # Directions for local search
    # 8 directions + stay
    directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)]
    directions = [np.array(d, dtype=float) / np.linalg.norm(d) for d in directions]
    
    def local_search(points, max_iters=5000):
        points = np.array(points, dtype=float)
        current_val = min_triangle_area(points)
        
        # Adaptive step size
        step = 0.1
        no_improve_count = 0
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                best_move = None
                best_new_val = current_val
                
                # Try different step sizes
                for s in [step, step*0.5, step*0.1]:
                    for d in directions:
                        new_p = points[i] + s * d
                        if not is_inside_triangle(new_p):
                            continue
                        
                        # Calculate new min area
                        # Only triangles involving i change
                        # We can just recalculate all for simplicity since N is small
                        # But let's optimize slightly:
                        # Actually, with N=11, 165 triangles is fast.
                        new_points = points.copy()
                        new_points[i] = new_p
                        new_val = min_triangle_area(new_points)
                        
                        if new_val > best_new_val:
                            best_new_val = new_val
                            best_move = new_p
                
                if best_move is not None and best_new_val > current_val:
                    points[i] = best_move
                    current_val = best_new_val
                    improved = True
                    # Reset no_improve count
                    no_improve_count = 0
                    # Optionally shrink step if we are improving a lot?
                    # Or keep step same.
                    
            if not improved:
                # Shrink step size
                step *= 0.5
                no_improve_count += 1
                if step < 1e-5:
                    break
        
        return points, current_val

    # Run multiple restarts
    rng = np.random.RandomState(42)
    
    for restart in range(20): # Up to 20 restarts
        if time.time() > deadline:
            break
            
        # Initialize points
        # Strategy: Random points, but ensure some spread
        # Or use a grid
        if restart == 0:
            # Initial grid-like structure
            # 3x3 grid + 2 corners?
            # Let's just use random for now, it's robust
            points = []
            while len(points) < N:
                u, v = rng.random(), rng.random()
                if u + v < 1.0:
                    p = np.array([u + 0.5*v, np.sqrt(3)/2 * v])
                    points.append(p)
            points = np.array(points)
        else:
            # Perturb the best found so far? Or new random?
            # New random with different seed
            points = []
            while len(points) < N:
                u, v = rng.random(), rng.random()
                if u + v < 1.0:
                    p = np.array([u + 0.5*v, np.sqrt(3)/2 * v])
                    points.append(p)
            points = np.array(points)
            
        # Run local search
        opt_points, val = local_search(points)
        
        if val > best_val:
            best_val = val
            best_points = opt_points
            
        # If we found a very good value, we can stop early?
        # The theoretical max is unknown, but we can set a threshold.
        # For now, just run all restarts if time permits.
        
    if best_points is None:
        # Fallback
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 1.0:
                p = np.array([u + 0.5*v, np.sqrt(3)/2 * v])
                points.append(p)
        best_points = np.array(points)
        
    return best_points
