import numpy as np
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill
    
    # Initial positions: Hex lattice
    # Generate R=2 hex lattice points
    points = []
    # Center
    points.append([0.0, 0.0])
    # Ring 1
    for i in range(6):
        angle = i * np.pi / 3
        points.append([np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)])
    # Ring 2
    # We need 5 more to make 12.
    # The 12 points in ring 2 are at distance 2*sqrt(3)
    # Angles: 0, 30, 60, ...
    # Let's pick 5 that are roughly symmetric
    # Indices 0, 2, 4, 6, 8 correspond to angles 0, 60, 120, 180, 240
    # This is a "star" pattern, leaving gaps at 30, 90, etc.
    # Or 0, 1, 2, 3, 4 (contiguous)
    # Let's try contiguous first.
    for i in range(5):
        angle = i * np.pi / 6
        points.append([2 * np.sqrt(3) * np.cos(angle), 2 * np.sqrt(3) * np.sin(angle)])
        
    points = np.array(points)
    
    # Function to calculate side length S
    def calc_side(points, phi):
        # Normals for the 6 sides of the outer hexagon
        # Rotated by phi
        # Normal angles: 0, 60, ... + phi
        # Wait, the normals to the sides of a regular hexagon are separated by 60 degrees.
        # If the hexagon is rotated by phi, the normals are at phi + k*pi/3.
        angles = np.arange(6) * np.pi / 3 + phi
        n_k = np.array([[np.cos(a), np.sin(a)] for a in angles])
        
        # Projections of points onto normals
        proj = points @ n_k.T
        
        # Max projection in each direction
        max_proj = np.max(proj, axis=0)
        
        # Apothem of enclosing hexagon for centers
        A_c = np.max(max_proj)
        
        # Add apothem of unit hexagon
        A = A_c + np.sqrt(3) / 2
        
        # Side length
        S = 2 * A / np.sqrt(3)
        
        return S

    # Function to check overlap
    def check_overlap(points):
        min_dist = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points[i] - points[j])
                if d < min_dist:
                    min_dist = d
        return min_dist

    # Local search
    best_S = calc_side(points, 0.0)
    best_points = points.copy()
    best_phi = 0.0
    
    # Optimize phi first
    for phi in np.linspace(0, np.pi/3, 200):
        S = calc_side(points, phi)
        if S < best_S:
            best_S = S
            best_phi = phi
            
    # Optimize positions with small perturbations
    step_size = 0.05
    for iter in range(1000):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    new_points = best_points.copy()
                    new_points[i] += [dx, dy]
                    if check_overlap(new_points) < np.sqrt(3) - 1e-6:
                        continue
                    S = calc_side(new_points, best_phi)
                    if S < best_S - 1e-6:
                        best_S = S
                        best_points = new_points
                        improved = True
        if not improved:
            step_size *= 0.9
            if step_size < 1e-4:
                break
                
    # Optimize center of outer hexagon
    def optimize_center(points, phi):
        cx, cy = 0.0, 0.0
        best_S = float('inf')
        step = 0.1
        for _ in range(100):
            improved = False
            for dcx in [-step, 0, step]:
                for dcy in [-step, 0, step]:
                    new_cx, new_cy = cx + dcx, cy + dcy
                    shifted = points - np.array([new_cx, new_cy])
                    S = calc_side(shifted, phi)
                    if S < best_S:
                        best_S = S
                        cx, cy = new_cx, new_cy
                        improved = True
            if not improved:
                step *= 0.5
                if step < 1e-4:
                    break
        return cx, cy, best_S

    cx, cy, best_S = optimize_center(best_points, best_phi)
    final_points = best_points - np.array([cx, cy])
    
    # Construct output
    inner = []
    for p in final_points:
        inner.append([p[0], p[1], 0.0]) # Angle 0
        
    return inner, (cx, cy), best_S, best_phi * 180 / np.pi
# EVOLVE-BLOCK-END
