import numpy as np
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit
    
    # Number of hexagons
    n = 11
    
    # Initial configuration: The "flower" pattern from the current program
    # Center
    centers = np.array([[0.0, 0.0]])
    # Ring 1
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3)
        centers = np.vstack([centers, [r * np.cos(angle), r * np.sin(angle)]])
    # Ring 2: 4 hexagons
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0
        centers = np.vstack([centers, [r * np.cos(angle), r * np.sin(angle)]])
    
    # All inner hexagons are rotated by 30 degrees (flat-top)
    # This means their "flat sides" are horizontal/vertical-ish?
    # Vertices at 30, 90, ...
    # If we assume all inner hexagons have the same orientation,
    # the non-overlap constraint is distance >= sqrt(3).
    
    best_side = float('inf')
    best_centers = centers.copy()
    best_outer_angle = 30.0  # degrees
    
    def calc_min_side(pts, outer_angle_deg):
        """
        Calculate the minimum side length S of a regular hexagon 
        centered at origin, rotated by outer_angle_deg, 
        that contains all points in pts.
        """
        phi = np.deg2rad(outer_angle_deg)
        # Normals to the sides of the outer hexagon
        # The sides are perpendicular to normals at phi, phi+60, phi+120
        # The distance from center to side is S * sqrt(3) / 2
        # Condition: |p . n| <= S * sqrt(3) / 2
        # So S >= |p . n| / (sqrt(3) / 2)
        
        n1 = np.array([np.cos(phi), np.sin(phi)])
        n2 = np.array([np.cos(phi + np.pi/3), np.sin(phi + np.pi/3)])
        n3 = np.array([np.cos(phi + 2*np.pi/3), np.sin(phi + 2*np.pi/3)])
        
        max_proj = 0.0
        for p in pts:
            proj1 = abs(np.dot(p, n1))
            proj2 = abs(np.dot(p, n2))
            proj3 = abs(np.dot(p, n3))
            max_proj = max(max_proj, proj1, proj2, proj3)
            
        return max_proj / (np.sqrt(3) / 2)

    def is_valid(pts, min_dist_sq):
        """
        Check if all pairwise distances are >= sqrt(min_dist_sq).
        """
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                d_sq = np.sum((pts[i] - pts[j])**2)
                if d_sq < min_dist_sq:
                    return False
        return True

    min_dist = np.sqrt(3)
    min_dist_sq = min_dist * min_dist
    eps = 1e-9
    
    # Local search
    # We will try to optimize the outer angle and the positions.
    # First, find the best outer angle for the initial configuration.
    best_angle_for_init = 30.0
    best_s_for_init = float('inf')
    for a in np.linspace(0, 60, 61):
        s = calc_min_side(centers, a)
        if s < best_s_for_init:
            best_s_for_init = s
            best_angle_for_init = a
            
    best_side = best_s_for_init
    best_centers = centers.copy()
    best_outer_angle = best_angle_for_init
    
    # Now, perform local search on positions
    # We can use a simple random walk with acceptance if S decreases.
    # To handle constraints, we can reject moves that cause overlap.
    
    current_centers = best_centers.copy()
    current_side = best_side
    current_angle = best_outer_angle
    
    # Temperature for simulated annealing? Or just greedy?
    # Greedy is safer.
    
    iterations = 0
    max_iterations = 500000  # High number, but we check time
    
    while time.time() - start_time < DEADLINE:
        iterations += 1
        
        # Try to perturb one or more centers
        # Perturb 1 center at a time
        idx = random.randint(0, n-1)
        delta = np.array([random.gauss(0, 0.1), random.gauss(0, 0.1)])
        new_centers = current_centers.copy()
        new_centers[idx] = new_centers[idx] + delta
        
        # Check validity
        if is_valid(new_centers, min_dist_sq):
            # Calculate new side length
            # We can either keep the same outer angle or re-optimize it.
            # Re-optimizing is better but slower.
            # Let's re-optimize angle for this new configuration.
            # Since n=11, checking 61 angles is fast.
            best_new_s = float('inf')
            best_new_a = current_angle
            for a in np.linspace(0, 60, 21): # Coarse search first
                s = calc_min_side(new_centers, a)
                if s < best_new_s:
                    best_new_s = s
                    best_new_a = a
            
            # Fine search around best_new_a
            if best_new_s < current_side:
                for a in np.linspace(best_new_a - 5, best_new_a + 5, 21):
                    if 0 <= a <= 60:
                        s = calc_min_side(new_centers, a)
                        if s < best_new_s:
                            best_new_s = s
                            best_new_a = a
                
                if best_new_s < current_side:
                    current_centers = new_centers
                    current_side = best_new_s
                    current_angle = best_new_a
                    best_centers = current_centers.copy()
                    best_side = current_side
                    best_outer_angle = current_angle
        
        # Occasionally, try to perturb the outer angle directly?
        # The position search implicitly optimizes it.
        
        # Occasionally, try a larger jump
        if iterations % 100 == 0:
            idx = random.randint(0, n-1)
            delta = np.array([random.gauss(0, 0.5), random.gauss(0, 0.5)])
            new_centers = current_centers.copy()
            new_centers[idx] = new_centers[idx] + delta
            if is_valid(new_centers, min_dist_sq):
                best_new_s = float('inf')
                best_new_a = current_angle
                for a in np.linspace(0, 60, 21):
                    s = calc_min_side(new_centers, a)
                    if s < best_new_s:
                        best_new_s = s
                        best_new_a = a
                if best_new_s < current_side:
                    current_centers = new_centers
                    current_side = best_new_s
                    current_angle = best_new_a
                    best_centers = current_centers.copy()
                    best_side = current_side
                    best_outer_angle = current_angle

    # Final high-precision polish
    # Re-optimize the outer angle with higher resolution
    final_side = float('inf')
    final_angle = best_outer_angle
    for a in np.linspace(0, 60, 601):
        s = calc_min_side(best_centers, a)
        if s < final_side:
            final_side = s
            final_angle = a
            
    # Construct the return values
    inner = []
    for x, y in best_centers:
        inner.append([x, y, 30.0])
        
    return inner, (0.0, 0.0), final_side, final_angle

