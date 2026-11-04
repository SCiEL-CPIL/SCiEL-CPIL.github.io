import numpy as np
from itertools import combinations
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Lattice generation
    # Axial coordinates (q, r)
    # Cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    # Distance squared = 3*(q^2 + qr + r^2)
    
    points = []
    # Generate points up to axial radius 3
    for q in range(-3, 4):
        for r in range(-3, 4):
            if abs(q) <= 3 and abs(r) <= 3 and abs(-q-r) <= 3:
                dist_sq = 3 * (q*q + q*r + r*r)
                x = np.sqrt(3) * (q + r/2.0)
                y = 1.5 * r
                points.append((x, y, q, r, dist_sq))
    
    # Sort by distance from center
    points.sort(key=lambda p: p[4])
    
    # The 7 central points (dist_sq 0 and 3)
    # dist_sq 0: (0,0)
    # dist_sq 3: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    core_points = []
    extra_candidates = []
    
    for p in points:
        if p[4] <= 3:
            core_points.append(p)
        elif p[4] <= 12: # dist_sq 12 is dist sqrt(12) ~ 3.46? No, dist_sq 12 is radius 2 in axial?
            # Axial radius 2 means max(|q|,|r|,|q+r|) = 2.
            # Max dist_sq for axial radius 2: (2,0) -> 3*4=12. (1,1) -> 3*3=9.
            # So dist_sq 9 and 12 are the next ring.
            extra_candidates.append(p)
            
    # We need 11 points total. We have 7 core. Need 4 extras.
    # Try combinations of 4 from extra_candidates
    # To limit search, only consider the closest extra candidates
    # There are 18 points in the ring of axial radius 2?
    # Let's just take the first 12 closest extra candidates to keep combinations manageable
    extra_candidates = extra_candidates[:12]
    
    best_L = float('inf')
    best_config = None
    
    # Helper to compute min enclosing hexagon side for a set of points
    def min_hex_side(pts):
        # pts is list of (x, y)
        if not pts:
            return 0.0
        # Optimize angle phi in [0, pi/6] due to symmetry
        # S(phi) = (2/sqrt(3)) * max_i |proj_{u_i}(p)|
        # u_i are normals to sides.
        # If hexagon is rotated by phi, normals are at phi, phi+pi/3, phi+2pi/3.
        # Wait, normals are perpendicular to sides.
        # Sides are at angles phi, phi+pi/3...
        # Normals are at phi+pi/2...
        
        min_val = float('inf')
        # Scan angles
        # 100 steps might be enough for precision
        for k in range(100):
            phi = k * np.pi / 600.0 # 0 to 30 deg
            # Normals angles
            n1 = phi + np.pi/2
            n2 = n1 + np.pi/3
            n3 = n2 + np.pi/3
            
            max_proj = 0.0
            for x, y in pts:
                # Projection onto n1, n2, n3
                # We can just check one from each pair? No, |proj|
                p1 = abs(x * np.cos(n1) + y * np.sin(n1))
                p2 = abs(x * np.cos(n2) + y * np.sin(n2))
                p3 = abs(x * np.cos(n3) + y * np.sin(n3))
                if p1 > max_proj: max_proj = p1
                if p2 > max_proj: max_proj = p2
                if p3 > max_proj: max_proj = p3
            
            S = (2.0 / np.sqrt(3)) * max_proj
            if S < min_val:
                min_val = S
        
        return min_val

    # Iterate over combinations of 4 extra points
    # If extra_candidates is large, this might be slow.
    # 12 choose 4 = 495. 495 * 100 angle steps * 11 points is fast.
    
    combos = list(combinations(range(len(extra_candidates)), 4))
    
    for i, combo in enumerate(combos):
        if time.time() > deadline:
            break
            
        selected_extras = [extra_candidates[j] for j in combo]
        all_pts = [ (p[0], p[1]) for p in core_points ] + [ (p[0], p[1]) for p in selected_extras ]
        
        S_core = min_hex_side(all_pts)
        L = S_core + 1.0
        
        if L < best_L:
            best_L = L
            # Store config
            # inner: list of (x, y, angle)
            # angle is 0 for all (aligned)
            inner = [ (p[0], p[1], 0.0) for p in all_pts ]
            best_config = inner

    if best_config is None:
        # Fallback: just the core 7 + 4 random? Or return seed
        # If no combo found, use core 7 + first 4 extras
        selected_extras = extra_candidates[:4]
        all_pts = [ (p[0], p[1]) for p in core_points ] + [ (p[0], p[1]) for p in selected_extras ]
        inner = [ (p[0], p[1], 0.0) for p in all_pts ]
        S_core = min_hex_side(all_pts)
        best_L = S_core + 1.0
        best_config = inner

    # Center is (0,0)
    # Side length is best_L
    # Outer angle is 0.0
    
    return best_config, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
