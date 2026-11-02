import numpy as np
import itertools

N = 11

def construct():
    # Generate lattice points
    # Axial coordinates (q, r)
    # Physical position: x = sqrt(3) * (q + r/2), y = sqrt(3) * (r * sqrt(3)/2) = 3/2 * r * sqrt(3)?
    # Wait.
    # Standard hex grid:
    # x = 3/2 * q
    # y = sqrt(3) * (r + q/2)
    # This is for pointy-top?
    # Let's use the one derived earlier:
    # u = (1, 0), v = (1/2, sqrt(3)/2) in scaled units.
    # x_scaled = q + r/2
    # y_scaled = r * sqrt(3)/2
    # Physical: x = sqrt(3) * x_scaled, y = sqrt(3) * y_scaled
    
    points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x_s = q + r/2.0
            y_s = r * np.sqrt(3)/2.0
            x_p = np.sqrt(3) * x_s
            y_p = np.sqrt(3) * y_s
            points.append((q, r, x_p, y_p))
            
    # Filter to ensure we have enough points and are close to center
    # We need 11 points.
    # The optimal set will be within radius 2 or 3.
    
    best_S = float('inf')
    best_set = None
    
    # Get indices of points
    # To reduce search space, only consider points within a certain distance
    # Max distance for S~4 is around 3.5.
    # Let's just use all points in the list (49 points).
    # C(49, 11) is too big.
    # But we can restrict to points with distance < 4.
    # Distance from origin:
    dists = [np.sqrt(x**2 + y**2) for _, _, x, y in points]
    # Sort by distance
    indexed_points = sorted(zip(dists, range(len(points))))
    
    # The 11 closest points are a good candidate.
    # Let's check the 11 closest points first.
    # Then maybe swap some?
    
    # Actually, just check the 11 closest points.
    # For N=11, the 11 closest are the center + ring 1 (6) + 4 from ring 2.
    # Which 4 from ring 2? The ones with smallest distance? All ring 2 have same distance.
    # So symmetry is broken.
    # Let's try the "cross" (4 axis-aligned) and the "cluster" (4 adjacent).
    
    # Cross: (2,0), (0,2), (-2,0), (0,-2)
    # Cluster: (2,0), (1,1), (0,2), (-1,1) ? No, (1,1) is ring 2?
    # (1,1) -> x=1.5, y=sqrt(3)/2. Dist^2 = 2.25 + 0.75 = 3. Dist=sqrt(3).
    # (2,0) -> x=2, y=0. Dist=2.
    # Wait, distance in scaled units:
    # (1,0) dist 1.
    # (1,1) dist sqrt(1.5^2 + (sqrt(3)/2)^2) = sqrt(2.25+0.75)=sqrt(3)=1.732.
    # (2,0) dist 2.
    # So ring 1 is dist 1. Ring 2 is dist sqrt(3) and 2?
    # (1,1) is dist sqrt(3).
    # (2,0) is dist 2.
    # So the 11 closest are:
    # 1 at dist 0.
    # 6 at dist 1.
    # 4 at dist sqrt(3). (There are 6 such points: (1,1), (1,-1), etc? No.
    # (1,1), (0,1)? No (0,1) is ring 1.
    # Ring 2 points:
    # (2,0) dist 2.
    # (1,1) dist sqrt(3).
    # (0,2) dist sqrt(3)? x=1, y=sqrt(3). Dist sqrt(1+3)=2.
    # So (1,1) is closer than (2,0).
    
    # Let's list ring 2 points and their distances:
    # (2,0): 2
    # (1,1): sqrt(3) ~ 1.732
    # (0,2): 2
    # (-1,1): sqrt(3)
    # (-2,0): 2
    # (-1,-1): sqrt(3)
    # (0,-2): 2
    # (1,-1): sqrt(3)
    # (2,-1): 2? x=1.5, y=-sqrt(3)/2. Dist sqrt(2.25+0.75)=sqrt(3).
    # So there are 6 points at dist sqrt(3) and 6 at dist 2.
    
    # The 11 closest are: Center + 6 Ring 1 + 4 of the 6 Ring 2 (dist sqrt(3)).
    
    # Let's test all combinations of 4 from the 6 "close" ring 2 points.
    # There are C(6,4) = 15 combinations.
    
    # Identify the 6 points at dist sqrt(3)
    close_ring2 = []
    for i, (q, r, x, y) in enumerate(points):
        d2 = x**2 + y**2
        if abs(d2 - 3) < 0.1: # dist^2 = 3
             close_ring2.append(i)
             
    # The 7 central points
    central = []
    for i, (q, r, x, y) in enumerate(points):
        d2 = x**2 + y**2
        if d2 < 2: # dist < 1.41, so dist 0 and 1
             central.append(i)
             
    # Combine
    candidates = []
    for combo in itertools.combinations(close_ring2, 4):
        candidates.append(central + list(combo))
        
    # Evaluate S for each candidate
    for cand in candidates:
        xs = [points[i][2] for i in cand]
        ys = [points[i][3] for i in cand]
        
        max_proj = 0
        # Normals at 30, 90, 150, 210, 270, 330
        for angle_deg in [30, 90, 150, 210, 270, 330]:
            angle = np.radians(angle_deg)
            nx = np.cos(angle)
            ny = np.sin(angle)
            for x, y in zip(xs, ys):
                proj = abs(nx * x + ny * y)
                if proj > max_proj:
                    max_proj = proj
                    
        S = max_proj + 0.5
        
        if S < best_S:
            best_S = S
            best_set = cand
            
    # Construct the return value
    inner = []
    for i in best_set:
        q, r, x, y = points[i]
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), best_S, 0.0
