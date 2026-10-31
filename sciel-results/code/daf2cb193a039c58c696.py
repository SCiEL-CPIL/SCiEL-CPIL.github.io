import numpy as np
import itertools
import math
import time

def construct():
    start_time = time.time()
    deadline = 150.0
    
    # Lattice vectors for flat-top hexagons
    # Distance between centers is sqrt(3)
    # Angles of neighbors: 30, 90, 150, 210, 270, 330
    # u = (1.5, sqrt(3)/2) -> 30 deg
    # v = (-1.5, sqrt(3)/2) -> 150 deg
    # u+v = (0, sqrt(3)) -> 90 deg
    
    u = np.array([1.5, math.sqrt(3)/2])
    v = np.array([-1.5, math.sqrt(3)/2])
    
    # Generate a set of candidate lattice points
    # We need 11 points.
    # Center + 6 neighbors = 7.
    # Next ring has 12 points.
    # We will select 4 from the next ring.
    
    # Coordinates in lattice basis (i, j)
    # Point = i*u + j*v
    
    # 7 central points
    center_points = [(0,0)]
    # Neighbors (i,j) such that i*u + j*v is a neighbor
    # Neighbors are (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) ?
    # Let's verify:
    # (1,0) -> u
    # (0,1) -> v
    # (-1,1) -> -u+v = u+v? No. -u+v = (-1.5, -sqrt(3)/2) + (-1.5, sqrt(3)/2) = (-3, 0). This is 180 deg.
    # My previous analysis: Neighbors are (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    # Let's check (-1,1): -u+v = (-3, 0). Angle 180.
    # (1,-1): u-v = (3, 0). Angle 0.
    # (1,1): u+v = (0, sqrt(3)). Angle 90.
    # (-1,-1): -u-v = (0, -sqrt(3)). Angle 270.
    # So the 6 neighbors are (1,0), (0,1), (1,1), (-1,-1), (-1,0), (0,-1).
    # Wait, (1,1) is a neighbor.
    # So the 7 points are:
    # (0,0)
    # (1,0), (0,1), (1,1)
    # (-1,0), (0,-1), (-1,-1)
    
    base_points = [(0,0), (1,0), (0,1), (1,1), (-1,0), (0,-1), (-1,-1)]
    
    # Next ring candidates
    # Points with hex distance 2
    # These are (i,j) such that max(|i|,|j|,|i-j|) == 2 ?
    # Or just generate points in a larger range and filter by distance.
    
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            if (i,j) in base_points:
                continue
            # Calculate Euclidean distance to origin
            p = i*u + j*v
            dist = np.linalg.norm(p)
            # Keep points that are reasonably close
            if dist < 5.0:
                candidates.append((i,j))
                
    # We need to choose 4 from candidates to add to base_points
    # To minimize the bounding box, we should try combinations.
    # 12 candidates, choose 4. C(12,4) = 495. Very fast.
    
    best_side = float('inf')
    best_config = None
    best_angle = 0.0
    
    # Pre-compute Cartesian coordinates for candidates
    cand_coords = {c: c[0]*u + c[1]*v for c in candidates}
    base_coords = {b: b[0]*u + b[1]*v for b in base_points}
    
    # Function to calculate required side length S for a set of points
    # Outer hexagon is flat-top, centered at (0,0), rotated by angle theta.
    # Due to symmetry, we only need to check a few angles.
    # The optimal rotation is likely 0 or 30 degrees relative to the lattice.
    
    def required_side(points, theta):
        # Rotate points by -theta
        c = math.cos(-theta)
        s = math.sin(-theta)
        min_side = 0.0
        for p in points:
            x = p[0]
            y = p[1]
            # Rotate
            rx = x*c - y*s
            ry = x*s + y*c
            # Constraints for flat-top hexagon side S:
            # |rx| <= S
            # |ry| <= sqrt(3)/2 * (S - |rx|) ? No.
            # |ry| <= sqrt(3)/2 * S
            # |sqrt(3)rx + ry| <= sqrt(3)S
            # |-sqrt(3)rx + ry| <= sqrt(3)S
            
            # From |ry| <= sqrt(3)/2 S:
            S1 = 2.0 * abs(ry) / math.sqrt(3)
            
            # From |sqrt(3)rx + ry| <= sqrt(3)S:
            S2 = abs(math.sqrt(3)*rx + ry) / math.sqrt(3)
            
            # From |-sqrt(3)rx + ry| <= sqrt(3)S:
            S3 = abs(-math.sqrt(3)*rx + ry) / math.sqrt(3)
            
            # From |rx| <= S:
            S4 = abs(rx)
            
            min_side = max(min_side, S1, S2, S3, S4)
        return min_side

    # Try a few rotations
    angles = [0.0, math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3, 5*math.pi/6]
    
    for combo in itertools.combinations(candidates, 4):
        if time.time() > deadline - 10:
            break
        points = list(base_coords.values()) + [cand_coords[c] for c in combo]
        for theta in angles:
            s = required_side(points, theta)
            if s < best_side:
                best_side = s
                best_config = points
                best_angle = theta
                
    # Convert best_config back to lattice indices for output
    # We need to return (x, y, angle_degrees) for each inner hexagon
    # The angle of the inner hexagon is 0 (flat-top)
    
    inner = []
    for p in best_config:
        inner.append([float(p[0]), float(p[1]), 0.0])
        
    return inner, (0.0, 0.0), float(best_side), float(best_angle)

construct()
