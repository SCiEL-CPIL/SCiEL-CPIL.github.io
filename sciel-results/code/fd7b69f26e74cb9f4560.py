import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit to be safe
    
    # Helper to generate hex grid points
    def get_ring_points(ring):
        points = []
        if ring == 0:
            return [(0, 0)]
        
        # Start at (ring, 0)
        q, r = ring, 0
        for _ in range(6):
            for _ in range(ring):
                points.append((q, r))
                # Move in the direction of the next side
                # Directions: (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1), (1, 0)
                # This standard loop might be tricky. Let's use a simpler generator.
                pass
        
        # Simpler generator for ring k
        # Directions:
        # 1: (1, 0)
        # 2: (0, 1) -> No, axial coords
        # Let's just iterate q, r and check distance
        # d(q,r) = (|q| + |r| + |q+r|)/2
        # Loop range: -ring to ring
        for q in range(-ring, ring + 1):
            for r in range(-ring, ring + 1):
                if (abs(q) + abs(r) + abs(q + r)) // 2 == ring:
                    points.append((q, r))
        return points

    # Generate candidate configurations
    # Config 1: 7 center + 4 Ring 2 (spread)
    # Config 2: 7 center + 4 Ring 2 (clustered)
    # Config 3: Maybe 5 center + 6 Ring 1? No, 7 center is best core.
    
    ring1 = get_ring_points(1)
    ring2 = get_ring_points(2)
    
    # Base: Center + Ring 1
    base_points = [(0, 0)] + ring1
    
    # We need 4 more from Ring 2
    # Strategy: Pick 4 points from Ring 2 that minimize the bounding box.
    # Since Ring 2 is symmetric, picking any 4 "equivalent" points should yield similar results.
    # However, picking adjacent points makes the shape elongated.
    # Picking spread-out points makes it more compact.
    
    # Let's try picking indices 0, 3, 6, 9 (opposite pairs)
    # And 0, 2, 4, 6 (every other)
    
    candidates = []
    
    # Candidate A: 4 alternating Ring 2 points
    # Ring 2 has 12 points.
    # Indices 0, 2, 4, 6
    idx_a = [0, 2, 4, 6]
    points_a = base_points + [ring2[i] for i in idx_a]
    candidates.append(points_a)
    
    # Candidate B: 4 opposite pairs (0, 3, 6, 9)
    idx_b = [0, 3, 6, 9]
    points_b = base_points + [ring2[i] for i in idx_b]
    candidates.append(points_b)
    
    # Candidate C: 4 adjacent (0, 1, 2, 3) - likely worse
    idx_c = [0, 1, 2, 3]
    points_c = base_points + [ring2[i] for i in idx_c]
    candidates.append(points_c)
    
    # Helper to calculate min side length S
    def calc_min_side(points):
        # Convert axial to Cartesian
        # x = sqrt(3) * (q + r/2)
        # y = 1.5 * r
        pts = []
        for q, r in points:
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        pts = np.array(pts)
        
        # Center the points
        center = np.mean(pts, axis=0)
        pts_centered = pts - center
        
        # Define 6 normal directions for the hexagon sides
        # The sides are at angles 0, 60, 120, 180, 240, 300
        # Normals are at 30, 90, 150, 210, 270, 330?
        # Let's use the standard hexagon inequalities.
        # For a hexagon with flat top (vertices at 0, 60...), the normals are at 30, 90...
        # Wait, if vertices are at 0, 60..., the sides are between 0-60, etc.
        # Normal to side 0-60 is at 30 degrees.
        angles = np.deg2rad(np.arange(30, 360, 60))
        normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
        
        # Project points onto normals
        # projections[i, j] is projection of point i onto normal j
        projections = pts_centered @ normals.T
        
        # For each normal, find max and min projection
        max_projs = np.max(projections, axis=0)
        min_projs = np.min(projections, axis=0)
        
        # Width in each direction
        widths = max_projs - min_projs
        
        # The width of the container hexagon in these directions is S * sqrt(3)
        # So S >= max(widths) / sqrt(3)
        S = np.max(widths) / np.sqrt(3)
        
        return S

    best_S = float('inf')
    best_config = None
    
    for config in candidates:
        S = calc_min_side(config)
        if S < best_S:
            best_S = S
            best_config = config
            
        if time.time() > deadline:
            break

    # If we haven't found a good one, or just to be safe, use the best found
    # The seed returned 40.0. We expect ~4.5.
    
    # Prepare output
    # inner: list of [x, y, angle_degrees]
    # We use angle 0 for all.
    inner = []
    for q, r in best_config:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([float(x), float(y), 0.0])
        
    # Outer hexagon center
    # We centered the points, so the center is the mean of the original points
    pts = np.array([[np.sqrt(3) * (q + r / 2.0), 1.5 * r] for q, r in best_config])
    center = tuple(map(float, np.mean(pts, axis=0)))
    
    return inner, center, float(best_S), 0.0

# EVOLVE-BLOCK-START
# (The function construct is defined above)
# EVOLVE-BLOCK-END
