"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Honeycomb lattice basis vectors for hexagon centers
    # Adjacent hexagon centers are sqrt(3) apart
    a = np.sqrt(3.0)
    e1 = np.array([a, 0.0])
    e2 = np.array([a/2.0, a * np.sqrt(3.0)/2.0])
    
    # Generate all lattice points in a reasonable range
    # For 11 hexagons, we need at most ~5x5 grid
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * e1 + j * e2
            lattice_points.append(p)
    lattice_points = np.array(lattice_points)
    
    # Precompute pairwise distances to filter valid pairs (non-overlapping)
    # Two hexagons overlap if center distance < 2 * circumradius = 2? 
    # No - for hexagons of side 1, the minimum non-overlapping center distance depends on orientation.
    # For aligned hexagons on honeycomb lattice, distance sqrt(3) is the minimum.
    # For rotated hexagons, we need to check more carefully.
    # For simplicity, assume honeycomb lattice arrangement where all hexagons are axis-aligned.
    # In that case, minimum center distance for non-overlap is sqrt(3).
    
    min_dist = np.sqrt(3.0) - 1e-9
    
    # Compute all pairs that are too close
    n_pts = len(lattice_points)
    valid_pairs = np.ones((n_pts, n_pts), dtype=bool)
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            d = np.linalg.norm(lattice_points[i] - lattice_points[j])
            if d < min_dist:
                valid_pairs[i, j] = False
                valid_pairs[j, i] = False
    
    # Search for configurations of 11 points from the lattice that are all pairwise non-overlapping
    # Use a greedy + local search approach for speed
    
    best_side = float('inf')
    best_config = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    def compute_enclosing_hexagon(points, angle_degrees):
        """Compute minimal side length of regular hexagon with given rotation that contains all points."""
        theta = np.radians(angle_degrees)
        # 6 normal directions
        normals = np.array([np.array([np.cos(theta + k * np.pi/3), np.sin(theta + k * np.pi/3)]) for k in range(6)])
        
        # Center the points to minimize the enclosing hexagon
        # Optimal center is the midpoint of the bounding box in each normal direction
        # For a regular hexagon, the center that minimizes max projection is the average of min and max in each direction
        # But it must be consistent across all 6 directions.
        # Simple approach: try center = centroid, and center = midpoint of overall bounding box
        
        c1 = np.mean(points, axis=0)
        c2 = (points.max(axis=0) + points.min(axis=0)) / 2.0
        
        best_s = float('inf')
        for c in [c1, c2]:
            shifted = points - c
            # For each normal, compute max absolute projection
            projections = shifted @ normals.T  # shape (n, 6)
            max_abs = np.max(np.abs(projections), axis=0)  # shape (6,)
            # s * sqrt(3)/2 >= max_abs[k] for all k
            s = 2.0 / np.sqrt(3.0) * np.max(max_abs)
            if s < best_s:
                best_s = s
        
        return best_s, c
    
    def evaluate_config(points):
        """Try multiple angles and find best enclosing hexagon side."""
        best_s = float('inf')
        best_angle = 0.0
        best_c = None
        
        # Try 12 angles (every 15 degrees) plus some fine-tuning around best
        angles = np.arange(0, 180, 15.0)
        for ang in angles:
            s, c = compute_enclosing_hexagon(points, ang)
            if s < best_s:
                best_s = s
                best_angle = ang
                best_c = c
        
        # Fine-tune around best angle
        if best_angle > 0:
            fine_angles = np.linspace(best_angle - 15, best_angle + 15, 31)
            for ang in fine_angles:
                s, c = compute_enclosing_hexagon(points, ang)
                if s < best_s:
                    best_s = s
                    best_angle = ang
                    best_c = c
        
        return best_s, best_angle, best_c
    
    # Generate candidate configurations
    # Strategy: start with compact clusters and grow
    
    # Configuration 1: Center + 6 neighbors + 4 from next ring
    # Honeycomb lattice: point (i,j) -> i*e1 + j*e2
    # Neighbors of (0,0): (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # Next ring: (2,0), (-2,0), (0,2), (0,-2), (2,-1), (-2,1), (1,1), (-1,-1)
    
    # Let's enumerate several candidate shapes
    candidates = []
    
    # Shape 1: Hexagonal number pattern - 1 + 6 + 4 = 11
    # Center (0,0), 6 neighbors, then 4 from the 8 in next ring
    center = np.array([0.0, 0.0])
    neighbors = [
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)
    ]
    next_ring = [
        (2, 0), (-2, 0), (0, 2), (0, -2), (2, -1), (-2, 1), (1, 1), (-1, -1)
    ]
    
    # Try all combinations of 4 from next_ring added to center + 6 neighbors
    from itertools import combinations
    base = [center] + [np.array([i, j], dtype=float) @ np.array([[a, a/2], [0, a*np.sqrt(3)/2]]) for i, j in neighbors]
    
    # Actually, let me just work in lattice coordinates and convert
    def lattice_to_cartesian(i, j):
        return np.array([i * a + j * a/2.0, j * a * np.sqrt(3.0)/2.0])
    
    base_coords = [(0,0)] + neighbors
    
    for combo in combinations(next_ring, 4):
        shape = base_coords + list(combo)
        candidates.append(shape)
    
    # Shape 2: More linear arrangements
    # 3x4 rectangle minus 1
    for offset in range(-1, 2):
        shape2 = []
        for i in range(4):
            for j in range(3):
                if i + j != 0:  # remove one corner
                    shape2.append((i-1, j-1 + offset))
        if len(shape2) == 11:
            candidates.append(shape2)
    
    # Shape 3: 2 rows of 5 + 1
    for shift in range(-2, 3):
        shape3 = []
        for i in range(5):
            shape3.append((i-2, 0))
        for i in range(5):
            shape3.append((i-2+shift, 1))
        shape3.append((0, 2))
        if len(shape3) == 11:
            candidates.append(shape3)
    
    # Shape 4: Compact - 1+6+4 but different 4
    # Already covered above
    
    # Evaluate all candidates
    for idx, shape in enumerate(candidates):
        if time.time() > deadline - 5.0:
            break
        
        points = np.array([lattice_to_cartesian(i, j) for i, j in shape])
        
        # Check non-overlap: all pairwise distances >= sqrt(3)
        valid = True
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if np.linalg.norm(points[i] - points[j]) < min_dist:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        s, ang, c = evaluate_config(points)
        
        if s < best_side:
            best_side = s
            best_config = points.copy()
            best_angle = ang
            best_center = c
    
    # If no valid configuration found, use a simple linear arrangement
    if best_config is None:
        # Linear arrangement along x-axis
        points = np.array([[k * np.sqrt(3), 0.0] for k in range(-5, 6)])
        s, ang, c = evaluate_config(points)
        best_side = s
        best_config = points
        best_angle = ang
        best_center = c
    
    # Build inner hexagon positions
    # Each inner hexagon is at its center with 0 rotation (aligned with lattice)
    inner = []
    for p in best_config:
        inner.append([float(p[0]), float(p[1]), 0.0])
    
    # The outer hexagon center and angle
    outer_center = (float(best_center[0]), float(best_center[1]))
    outer_angle = float(best_angle)
    outer_side = float(best_side)
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
