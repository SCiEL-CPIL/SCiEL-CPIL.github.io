import math
import time
import numpy as np
from itertools import combinations
import random

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit
    
    # Precompute unit hexagon vertices (flat-top orientation)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    # Honeycomb lattice basis
    # v1 = (1.5, sqrt(3)/2)
    # v2 = (0, sqrt(3))
    sqrt3 = math.sqrt(3)
    
    def lattice_to_cartesian(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        return (x, y)
        
    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        # Normals to the sides of the outer hexagon
        # If outer hexagon is rotated by theta, its side normals are at theta + 30 + k*60
        normals = np.array([[math.cos(theta_rad + math.pi/6 + k * math.pi/3), 
                             math.sin(theta_rad + math.pi/6 + k * math.pi/3)] for k in range(6)])
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        # Side length s relates to inradius r by r = s * sqrt(3) / 2
        # So s = 2 * r / sqrt(3)
        return max_proj * 2.0 / math.sqrt(3)

    # Strategy: Generate "polyhexes" of size 12.
    # A polyhex is a connected set of hexagons on the honeycomb lattice.
    # By using connected sets, we ensure high density and compactness.
    # We will generate several distinct polyhex shapes (up to symmetry) and test them.
    
    # 1. Generate candidate polyhexes
    # We use a BFS/DFS growth method to find connected sets of 12 hexagons.
    # To avoid duplicates and keep it fast, we generate a few specific shapes:
    # - 2x3 rectangle (2 rows of 3)
    # - 3x2 rectangle (3 rows of 2)
    # - 1x12 line (worst case, but valid)
    # - "Star" shapes or compact blobs
    
    candidate_shapes = []
    
    # Helper to check if a set of (q,r) coords is connected
    def is_connected(coords):
        if not coords: return True
        coord_set = set(coords)
        start = next(iter(coord_set))
        visited = {start}
        stack = [start]
        while stack:
            q, r = stack.pop()
            # Neighbors in axial coordinates (q,r)
            neighbors = [
                (q+1, r), (q-1, r),
                (q, r+1), (q, r-1),
                (q+1, r-1), (q-1, r+1)
            ]
            for nq, nr in neighbors:
                if (nq, nr) in coord_set and (nq, nr) not in visited:
                    visited.add((nq, nr))
                    stack.append((nq, nr))
        return len(visited) == len(coords)
        
    # Helper to normalize a shape (center at origin)
    def normalize_shape(coords):
        # Find centroid
        xs = [lattice_to_cartesian(q, r)[0] for q, r in coords]
        ys = [lattice_to_cartesian(q, r)[1] for q, r in coords]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        # Translate so centroid is at (0,0)
        # Since lattice points are discrete, we just return the list of cartesian coords centered
        return [(lattice_to_cartesian(q, r)[0] - cx, lattice_to_cartesian(q, r)[1] - cy) for q, r in coords]

    # Generate specific shapes
    # Shape 1: 2x3 Rectangle
    # Axial coords for a 2x3 block:
    # Row 0: (0,0), (1,0), (2,0)
    # Row 1: (0,1), (1,1), (2,1)
    shape_2x3 = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)]
    if is_connected(shape_2x3):
        candidate_shapes.append(normalize_shape(shape_2x3))
        
    # Shape 2: 3x2 Rectangle (rotated version essentially)
    # Row 0: (0,0), (1,0)
    # Row 1: (0,1), (1,1)
    # Row 2: (0,2), (1,2)
    shape_3x2 = [(0,0), (1,0), (0,1), (1,1), (0,2), (1,2)]
    if is_connected(shape_3x2):
        candidate_shapes.append(normalize_shape(shape_3x2))
        
    # Shape 3: "Honeycomb" cluster around center
    # Center (0,0) + 6 neighbors + 5 more?
    # Let's just do a random search for compact connected sets
    # Start with (0,0) and grow
    def generate_random_polyhex(size):
        # Start with origin
        coords = {(0,0)}
        frontier = [(0,0)]
        while len(coords) < size:
            # Pick a random cell from frontier to expand
            if not frontier:
                return None # Disconnected or stuck
            # Shuffle frontier for randomness
            random.shuffle(frontier)
            q, r = frontier[0]
            # Pick a random neighbor not in coords
            neighbors = [
                (q+1, r), (q-1, r),
                (q, r+1), (q, r-1),
                (q+1, r-1), (q-1, r+1)
            ]
            random.shuffle(neighbors)
            added = False
            for nq, nr in neighbors:
                if (nq, nr) not in coords:
                    coords.add((nq, nr))
                    frontier.append((nq, nr))
                    added = True
                    break
            if not added:
                return None
        if is_connected(list(coords)):
            return list(coords)
        return None

    # Generate 100 random polyhexes
    for _ in range(100):
        if time.time() > deadline - 30: break
        shape = generate_random_polyhex(N)
        if shape:
            candidate_shapes.append(normalize_shape(shape))
            
    # Remove duplicates based on sorted tuple of rounded coordinates
    unique_shapes = []
    seen = set()
    for shape in candidate_shapes:
        # Round to 3 decimals for hashing
        key = tuple(sorted((round(x, 3), round(y, 3)) for x, y in shape))
        if key not in seen:
            seen.add(key)
            unique_shapes.append(shape)
            
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0
    best_outer_center = (0.0, 0.0)
    
    # For each shape, optimize the global rotation and outer hexagon parameters
    # The inner hexagons maintain their relative lattice positions but the whole cluster rotates.
    # We also allow the outer hexagon to rotate.
    # Actually, since the inner hexagons are on a lattice, their individual rotations are fixed relative to the lattice.
    # If we rotate the whole cluster by alpha, each inner hexagon rotates by alpha.
    # So the problem reduces to:
    # 1. Choose a shape (set of relative positions).
    # 2. Choose a global rotation alpha for the cluster.
    # 3. Choose an outer hexagon rotation beta.
    # 4. Minimize the side length of the outer hexagon containing the rotated cluster.
    # The outer hexagon center is fixed at (0,0) for the calculation, but we can shift the cluster?
    # No, the task allows translating the inner hexagons. So we can center the cluster optimally.
    # However, due to symmetry, centering the centroid of the cluster at (0,0) is usually optimal for regular containers.
    
    # Let's iterate over unique shapes
    for shape in unique_shapes:
        if time.time() > deadline - 10: break
        
        # Convert shape to numpy array of positions
        pos = np.array(shape)
        
        # The angles of the inner hexagons are all 0 relative to the lattice.
        # If we rotate the cluster by alpha, the angles become alpha.
        
        # We need to find alpha (cluster rotation) and beta (outer rotation) that minimize side length.
        # Since the shape has 6-fold symmetry (if it's a regular hexagon cluster) or lower,
        # we can search alpha in [0, 60) and beta in [0, 60).
        
        # To be fast, we will sample alpha and beta.
        # A finer grid search is better than Nelder-Mead here because the function is smooth and periodic.
        
        best_shape_side = float('inf')
        best_alpha = 0
        best_beta = 0
        
        # Sample 180 steps for alpha (0.33 deg resolution) and 180 for beta?
        # That's 32400 evaluations per shape. 100 shapes * 32400 = 3.2M evals.
        # Each eval is 12 hexagons * 6 vertices * 6 normals = 432 dot products. Very fast.
        # 3.2M * 432 ops ~ 1.3B ops. Might be too slow in Python.
        # Let's reduce resolution or use scipy minimize on (alpha, beta).
        
        # Let's use scipy minimize on the 2 parameters (alpha, beta).
        # Objective: side length.
        
        def objective(params):
            alpha, beta = params
            # Rotate positions
            alpha_rad = math.radians(alpha)
            cos_a = math.cos(alpha_rad)
            sin_a = math.sin(alpha_rad)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_pos = pos @ rot.T
            
            # Calculate min outer side
            # Inner hexagons are rotated by alpha
            return calc_min_outer_side(rotated_pos, np.full(N, alpha), beta)
            
        # Try multiple starting points for (alpha, beta)
        # Symmetry suggests checking 0, 15, 30, 45 degrees etc.
        starts = [(0, 0), (15, 0), (30, 0), (0, 15), (15, 15), (30, 15), (0, 30), (15, 30), (30, 30)]
        
        for start in starts:
            try:
                res = minimize(objective, start, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.fun < best_shape_side:
                    best_shape_side = res.fun
                    best_alpha, best_beta = res.x
            except Exception:
                pass
        
        if best_shape_side < best_side:
            best_side = best_shape_side
            best_alpha, best_beta = best_alpha, best_beta
            
            # Reconstruct positions
            alpha_rad = math.radians(best_alpha)
            cos_a = math.cos(alpha_rad)
            sin_a = math.sin(alpha_rad)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            final_pos = pos @ rot.T
            final_angles = np.full(N, best_alpha)
            
            best_positions = final_pos
            best_angles = final_angles
            best_outer_angle = best_beta

    # Fallback if no shape was processed
    if best_positions is None:
        # Use a simple line
        pos = np.array([(i * 1.5, 0) for i in range(N)])
        best_positions = pos
        best_angles = np.zeros(N)
        best_side = calc_min_outer_side(pos, best_angles, 0.0)
        best_outer_angle = 0.0

    # Final verification and return
    # The outer hexagon center is (0,0) in our calculation.
    # The inner hexagons are at best_positions.
    
    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], best_angles[i]])
    
    return inner, (0.0, 0.0), best_side, best_outer_angle

