# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses Simulated Annealing to minimize the outer hexagon size.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    sqrt3 = math.sqrt(3)
    N = 12
    
    # Helper: Get vertices of a hexagon
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + side * math.cos(a)
            y = cy + side * math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    # Helper: Check if two convex polygons are separated using SAT
    def are_polygons_separated(poly1, poly2):
        # poly1, poly2 are (6, 2) arrays of vertices
        # Get edges and normals for both polygons
        for poly in [poly1, poly2]:
            # Edges
            edges = np.roll(poly, -1, axis=0) - poly
            # Normals (perpendicular to edges)
            # For a convex polygon defined in CCW order, the outward normal is (dy, -dx) or (-dy, dx)?
            # Let's just use the normal direction and check separation.
            # Normal n = (edge[1], -edge[0])
            normals = np.column_stack((edges[:, 1], -edges[:, 0]))
            # Normalize normals (not strictly necessary for SAT if we project, but good for consistency)
            # We just need the direction.
            for n in normals:
                # Project all vertices of poly1 onto n
                proj1 = poly1 @ n
                # Project all vertices of poly2 onto n
                proj2 = poly2 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # Check for separation
                # If max1 < min2 or max2 < min1, they are separated
                if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                    return True
        return False

    # Helper: Calculate minimum side length s for a set of inner hexes to fit in outer hex (centered at origin, angle 0)
    # Outer hex with side s, angle 0 has vertices at distance s from origin.
    # The "radius" (center to vertex) is s.
    # The "apothem" (center to edge midpoint) is s * sqrt(3)/2.
    # Normals of outer hex edges are at angles 30, 90, 150, 210, 270, 330 degrees.
    normal_angles = [math.radians(30 + i*60) for i in range(6)]
    normals = np.array([[math.cos(a), math.sin(a)] for a in normal_angles])
    
    # Vertex offsets for unit hexagon centered at origin, angle 0
    v_offsets = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])

    def calc_min_s(centers):
        # centers: (N, 2) array
        max_proj = 0.0
        for c in centers:
            verts = c + v_offsets
            # Project onto normals
            projs = verts @ normals.T
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        return max_proj * 2.0 / sqrt3

    def is_valid(centers, s):
        # Check containment and non-overlap
        # 1. Containment
        apothem = s * sqrt3 / 2
        for c in centers:
            verts = c + v_offsets
            projs = verts @ normals.T
            # max projection along each normal must be <= apothem
            if np.any(np.max(projs, axis=0) > apothem + 1e-6):
                return False
        
        # 2. Non-overlap
        # Pre-compute vertices for all hexes
        all_verts = []
        for c in centers:
            all_verts.append(c + v_offsets)
            
        for i in range(N):
            for j in range(i+1, N):
                if not are_polygons_separated(all_verts[i], all_verts[j]):
                    return False
        return True

    best_s = float('inf')
    best_centers = None
    
    # Simulated Annealing
    # Initial guess: Lattice
    centers = []
    for i in range(3):
        for j in range(4):
            x = j * 2.5
            y = i * sqrt3
            if i % 2 == 1:
                x += 1.25
            centers.append([x, y])
    centers = np.array(centers)
    centers -= np.mean(centers, axis=0)
    
    current_s = calc_min_s(centers)
    if is_valid(centers, current_s):
        best_s = current_s
        best_centers = centers.copy()
    
    T = 10.0 # Initial temperature
    alpha = 0.9999 # Cooling rate
    min_T = 0.001
    
    while T > min_T:
        if time.time() > deadline - 5:
            break
        
        # Generate a new candidate by perturbing one or two centers
        num_perturb = 1
        if random.random() < 0.2:
            num_perturb = 2
            
        new_centers = centers.copy()
        for _ in range(num_perturb):
            idx = random.randint(0, N-1)
            # Perturbation magnitude
            delta = random.gauss(0, 0.5)
            dx = random.gauss(0, 0.5)
            dy = random.gauss(0, 0.5)
            new_centers[idx, 0] += dx
            new_centers[idx, 1] += dy
            
        # Recenter
        new_centers -= np.mean(new_centers, axis=0)
        
        new_s = calc_min_s(new_centers)
        
        # Check validity of new state
        valid = is_valid(new_centers, new_s)
        
        if valid:
            # If valid, we accept it if it's better, or with some probability if worse?
            # Actually, for packing, we want to minimize s.
            # If new_s < current_s, accept.
            # If new_s > current_s, accept with probability exp(-(new_s - current_s) / T)
            # But we also need to maintain validity.
            # If the move is valid but increases s, we might accept it to escape local minima in s-space?
            # However, if we increase s, we are moving away from the goal.
            # But maybe a larger s allows a better arrangement for a subsequent decrease?
            # Let's try accepting valid moves that increase s with a small probability.
            if new_s < current_s:
                centers = new_centers
                current_s = new_s
                if new_s < best_s:
                    best_s = new_s
                    best_centers = centers.copy()
            else:
                # Accept worse (larger s) with probability
                p = math.exp(-(new_s - current_s) / T)
                if random.random() < p:
                    centers = new_centers
                    current_s = new_s
        else:
            # Invalid move: reject unless we are very hot?
            # Usually we just reject invalid moves in hard-constraint problems.
            # But maybe we should allow "repair" steps?
            # For simplicity, reject invalid moves.
            pass
            
        T *= alpha

    # Fallback if no valid solution found (should not happen with lattice start)
    if best_centers is None:
        centers = []
        for i in range(3):
            for j in range(4):
                x = j * 2.5
                y = i * sqrt3
                if i % 2 == 1:
                    x += 1.25
                centers.append([x, y])
        best_centers = np.array(centers)
        best_centers -= np.mean(best_centers, axis=0)
        best_s = calc_min_s(best_centers)
        
    # Construct return values
    inner_ret = np.zeros((12, 3))
    for i in range(12):
        inner_ret[i, 0] = best_centers[i, 0]
        inner_ret[i, 1] = best_centers[i, 1]
        inner_ret[i, 2] = 0.0
        
    outer_ret = np.array([0.0, 0.0, 0.0])
    
    return inner_ret, outer_ret, best_s

# EVOLVE-BLOCK-END
