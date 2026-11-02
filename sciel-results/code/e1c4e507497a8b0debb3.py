import numpy as np
import time
import random
import math

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Precompute hexagon vertices for unit hexagon at origin, angle 0
    # Vertices at angles 0, 60, 120, 180, 240, 300
    hex_verts_local = np.array([
        [math.cos(math.radians(a)), math.sin(math.radians(a))]
        for a in [0, 60, 120, 180, 240, 300]
    ])

    def get_world_vertices(axial_coords, scale=1.0):
        """
        Convert axial coordinates (q, r) to world coordinates (x, y)
        and then to world vertices for the hexagons.
        Axial to pixel:
        x = SQRT3 * (q + r/2)
        y = 1.5 * r
        """
        q = axial_coords[:, 0]
        r = axial_coords[:, 1]
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        centers = np.column_stack((x, y))
        
        # Return array of shape (N, 6, 2)
        # vertices = centers[:, np.newaxis, :] + hex_verts_local[np.newaxis, :, :]
        return centers[:, np.newaxis, :] + hex_verts_local[np.newaxis, :, :]

    def check_separation(verts1, verts2):
        """
        Check if two convex hexagons are separated.
        Uses Separating Axis Theorem (SAT).
        Returns True if separated (no overlap), False if overlapping.
        """
        # Precomputed normals for unit hexagon edges
        # Edge 0: (0,0) to (0.5, SQRT3/2). Normal: (SQRT3/2, -0.5) or (-SQRT3/2, 0.5)
        # Let's just compute normals on the fly for robustness, or use fixed 6 axes
        # Since all hexagons are aligned (angle 0), the separating axes are the same for all.
        # The 6 axes are perpendicular to the 6 edges.
        # Edge vectors:
        # v0 = (0.5, SQRT3/2) - (1, 0) = (-0.5, SQRT3/2) -> Normal (SQRT3/2, 0.5)
        # v1 = (0, 1) - (0.5, SQRT3/2) = (-0.5, 1-SQRT3/2)... wait, vertices are at 60 deg steps.
        
        # Let's define the 6 unit normals for the edges of a unit hexagon.
        # The edges are between vertices.
        # Vertices:
        # 0: (1, 0)
        # 1: (0.5, SQRT3/2)
        # 2: (-0.5, SQRT3/2)
        # 3: (-1, 0)
        # 4: (-0.5, -SQRT3/2)
        # 5: (0.5, -SQRT3/2)
        
        # Edge 0: 0->1. Midpoint (0.75, SQRT3/4). Normal points out.
        # Vector 1->0 is (0.5, -SQRT3/2). Normal is (SQRT3/2, 0.5).
        # Let's verify: Dot((SQRT3/2, 0.5), (0.5, -SQRT3/2)) = SQRT3/4 - SQRT3/4 = 0. Correct.
        
        normals = np.array([
            [SQRT3/2, 0.5],
            [0, 1],
            [-SQRT3/2, 0.5],
            [-SQRT3/2, -0.5],
            [0, -1],
            [SQRT3/2, -0.5]
        ])
        
        # Project both polygons onto these 6 axes
        # verts1, verts2 are (6, 2)
        # Projections: (6, 6) where row is axis, col is vertex?
        # proj1 = verts1 @ normals.T -> (6, 6)
        # min1 = min along axis 1
        # max1 = max along axis 1
        
        proj1 = verts1 @ normals.T
        proj2 = verts2 @ normals.T
        
        min1 = np.min(proj1, axis=1)
        max1 = np.max(proj1, axis=1)
        min2 = np.min(proj2, axis=1)
        max2 = np.max(proj2, axis=1)
        
        # Check for separation on any axis
        # Separated if max1 < min2 OR max2 < min1
        for i in range(6):
            if max1[i] < min2[i] or max2[i] < min1[i]:
                return True
        return False

    def is_valid(verts):
        """
        Check if all hexagons in the configuration are disjoint.
        verts: (N, 6, 2)
        """
        for i in range(N):
            for j in range(i + 1, N):
                if not check_separation(verts[i], verts[j]):
                    return False
        return True

    def calculate_side(verts):
        """
        Calculate the side length of the smallest regular hexagon (angle 0)
        centered at origin that contains all vertices.
        """
        # Flatten vertices
        flat = verts.reshape(-1, 2)
        x = flat[:, 0]
        y = flat[:, 1]
        
        # The support function for a regular hexagon of side S centered at origin.
        # The distance from center to side (apothem) is S * SQRT3 / 2.
        # The width in the normal direction is S * SQRT3.
        # The projection of a point (x,y) onto the normal n_k must be <= S * SQRT3 / 2.
        # S >= (2 / SQRT3) * max_k (p . n_k)
        
        # Normals for the outer hexagon containment (same as inner hexagon edges normals)
        # n1 = (SQRT3/2, 1/2)
        # n2 = (0, 1)
        # n3 = (-SQRT3/2, 1/2)
        # n4 = (-SQRT3/2, -1/2)
        # n5 = (0, -1)
        # n6 = (SQRT3/2, -1/2)
        
        # We can compute the max projection for each direction
        # proj1 = x * SQRT3/2 + y * 0.5
        # proj2 = y
        # proj3 = -x * SQRT3/2 + y * 0.5
        # proj4 = -x * SQRT3/2 - y * 0.5
        # proj5 = -y
        # proj6 = x * SQRT3/2 - y * 0.5
        
        # Actually, we can just use the max of all absolute projections?
        # No, the hexagon is not symmetric in all directions the same way a circle is.
        # But it IS symmetric under 60 degree rotation.
        # The condition is: max( p.n1, p.n2, ..., p.n6 ) <= S * SQRT3 / 2
        # And min( p.n1, ..., p.n6 ) >= -S * SQRT3 / 2
        # So S >= (2/SQRT3) * max( max(p.n_k), -min(p.n_k) )
        # Which is (2/SQRT3) * max( max(p.n_k), max(-p.n_k) ) = (2/SQRT3) * max( max(p.n_k), max(p.n_k') )
        # Where n_k' are the opposite normals.
        # Since the set of 6 normals includes opposite pairs (n1, n4), (n2, n5), (n3, n6),
        # max(p.n_k) over all k covers the positive side.
        # But we need the max over all 6 directions of the projection.
        
        normals = np.array([
            [SQRT3/2, 0.5],
            [0, 1],
            [-SQRT3/2, 0.5],
            [-SQRT3/2, -0.5],
            [0, -1],
            [SQRT3/2, -0.5]
        ])
        
        projs = flat @ normals.T
        max_proj = np.max(projs, axis=0)
        # The required apothem is max(max_proj)
        # Side = 2 * apothem / SQRT3
        return 2.0 * np.max(max_proj) / SQRT3

    # Initial Configuration: Honeycomb Lattice
    # Generate a cluster of 12 hexagons in axial coordinates
    # Center (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: 12 hexagons. We need 5 more to make 12 total? No, 1+6=7. Need 5 from Ring 2.
    # Let's pick a compact set.
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (1,1), (0,2), (-1,2), (-2,1)
    
    init_axial = np.array([
        [0, 0],
        [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
        [2, 0], [1, 1], [0, 2], [-1, 2], [-2, 1]
    ], dtype=float)
    
    current_axial = init_axial.copy()
    current_verts = get_world_vertices(current_axial)
    current_side = calculate_side(current_verts)
    
    best_axial = current_axial.copy()
    best_side = current_side
    best_verts = current_verts.copy()
    
    # Simulated Annealing Parameters
    T = 1.0  # Initial Temperature
    T_min = 0.01
    cooling_rate = 0.999
    max_iter = 50000
    move_magnitude = 0.5 # Step size in axial coordinates
    
    # Precompute neighbors for speed? No, N=12 is small.
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Generate a random move
        # Pick a random hexagon and a random direction
        idx = random.randint(0, N - 1)
        # Random vector in axial space
        # To ensure we move to a valid "hex-like" step, we can use random float steps
        # Or use the 6 axial directions + stay?
        # Using continuous steps allows for better packing potentially, but axial lattice is natural.
        # Let's use small random perturbations in axial coordinates.
        dq = random.uniform(-move_magnitude, move_magnitude)
        dr = random.uniform(-move_magnitude, move_magnitude)
        
        new_axial = current_axial.copy()
        new_axial[idx, 0] += dq
        new_axial[idx, 1] += dr
        
        new_verts = get_world_vertices(new_axial)
        
        # Check validity
        if not is_valid(new_verts):
            continue # Reject invalid moves (strict constraint handling)
            
        new_side = calculate_side(new_verts)
        delta = new_side - current_side
        
        # Metropolis criterion
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_axial = new_axial
            current_verts = new_verts
            current_side = new_side
            
            if current_side < best_side:
                best_side = current_side
                best_axial = current_axial.copy()
                best_verts = current_verts.copy()
                
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Restart with higher T if stuck? Or just stop?
            # Let's do a few restarts if time permits
            if i > max_iter * 0.8:
                break

    # Final verification
    if not is_valid(best_verts):
        # Fallback to initial
        best_verts = get_world_vertices(init_axial)
        best_side = calculate_side(best_verts)

    # Extract positions
    centers = get_world_vertices(best_axial).mean(axis=1) # Average of vertices is center?
    # Wait, get_world_vertices returns vertices. Center is mean of vertices.
    # Or just calculate from axial again.
    q = best_axial[:, 0]
    r = best_axial[:, 1]
    x = SQRT3 * (q + r / 2.0)
    y = 1.5 * r
    
    inner = []
    for i in range(N):
        inner.append([float(x[i]), float(y[i]), 0.0])
        
    return inner, (0.0, 0.0), float(best_side), 0.0

# EVOLVE-BLOCK-END
