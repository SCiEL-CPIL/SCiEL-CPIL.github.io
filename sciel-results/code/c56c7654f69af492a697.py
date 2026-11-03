import math
import time
import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin for 150s limit

    # Hexagonal lattice basis vectors for unit hexagons (side length 1)
    # Distance between centers of adjacent hexagons is sqrt(3)
    sqrt3 = math.sqrt(3)
    # Using axial coordinates (q, r) -> Cartesian (x, y)
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])

    # Generate all connected sets of N hexagons (polyhexes) starting from a seed
    # We use a BFS approach to grow a cluster of size N
    # To avoid duplicates, we can fix one hexagon at the origin and use a canonical form,
    # but for N=12, the number of polyhexes is manageable (a few thousand).
    # We will generate all unique polyhexes of size 12.
    
    def get_neighbors(q, r):
        return [(q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)]

    def canonical_form(cells):
        # Translate so min q is 0, then min r is 0
        min_q = min(c[0] for c in cells)
        min_r = min(c[1] for c in cells)
        translated = [(q - min_q, r - min_r) for q, r in cells]
        return tuple(sorted(translated))

    def generate_polyhexes(n_cells):
        seen = set()
        current_layer = [((0, 0),)]  # List of tuples of cells
        results = []
        
        # This simple BFS might generate duplicates or disconnected sets if not careful.
        # A better way for small N: 
        # Start with 1 cell. Add a neighbor. Check connectivity. Check canonical form.
        
        # Since N=12 is small, we can use a recursive backtracking or BFS with canonicalization.
        # However, standard polyhex enumeration is tricky. 
        # Alternative: Since we only care about the *shape* up to rotation/reflection,
        # and we want to minimize the bounding box, we can just generate all *connected* sets
        # containing the origin, and then for each, check if it's a new shape.
        
        # Actually, for packing, the specific shape matters. 
        # Let's use a simpler heuristic: Generate a set of diverse "seed" topologies
        # manually or via a restricted search, as full enumeration might be slow in Python.
        
        # Let's generate a few specific topologies known to be compact:
        # 1. 2x3 rectangle (in hex coords)
        # 2. 3x2 rectangle
        # 3. Hexagonal cluster (1+6+5)
        # 4. 1x12 line (bad, but check)
        # 5. 2x6 rectangle
        # 6. "Fat" hexagon: 1+6+5 (standard)
        # 7. 3 rows of 4 (staggered)
        # 8. 4 rows of 3 (staggered)
        
        topologies = []
        
        # Topology 1: 3x4 Rectangular Grid (in axial coords)
        # q from 0 to 3, r from 0 to 2
        # But hex grids are staggered.
        # Let's define shapes by their axial coordinates.
        
        # Shape A: 3 wide, 4 high
        cells_a = []
        for r in range(4):
            for q in range(3):
                cells_a.append((q, r))
        topologies.append(canonical_form(cells_a))
        
        # Shape B: 4 wide, 3 high
        cells_b = []
        for r in range(3):
            for q in range(4):
                cells_b.append((q, r))
        topologies.append(canonical_form(cells_b))
        
        # Shape C: Hexagonal Cluster (Radius 1 + 5 of Radius 2)
        # Center (0,0)
        # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # Ring 2: Pick 5 consecutive from Ring 2
        # Ring 2 points: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
        # Let's take the first 5 in counter-clockwise order starting from East
        ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
        ring2_all = [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)]
        # Note: The order above for ring2 is roughly CCW.
        # Let's just take 5 consecutive.
        cells_c = [(0,0)] + ring1 + ring2_all[:5]
        topologies.append(canonical_form(cells_c))
        
        # Shape D: "Pentagon" like cluster? 
        # Let's try a 2x5 block plus 2 extra
        cells_d = []
        for r in range(2):
            for q in range(5):
                cells_d.append((q, r))
        # Add 2 more to make 12? 2*5=10. Need 2 more.
        # Add (0, 2) and (1, 2)
        cells_d.append((0, 2))
        cells_d.append((1, 2))
        topologies.append(canonical_form(cells_d))

        # Shape E: 3x4 but with corners cut? 
        # Let's stick to the generated ones and maybe add a "diamond" shape
        # (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 7
        # Add (2,0), (1,1), (0,2), (-1,2), (-2,1) -> 12
        cells_e = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (1,1), (0,2), (-1,2), (-2,1)]
        topologies.append(canonical_form(cells_e))

        # Remove duplicates
        unique_topologies = list(set(topologies))
        
        # For each topology, calculate the optimal rotation and translation
        best_side = float('inf')
        best_config = None
        
        for topo in unique_topologies:
            # Convert axial to Cartesian
            points = np.array([axial_to_cartesian(q, r) for q, r in topo])
            
            # Center the points
            centroid = np.mean(points, axis=0)
            points_centered = points - centroid
            
            # The outer hexagon is also a regular hexagon.
            # We can rotate the cluster by an angle theta.
            # The side length of the enclosing hexagon depends on theta.
            # We need to find theta that minimizes the side length.
            
            def enclosing_side_length(theta):
                if time.time() > deadline - 5:
                    return float('inf')
                
                # Rotate points
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                rotated_points = points_centered @ rot_matrix.T
                
                # The outer hexagon has side length S.
                # Its vertices are at distance S from center (circumradius = S).
                # Wait, for a regular hexagon of side S:
                # Apothem = S * sqrt(3) / 2
                # Circumradius = S
                # The condition for a point p to be inside a regular hexagon centered at origin
                # with "flat-top" orientation (vertex at 0, 60, ...):
                # The supporting lines are at distance apothem from center.
                # The normals to the edges are at angles 30, 90, 150, 210, 270, 330 degrees.
                # Let's define the outer hexagon orientation. 
                # We can fix the outer hexagon orientation to 0 and rotate the inner cluster,
                # or fix the inner cluster and rotate the outer. It's equivalent.
                # Let's fix the outer hexagon to have vertices at angles 0, 60, ... (pointy-top? No, vertex at 0 is pointy-top).
                # Standard "pointy-top" hexagon: vertices at 0, 60, 120...
                # Edges are vertical? No.
                # Let's use the standard definition:
                # Vertices at (S*cos(k*60), S*sin(k*60)).
                # The 6 edges have outward normals at angles 30, 90, 150, 210, 270, 330.
                # For a point to be inside, its projection onto each normal must be <= apothem.
                # apothem = S * sqrt(3) / 2.
                
                normals = np.array([[math.cos(math.radians(30 + k*60)), math.sin(math.radians(30 + k*60))] for k in range(6)])
                
                # Project rotated points onto normals
                projs = rotated_points @ normals.T
                max_proj = np.max(projs)
                
                # We need max_proj <= apothem = S * sqrt(3) / 2
                # So S >= max_proj * 2 / sqrt(3)
                # Also, the hexagon must contain the points in all directions.
                # Since the hexagon is convex and symmetric, the maximum projection on the 6 normals determines the size.
                
                return max_proj * 2.0 / sqrt3
            
            # Optimize theta
            # The function is periodic with period 60 degrees (pi/3).
            # We can search in [0, pi/3].
            res = minimize_scalar(enclosing_side_length, bounds=(0, math.pi/3), method='bounded', options={'xatol': 1e-6})
            
            if res.success:
                side = res.fun
                if side < best_side:
                    best_side = side
                    # Store the best configuration
                    # We need to return the positions and angles of the inner hexagons.
                    # Inner hexagons are aligned with the lattice (angle 0 relative to each other).
                    # But the whole cluster is rotated by res.x.
                    # The inner hexagons themselves can be considered to have angle 0, 
                    # and their positions are rotated.
                    # Or, we can say the inner hexagons have angle = res.x, and positions are unrotated?
                    # The interface asks for (x, y, angle_degrees).
                    # If we rotate the cluster by theta, each hexagon's center is rotated, 
                    # and each hexagon's orientation is rotated by theta.
                    
                    # Let's compute the final positions and angles
                    theta_opt = res.x
                    cos_t = math.cos(theta_opt)
                    sin_t = math.sin(theta_opt)
                    rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                    
                    final_points = points_centered @ rot_matrix.T
                    final_angles = np.full(N, math.degrees(theta_opt))
                    
                    best_config = (final_points, final_angles)
        
        # Fallback if no topology found or failed
        if best_config is None:
            # Use a simple 3x4 grid as fallback
            points = []
            for r in range(3):
                for q in range(4):
                    points.append(axial_to_cartesian(q, r))
            points = np.array(points)
            centroid = np.mean(points, axis=0)
            points = points - centroid
            best_config = (points, np.zeros(N))
            best_side = enclosing_side_length(0.0) # Just for reporting, not used
        
        final_points, final_angles = best_config
        
        # Construct the return values
        inner = []
        for i in range(N):
            inner.append([float(final_points[i, 0]), float(final_points[i, 1]), float(final_angles[i])])
        
        # Outer hexagon: Center (0,0), Side best_side, Angle 0
        # The evaluator expects (inner, center, side, angle)
        # center is (0,0)
        # side is best_side
        # angle is 0 (we assumed outer is fixed at 0)
        
        return inner, (0.0, 0.0), float(best_side), 0.0

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Side: {res[2]:.4f}")
