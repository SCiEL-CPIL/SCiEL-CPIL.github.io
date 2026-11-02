import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def required_outer_side(positions, outer_center, outer_angle_deg):
        """
        Compute the exact side length of the smallest regular hexagon 
        (centered at outer_center, rotated by outer_angle_deg) 
        that contains all unit hexagons at given positions.
        """
        # Vertices of inner hexagons (side 1, angle 0)
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
        
        # Translate vertices relative to outer center
        verts_rel = verts - np.array(outer_center)
        
        # Rotate vertices to align with outer hexagon axes
        angle_rad = np.radians(outer_angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        x = verts_rel[:, :, 0]
        y = verts_rel[:, :, 1]
        
        # Rotation
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        
        sqrt3 = np.sqrt(3.0)
        
        # Support function projections for a regular hexagon with vertices at 0, 60, ...
        # The normals to the edges are at 30, 90, 150, 210, 270, 330 degrees.
        # For a hexagon of side S, the distance from center to edge (apothem) is S * sqrt(3) / 2.
        # The projection of a point (x, y) onto the normal at 30 deg is x * cos(30) + y * sin(30) = x * sqrt(3)/2 + y * 0.5
        # We need max_projection <= S * sqrt(3) / 2
        # So S >= 2/sqrt(3) * max_projection
        
        # Normal 1: 30 deg -> (sqrt(3)/2, 1/2)
        L1 = x_rot * (sqrt3 / 2) + y_rot * 0.5
        # Normal 2: 90 deg -> (0, 1)
        L2 = y_rot
        # Normal 3: 150 deg -> (-sqrt(3)/2, 1/2)
        L3 = -x_rot * (sqrt3 / 2) + y_rot * 0.5
        # Normal 4: 210 deg -> (-sqrt(3)/2, -1/2)
        L4 = -x_rot * (sqrt3 / 2) - y_rot * 0.5
        # Normal 5: 270 deg -> (0, -1)
        L5 = -y_rot
        # Normal 6: 330 deg -> (sqrt(3)/2, -1/2)
        L6 = x_rot * (sqrt3 / 2) - y_rot * 0.5
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=0)
        max_proj = np.max(all_L)
        
        return (2.0 / sqrt3) * max_proj

    def generate_lattice_configs():
        """
        Generate candidate sets of 12 hexagons on a hexagonal lattice.
        Hexagons of side 1 placed at integer axial coordinates (q, r) do not overlap.
        We search for clusters of 12 cells that minimize the bounding hexagon.
        """
        configs = []
        
        # Define a search range for the "center" of the cluster
        # A cluster of 12 hexagons roughly fits in a radius of 3-4 hexagons
        max_radius = 4
        
        # Generate all hexagons within radius
        hexes = []
        for q in range(-max_radius, max_radius + 1):
            for r in range(-max_radius, max_radius + 1):
                # Axial to Cartesian (pointy top? or flat top? 
                # Standard axial: x = sqrt(3)*(q + r/2), y = 3/2*r for pointy top vertices?
                # Let's stick to the coordinate system used in the original seed for consistency with vertices.
                # Original seed: x = sqrt(3) * (q + r / 2.0), y = 1.5 * r
                x = np.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                hexes.append((q, r, x, y))
        
        # We need to find 12 hexagons. 
        # Instead of combinatorial explosion, we can use a "sliding window" or "center of mass" approach.
        # Or, simply enumerate all subsets of size 12 from a smaller set? 
        # Number of hexes in radius 3 is 1 + 6*1 + 6*2 + 6*3 = 1 + 6 + 12 + 18 = 37.
        # C(37, 12) is huge.
        
        # Alternative: Enumerate "polyhexes" of size 12? Also complex.
        
        # Simpler heuristic: 
        # 1. Take the "core" 7 hexagons (radius 1).
        # 2. Add 5 more from the next ring (radius 2).
        # The ring of radius 2 has 12 hexagons. C(12, 5) = 792. This is very manageable.
        
        # Core (radius 1)
        core_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
        ]
        core_pos = np.array([(np.sqrt(3) * (q + r / 2.0), 1.5 * r) for q, r in core_coords])
        
        # Ring 2 candidates
        ring2_coords = [
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
            (-1, -1), (0, -2), (1, -2), (2, -1), (2, -2), (1, -3) # Wait, need correct ring 2
        ]
        # Correct ring 2 neighbors of the core (distance 2 from center in hex grid)
        # Actually, just generate all hexes at distance 2 from origin.
        ring2 = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                # Hex distance
                dist = (abs(q) + abs(r) + abs(q+r)) // 2
                if dist == 2:
                    x = np.sqrt(3) * (q + r / 2.0)
                    y = 1.5 * r
                    ring2.append((x, y))
        
        # There are 12 hexes in ring 2.
        # We need to pick 5 from these 12 to add to the core of 7.
        # Total 12.
        
        candidates = list(itertools.combinations(range(len(ring2)), 5))
        
        for combo in candidates:
            selected_ring2 = [ring2[i] for i in combo]
            all_pos = np.vstack([core_pos, np.array(selected_ring2)])
            configs.append(all_pos)
            
        # Also consider other topologies?
        # E.g. 2 rows of 6? 
        # Or a 3x4 rectangle?
        # Let's add a few manual "rectangular" or "linear" lattice configs to be safe.
        
        # 3x4 grid (approx)
        # Axial coords for a 3x4 block
        # q: 0..2, r: 0..3 ? No, hex grids are skewed.
        # Let's just add a few random valid lattice clusters found by a simple local search on the lattice.
        
        return configs

    def solve():
        best_side = float('inf')
        best_config = None
        best_outer_center = (0,0)
        best_outer_angle = 0.0
        
        configs = generate_lattice_configs()
        
        # If generation failed or is empty, fallback
        if not configs:
            # Fallback to simple linear
            pos = np.array([[i * 3.0, 0.0] for i in range(N)])
            return pos, required_outer_side(pos, (0,0), 0.0), (0,0), 0.0

        # For each lattice config, find the optimal outer hexagon
        # The outer hexagon can be rotated and translated.
        # Due to symmetry, we only need to check outer angles 0 to 60 degrees.
        # And the center should be near the centroid of the inner hexagons.
        
        for pos in configs:
            if time.time() > deadline:
                break
                
            # Centroid of inner hexagons
            centroid = np.mean(pos, axis=0)
            
            # Optimize outer center and angle
            # We can do a fine grid search over angle (0 to 60, step 1 deg) 
            # and center (centroid +/- small range, step 0.1)
            
            # Actually, for a fixed set of points, the minimum enclosing regular hexagon 
            # is determined by the points on the boundary.
            # The optimal center is likely the centroid or slightly shifted.
            # The optimal angle aligns the "widest" projection with the hexagon axes.
            
            # Let's use scipy minimize for each config? 
            # 792 configs * minimize might be slow.
            # Let's try a coarse grid first.
            
            best_local_side = float('inf')
            best_local_center = centroid
            best_local_angle = 0.0
            
            # Angle search: 0 to 60, step 2 deg (30 steps)
            for angle_deg in np.arange(0, 60, 2.0):
                # Center search: centroid +/- 1.0, step 0.2 (11x11 = 121 steps)
                # This is 30 * 121 = 3630 evaluations per config.
                # 792 configs * 3630 = 2.8M evaluations.
                # Each eval is vectorized numpy on 12*6 points. Should be fast enough in < 145s?
                # Let's check speed.
                
                for cx_offset in np.arange(-1.0, 1.1, 0.2):
                    for cy_offset in np.arange(-1.0, 1.1, 0.2):
                        oc = (centroid[0] + cx_offset, centroid[1] + cy_offset)
                        side = required_outer_side(pos, oc, angle_deg)
                        if side < best_local_side:
                            best_local_side = side
                            best_local_center = oc
                            best_local_angle = angle_deg
            
            if best_local_side < best_side:
                best_side = best_local_side
                best_config = pos
                best_outer_center = best_local_center
                best_outer_angle = best_local_angle
        
        # Refine the best solution with a local optimization
        if best_config is not None:
            # Define objective for refinement
            def refine_obj(params):
                # params: [cx, cy, angle_deg]
                cx, cy, angle = params
                return required_outer_side(best_config, (cx, cy), angle)
            
            # Initial guess
            x0 = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
            
            try:
                res = minimize(refine_obj, x0, method='Nelder-Mead', 
                               options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 200})
                if res.success or res.fun < best_side:
                    cx, cy, angle = res.x
                    new_side = res.fun
                    if new_side < best_side:
                        best_side = new_side
                        best_outer_center = (cx, cy)
                        best_outer_angle = angle
            except Exception:
                pass
        
        return best_config, best_side, best_outer_center, best_outer_angle

    # Run the solver
    positions, side, outer_center, outer_angle = solve()
    
    # Fallback if something went wrong
    if positions is None:
        positions = np.array([[i * 3.0, 0.0] for i in range(N)])
        side = required_outer_side(positions, (0,0), 0.0)
        outer_center = (0.0, 0.0)
        outer_angle = 0.0

    # Format output
    # Inner hexagons are at 'positions', angle 0
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), 0.0])
        
    # Outer hexagon
    # The evaluator expects: inner, center, side, angle
    # center is (x, y) of outer hex
    # side is the side length
    # angle is rotation of outer hex
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(side), float(outer_angle)

# EVOLVE-BLOCK-END
