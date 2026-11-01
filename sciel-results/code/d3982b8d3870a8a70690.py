"""Combined program for packing 12 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon."""
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px, py) is inside hexagon centered at (cx, cy)."""
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a

        apothem = side * np.sqrt(3) / 2
        # Normals for a flat-top hexagon (angle 0) are at 30, 90, 150, 210, 270, 330
        # If angle_deg is 0, the vertices are at 0, 60, ...
        # The normals are perpendicular to the edges.
        # For a hexagon with vertices at angles theta, the edges are between theta and theta+60.
        # The normal to the edge between 0 and 60 is at 30.
        normal_angles = angle_rad + np.radians(np.arange(30, 360, 60))
        
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def polygons_disjoint_sat(v1, v2):
        """Check if two convex polygons are disjoint using SAT."""
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)

        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)

        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)

            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        return False

    def compute_minimal_outer(inner_positions, outer_angle_deg):
        """
        Compute the minimal side length of an outer hexagon centered at (0,0) 
        with given angle that contains all inner hexagons.
        """
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        
        if not all_verts:
            return 0.0
            
        all_verts = np.vstack(all_verts)
        
        # The outer hexagon is centered at (0,0).
        # Its apothem determines the side length: side = 2 * apothem / sqrt(3)
        # We need the maximum projection of any vertex onto the 6 normal directions.
        angle_rad = np.radians(outer_angle_deg)
        # Normals for the outer hexagon
        normal_angles = angle_rad + np.radians(np.arange(30, 360, 60))
        
        max_apothem_needed = 0
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            # Projection of all vertices onto this normal
            dists = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            # Since the hexagon is centered at 0, the "distance" from center to the face
            # is the maximum absolute projection? No.
            # The hexagon is defined by |x . n_i| <= apothem for all 6 normals?
            # Actually, a regular hexagon centered at origin with apothem 'a' is the intersection
            # of 6 half-planes: x . n_i <= a for all i.
            # So we need max(x . n_i) <= a for all i.
            max_dist = np.max(dists)
            if max_dist > max_apothem_needed:
                max_apothem_needed = max_dist
                
        # side = 2 * apothem / sqrt(3)
        return 2.0 * max_apothem_needed / np.sqrt(3)

    def make_params(inner):
        """Convert inner positions to flat parameter array."""
        params = []
        for pos in inner:
            params.extend([pos[0], pos[1], pos[2]])
        return np.array(params, dtype=float)

    def parse_params(params):
        """Convert flat parameter array back to structured form."""
        inner = []
        for i in range(N):
            inner.append((params[3 * i], params[3 * i + 1], params[3 * i + 2]))
        return inner

    def objective(params):
        """
        Objective function: minimize outer side length.
        Params: 3*N values (x, y, angle for each inner hex).
        Outer hexagon is fixed at center (0,0) and angle 0.0 (or 30.0).
        """
        n_inner = N
        inner_pos = []
        for i in range(n_inner):
            x = params[3 * i]
            y = params[3 * i + 1]
            a = params[3 * i + 2]
            inner_pos.append((x, y, a))
        
        # Fixed outer orientation
        outer_angle = 0.0 
        
        # Calculate minimal side length required
        outer_side = compute_minimal_outer(inner_pos, outer_angle)
        
        # Check disjointness
        disjoint_ok = True
        overlap_count = 0
        for i in range(n_inner):
            for j in range(i + 1, n_inner):
                ci = (inner_pos[i][0], inner_pos[i][1])
                cj = (inner_pos[j][0], inner_pos[j][1])
                ai = inner_pos[i][2]
                aj = inner_pos[j][2]
                v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                if not polygons_disjoint_sat(v1, v2):
                    disjoint_ok = False
                    overlap_count += 1
                    break
            if not disjoint_ok:
                break

        if disjoint_ok:
            return outer_side
        
        # Penalty for overlaps
        return outer_side + overlap_count * 5.0

    def make_hex_cluster():
        """
        Generate a high-quality initial configuration:
        1 center + 6 first ring + 5 second ring.
        Selects the 5 second ring positions that minimize the bounding hexagon.
        """
        positions = []
        positions.append((0.0, 0.0, 0.0))
        
        # First ring: 6 hexagons at distance 2.0 (tangent)
        for k in range(6):
            angle = np.radians(60 * k)
            x = 2.0 * np.cos(angle)
            y = 2.0 * np.sin(angle)
            positions.append((x, y, 0.0))
        
        sqrt3 = np.sqrt(3)
        # Second ring candidates: distance ~3.0 to 4.0
        # In a hexagonal lattice with spacing 2.0:
        # (2, 0) is ring 1.
        # (4, 0) is ring 2.
        # (3, sqrt(3)) is ring 2.
        # etc.
        second_ring_candidates = [
            (4.0, 0.0),
            (3.0, sqrt3),
            (1.0, 2*sqrt3),
            (-1.0, 2*sqrt3),
            (-3.0, sqrt3),
            (-4.0, 0.0),
            (-3.0, -sqrt3),
            (-1.0, -2*sqrt3),
            (1.0, -2*sqrt3),
            (3.0, -sqrt3),
        ]
        
        best_combo = None
        best_side = float('inf')
        
        # Try all combinations of 5 from 10
        for combo in combinations(range(len(second_ring_candidates)), 5):
            test_positions = positions.copy()
            for idx in combo:
                x, y = second_ring_candidates[idx]
                test_positions.append((x, y, 0.0))
            
            side = compute_minimal_outer(test_positions, 0.0)
            if side < best_side:
                best_side = side
                best_combo = combo
        
        for idx in best_combo:
            x, y = second_ring_candidates[idx]
            positions.append((x, y, 0.0))
        
        return positions, best_side

    def get_initial_configurations():
        configs = []
        
        # Config 1: Hexagonal cluster (Best guess)
        hex_pos, _ = make_hex_cluster()
        configs.append(hex_pos)
        
        # Config 2: 4x3 Grid (Staggered)
        pos_grid = []
        for row in range(3):
            for col in range(4):
                x = (col - 1.5) * 2.0
                y = (row - 1) * 1.732 # sqrt(3)
                if row % 2 == 1:
                    x += 1.0
                pos_grid.append((x, y, 0.0))
        configs.append(pos_grid)
        
        # Config 3: 3x4 Grid (Staggered other way)
        pos_grid2 = []
        for row in range(4):
            for col in range(3):
                x = (col - 1.0) * 2.0
                y = (row - 1.5) * 1.732
                if row % 2 == 1:
                    x += 1.0
                pos_grid2.append((x, y, 0.0))
        configs.append(pos_grid2)

        # Config 4: Linear rows
        pos_linear = []
        for i in range(4):
             for j in range(3):
                 x = (i - 1.5) * 2.0
                 y = (j - 1.0) * 2.0
                 pos_linear.append((x, y, 0.0))
        configs.append(pos_linear)

        return configs

    best_params = None
    best_score = float('inf')

    configs = get_initial_configurations()

    for config_idx, config in enumerate(configs):
        if time.time() > deadline - 10:
            break

        params = make_params(config)
        
        # Run optimization
        try:
            # Use Nelder-Mead for local refinement
            # The objective is non-smooth due to SAT and max operations, but NM handles it well
            res = minimize(objective, params, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 0.001, 'fatol': 0.001, 'adaptive': True})
            
            # Validate the result
            inner_res = parse_params(res.x)
            # Check if it's actually valid (disjoint)
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    v1 = hexagon_vertices(inner_res[i][0], inner_res[i][1], 1.0, inner_res[i][2])
                    v2 = hexagon_vertices(inner_res[j][0], inner_res[j][1], 1.0, inner_res[j][2])
                    if not polygons_disjoint_sat(v1, v2):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                score = compute_minimal_outer(inner_res, 0.0)
                if score < best_score:
                    best_score = score
                    best_params = res.x.copy()
            else:
                # If invalid, don't update best, but we might still want to see if it's close
                # For now, only accept valid solutions
                pass
                
        except Exception:
            pass

    # If no valid result found from optimization, fallback to the best initial config
    if best_params is None:
        # Use the hex cluster which is likely valid or close
        config = configs[0]
        inner = config
        # Ensure it's disjoint by checking
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                v1 = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
                v2 = hexagon_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                if not polygons_disjoint_sat(v1, v2):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            best_score = compute_minimal_outer(inner, 0.0)
            best_params = make_params(inner)
        else:
            # Emergency fallback: large side
            # Just return a valid grid, even if side is large
            # The grid configs are usually disjoint
            for c in configs:
                 inner_c = c
                 valid_c = True
                 for i in range(N):
                     for j in range(i + 1, N):
                         v1 = hexagon_vertices(inner_c[i][0], inner_c[i][1], 1.0, inner_c[i][2])
                         v2 = hexagon_vertices(inner_c[j][0], inner_c[j][1], 1.0, inner_c[j][2])
                         if not polygons_disjoint_sat(v1, v2):
                             valid_c = False
                             break
                     if not valid_c:
                         break
                 if valid_c:
                     best_score = compute_minimal_outer(inner_c, 0.0)
                     best_params = make_params(inner_c)
                     break
                     
    # Final construction of the return values
    inner = parse_params(best_params)
    outer_center = (0.0, 0.0)
    outer_side = compute_minimal_outer(inner, 0.0)
    outer_angle = 0.0

    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
