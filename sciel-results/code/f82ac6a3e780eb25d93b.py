"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import linprog
from itertools import product

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin
    
    def hex_vertices(cx, cy, side, angle_deg):
        """Return vertices of a regular hexagon (side length 'side') centered at (cx,cy), rotated by angle_deg degrees."""
        angles = np.radians(angle_deg) + np.linspace(0, 2*np.pi, 7)[:-1]
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])
    
    def hex_edge_constraints(cx, cy, side, angle_deg):
        """
        For a hexagon centered at (cx, cy) with side length 'side' and rotation angle_deg,
        return the 6 half-plane constraints (ax, by, c) such that ax*x + by*y <= c
        describes the interior (including boundary).
        """
        # For a regular hexagon, the 6 sides correspond to 6 normals.
        # The distance from center to each side (apothem) is side * sqrt(3)/2
        apothem = side * np.sqrt(3) / 2.0
        base_angles = np.radians(angle_deg) + np.linspace(0, 2*np.pi, 7)[:-1]
        # The normal to each side points outward. For a vertex at angle theta,
        # the side is perpendicular, so the normal is at theta.
        # Actually, for a regular hexagon, if vertices are at angles 0, 60, ..., 300,
        # the sides are at angles 30, 90, ..., 330.
        # Let's be precise.
        # Vertices at angles: theta_i = angle_deg + i*60
        # Side i connects vertex i and i+1. The midpoint of side i is at angle theta_i + 30.
        # The normal to side i points outward, in direction theta_i + 30.
        normals = base_angles + np.pi / 6.0
        ax = apothem * np.cos(normals)
        by = apothem * np.sin(normals)
        # The line equation: ax*(x-cx) + by*(y-cy) = apothem^2 ? No.
        # The support function: max_{v in hex} (n . v) = apothem (if n is unit normal).
        # Here, n = (cos(normal), sin(normal)) is unit.
        # The constraint is n . (x - cx, y - cy) <= apothem
        # => (cos(normal) * x + sin(normal) * y) <= cos(normal)*cx + sin(normal)*cy + apothem
        c = np.cos(normals) * cx + np.sin(normals) * cy + apothem
        return np.column_stack([np.cos(normals), np.sin(normals), c])
    
    def min_enclosing_hexagon(inner_hexes):
        """
        Find the minimum side length of a regular hexagon that encloses all given inner hexagons.
        inner_hexes: list of (cx, cy, side, angle_deg)
        Returns: (center_x, center_y, side_length, angle_deg)
        """
        # For a given orientation theta of the outer hexagon, the problem becomes:
        # minimize R (apothem) such that for all inner hexagons, all their vertices
        # satisfy the 6 half-plane constraints of the outer hexagon.
        # Outer hexagon with apothem R, center (cx, cy), orientation theta:
        # Constraint k: cos(theta_k) * (x - cx) + sin(theta_k) * (y - cy) <= R
        # where theta_k = theta + k*pi/3 + pi/6 (for k=0..5)
        
        # This is a linear program in variables: cx, cy, R
        # Minimize R
        # Subject to: cos(theta_k)*x_i + sin(theta_k)*y_i - cos(theta_k)*cx - sin(theta_k)*cy <= R
        #             for all k=0..5, for all vertices (x_i, y_i) of all inner hexagons.
        
        # Variables: [cx, cy, R]
        # c = [0, 0, 1]
        # A_ub * [cx, cy, R] <= b_ub
        
        best_side = np.inf
        best_center = (0, 0)
        best_angle = 0
        
        # We need to try different orientations theta of the outer hexagon.
        # The optimal theta is likely aligned with one of the inner hexagons' orientations,
        # or at some critical angle. For simplicity, try a range of angles.
        
        for outer_angle_deg in np.linspace(0, 60, 61):
            outer_angle = np.radians(outer_angle_deg)
            normals_angles = outer_angle + np.linspace(0, 2*np.pi, 7)[:-1] + np.pi/6.0
            cos_n = np.cos(normals_angles)
            sin_n = np.sin(normals_angles)
            
            # Collect all vertices of all inner hexagons
            all_vertices = []
            for (icx, icy, iside, iangle) in inner_hexes:
                verts = hex_vertices(icx, icy, iside, iangle)
                all_vertices.append(verts)
            all_vertices = np.vstack(all_vertices)
            
            # For each normal direction k, and each vertex, we have a constraint.
            # cos_n[k] * x_i + sin_n[k] * y_i - cos_n[k] * cx - sin_n[k] * cy <= R
            # => -cos_n[k] * cx - sin_n[k] * cy + R >= cos_n[k] * x_i + sin_n[k] * y_i
            # => cos_n[k] * cx + sin_n[k] * cy - R <= -(cos_n[k] * x_i + sin_n[k] * y_i)
            
            n_vertices = len(all_vertices)
            n_constraints = 6 * n_vertices
            
            A_ub = np.zeros((n_constraints, 3))
            b_ub = np.zeros(n_constraints)
            
            idx = 0
            for k in range(6):
                c_k = cos_n[k]
                s_k = sin_n[k]
                for v in all_vertices:
                    A_ub[idx, 0] = c_k
                    A_ub[idx, 1] = s_k
                    A_ub[idx, 2] = -1.0
                    b_ub[idx] = -(c_k * v[0] + s_k * v[1])
                    idx += 1
            
            # Objective: minimize R
            c_obj = np.array([0.0, 0.0, 1.0])
            
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, method='highs')
                if res.success:
                    R = res.x[2]
                    side_length = R * 2.0 / np.sqrt(3)
                    if side_length < best_side:
                        best_side = side_length
                        best_center = (res.x[0], res.x[1])
                        best_angle = outer_angle_deg
            except Exception:
                pass
            
            if time.time() > deadline:
                break
        
        if best_side == np.inf:
            # Fallback: return a large hexagon
            return (0.0, 0.0, 100.0, 0.0)
        
        return (best_center[0], best_center[1], best_side, best_angle)
    
    # Define candidate cluster configurations for N=11 unit hexagons.
    # Each configuration is a list of (dx, dy, angle_deg) relative to cluster center.
    # We'll try a few dense packings.
    
    configs = []
    
    # Configuration 1: Hexagonal grid, 3-4-3-1 or similar.
    # Actually, let's use a known dense packing.
    # For 11 hexagons, a 3x3 grid plus 2 extra, or 4x3 minus 1, etc.
    # Let's try a 3x3 grid (9 hexagons) plus 2 attached.
    
    # Hexagonal grid coordinates:
    # dx = (col - row/2) * sqrt(3)
    # dy = row * 1.5
    # For unit hexagons (side=1), distance between centers in grid is sqrt(3).
    
    def hex_grid_positions(rows, cols, offset_row=0, offset_col=0):
        positions = []
        for r in range(rows):
            for c in range(cols):
                x = (c + offset_col - (r + offset_row) / 2.0) * np.sqrt(3)
                y = (r + offset_row) * 1.5
                positions.append((x, y, 0.0))
        return positions
    
    # Config 1: 3x3 grid
    configs.append(hex_grid_positions(3, 3))
    
    # Config 2: 4x3 grid minus one corner
    full_4x3 = hex_grid_positions(4, 3)
    configs.append(full_4x3)
    
    # Config 3: 5x3 grid minus 4
    full_5x3 = hex_grid_positions(5, 3)
    configs.append(full_5x3)
    
    # Config 4: 2x4 grid plus 3
    # Let's just try rectangular grids and see.
    
    best_result = None
    best_side = np.inf
    
    for config in configs:
        if len(config) != N:
            continue
        if time.time() > deadline:
            break
        
        # Center the configuration
        xs = [p[0] for p in config]
        ys = [p[1] for p in config]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        
        centered_config = [(p[0] - cx, p[1] - cy, p[2]) for p in config]
        
        # Add some random perturbations to break symmetry and explore different packings
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            if attempt == 0:
                perturbed = centered_config
            else:
                # Random small perturbation
                rng = np.random.default_rng(attempt)
                perturbed = []
                for (x, y, a) in centered_config:
                    dx = rng.normal(0, 0.1)
                    dy = rng.normal(0, 0.1)
                    da = rng.normal(0, 10)
                    perturbed.append((x + dx, y + dy, a + da))
            
            # Convert to inner hexagons: (cx, cy, side, angle_deg)
            inner_hexes = [(x, y, 1.0, a) for (x, y, a) in perturbed]
            
            # Find min enclosing hexagon
            center_x, center_y, side, angle = min_enclosing_hexagon(inner_hexes)
            
            if side < best_side:
                best_side = side
                best_result = (perturbed, (center_x, center_y), side, angle)
            
            if time.time() > deadline:
                break
    
    if best_result is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    perturbed, center, side, angle = best_result
    inner = [[x, y, a] for (x, y, a) in perturbed]
    return inner, center, side, angle
# EVOLVE-BLOCK-END
