"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Vertices of a regular hexagon (flat-top when angle=0)."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_to_edge_distance(p, e1, e2):
        """Distance from point p to edge e1-e2 (positive if outside)."""
        d = e2 - e1
        L2 = np.dot(d, d)
        if L2 < 1e-12:
            return np.linalg.norm(p - e1)
        t = np.dot(p - e1, d) / L2
        t = max(0.0, min(1.0, t))
        closest = e1 + t * d
        return np.linalg.norm(p - closest)

    def min_vertex_edge_distance(hex1_verts, hex2_verts):
        """Minimum distance from vertices of hex1 to edges of hex2 and vice versa.
        Negative if overlapping."""
        min_dist = np.inf
        # Check vertices of hex1 against edges of hex2
        for v in hex1_verts:
            for i in range(6):
                e1 = hex2_verts[i]
                e2 = hex2_verts[(i + 1) % 6]
                d = point_to_edge_distance(v, e1, e2)
                if d < min_dist:
                    min_dist = d
        # Check vertices of hex2 against edges of hex1
        for v in hex2_verts:
            for i in range(6):
                e1 = hex1_verts[i]
                e2 = hex1_verts[(i + 1) % 6]
                d = point_to_edge_distance(v, e1, e2)
                if d < min_dist:
                    min_dist = d
        return min_dist

    def polygons_disjoint_sat(poly1, poly2, tol=1e-9):
        """Check if two convex polygons are disjoint using SAT."""
        for poly in [poly1, poly2]:
            for i in range(len(poly)):
                v1 = poly[i]
                v2 = poly[(i + 1) % len(poly)]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                
                proj1 = np.dot(poly1, normal)
                proj2 = np.dot(poly2, normal)
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2 - tol or max2 < min1 - tol:
                    return True
        return False

    def compute_outer_side(inner_hexes):
        """
        Compute minimum outer hexagon side length to contain all inner hexagons.
        Assumes outer hexagon is centered at (0,0) and flat-top (angle=0).
        """
        max_required = 0.0
        # Normals for a flat-top hexagon centered at origin
        # Edges are perpendicular to angles 0, 60, 120, 180, 240, 300
        normals = np.array([
            [np.cos(np.radians(a)), np.sin(np.radians(a))] 
            for a in [0, 60, 120, 180, 240, 300]
        ])
        
        for hex_verts in inner_hexes:
            for v in hex_verts:
                # Project vertex onto normals
                dists = np.dot(normals, v)
                # The required apothem is the max projection
                # Side S = 2 * apothem / sqrt(3)
                max_proj = np.max(dists)
                required_S = max_proj * 2.0 / np.sqrt(3.0)
                if required_S > max_required:
                    max_required = required_S
        return max_required

    def objective(params):
        if time.time() > deadline:
            return 1e6
        
        # Parse params: x1, y1, a1, ..., x12, y12, a12
        inner_x = params[0:12]
        inner_y = params[12:24]
        inner_a = params[24:36]
        
        inner_hexes = []
        for i in range(N):
            verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
            inner_hexes.append(verts)
            
        # Calculate side length
        side = compute_outer_side(inner_hexes)
        
        # Calculate overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dist = min_vertex_edge_distance(inner_hexes[i], inner_hexes[j])
                if dist < 0:
                    # Smooth penalty for overlap
                    penalty += 10.0 * dist * dist
        
        return side + penalty

    # Generate initial configuration: Honeycomb lattice
    # Centered at 0,0
    # 12 hexagons can be arranged in a 4x3 grid or similar
    # Let's try a compact cluster
    np.random.seed(42)
    
    # Initial guess: 4 columns, 3 rows, staggered
    sqrt3 = np.sqrt(3)
    hx, hy, ha = [], [], []
    # Row 0: 4 hexes
    for c in range(4):
        x = (c - 1.5) * sqrt3 * 1.05
        y = 0.0
        hx.append(x); hy.append(y); ha.append(0.0)
    # Row 1: 4 hexes, shifted
    for c in range(4):
        x = (c - 1.5) * sqrt3 * 1.05 + sqrt3 * 1.05 / 2
        y = 1.55
        hx.append(x); hy.append(y); ha.append(0.0)
    # Row 2: 4 hexes
    for c in range(4):
        x = (c - 1.5) * sqrt3 * 1.05
        y = 3.1
        hx.append(x); hy.append(y); ha.append(0.0)
        
    # Center the configuration
    hx = np.array(hx); hy = np.array(hy); ha = np.array(ha)
    hx -= np.mean(hx); hy -= np.mean(hy)
    
    init_params = np.concatenate([hx, hy, ha])
    
    # Phase 1: Differential Evolution
    bounds = [(-6, 6)] * 12 + [(-6, 6)] * 12 + [(-np.pi, np.pi)] * 12
    
    best_params = init_params
    best_obj = objective(best_params)
    
    try:
        de_result = differential_evolution(
            objective, bounds,
            seed=42, maxiter=200, popsize=30,
            tol=1e-6, mut=(0.5, 1.0), recomb=0.8,
            polish=False, init='sohal',
            x0=init_params
        )
        if de_result.fun < best_obj:
            best_obj = de_result.fun
            best_params = de_result.x
    except Exception:
        pass

    # Phase 2: Local refinement
    if time.time() < deadline - 20:
        try:
            result = minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass

    # Phase 3: Additional local refinements from perturbed best
    if time.time() < deadline - 15:
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            perturbed = best_params + np.random.normal(0, 0.05, len(best_params))
            try:
                result = minimize(
                    objective, perturbed, method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_params = result.x
            except Exception:
                pass

    # Extract solution
    params = best_params
    inner_x = params[0:12]
    inner_y = params[12:24]
    inner_a = params[24:36]
    
    inner = [[inner_x[i], inner_y[i], np.degrees(inner_a[i])] for i in range(N)]
    
    # Verify validity
    inner_hexes = []
    for i in range(N):
        verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
        inner_hexes.append(verts)
    
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            if not polygons_disjoint_sat(inner_hexes[i], inner_hexes[j], 1e-7):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to a known valid configuration with wider spacing
        inner = [[3.5 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 35.0, 0.0
    
    # Recompute exact side length without penalty
    exact_side = compute_outer_side(inner_hexes)
    
    # Return: inner, (center_x, center_y), side, angle_degrees
    # Outer hexagon is fixed at (0,0) with angle 0
    return inner, (0.0, 0.0), exact_side, 0.0

# EVOLVE-BLOCK-END
