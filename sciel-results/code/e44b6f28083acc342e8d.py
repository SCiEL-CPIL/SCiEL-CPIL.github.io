"""Pack 12 unit hexagons in a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get 6 vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + side * math.cos(a)
            y = cy + side * math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def project_on_normal(verts, nx, ny):
        """Project vertices onto a normal direction."""
        return np.dot(verts, [nx, ny])

    def min_side_for_container(inner_hexes, outer_center, outer_angle_deg):
        """
        Compute the minimum side length of a regular hexagon container
        that contains all inner hexagons.
        The outer hexagon has 6 sides; we need all inner vertices to be
        inside the outer hexagon.
        
        For a regular hexagon with center (cx,cy), side length s, and rotation theta,
        the constraint for a point (px,py) to be inside is:
        For each of the 6 outward normals n_i, the projection of (px,py) onto n_i
        must be <= s * sqrt(3)/2 (the apothem) + projection of center onto n_i.
        
        Actually, for a regular hexagon with side length s, the apothem (distance from
        center to each side) is s * sqrt(3) / 2.
        
        The 6 outward normals are at angles theta + i*60 degrees for i=0..5.
        A point is inside if for all i: dot(point - center, n_i) <= apothem
        where apothem = s * sqrt(3) / 2.
        
        So s >= max_i [ dot(point - center, n_i) / (sqrt(3)/2) ] for all points.
        """
        cx, cy = outer_center
        theta = math.radians(outer_angle_deg)
        # 6 outward normals
        normals = []
        for i in range(6):
            a = theta + i * math.pi / 3
            normals.append((math.cos(a), math.sin(a)))
        
        max_ratio = 0.0
        for hx, hy, ha in inner_hexes:
            verts = hex_vertices(hx, hy, ha, 1.0)
            for nx, ny in normals:
                # Project each vertex onto the normal
                projections = (verts[:, 0] - cx) * nx + (verts[:, 1] - cy) * ny
                max_proj = np.max(projections)
                # s must satisfy: max_proj <= s * sqrt(3) / 2
                # so s >= max_proj / (sqrt(3)/2)
                if max_proj > 0:
                    ratio = max_proj / (math.sqrt(3) / 2)
                    if ratio > max_ratio:
                        max_ratio = ratio
        return max_ratio

    def objective(params):
        """Objective: minimize the outer hexagon side length."""
        # params: for each of 12 hexagons: x, y, angle (36 params)
        inner_hexes = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            angle = params[3*i+2]
            inner_hexes.append((x, y, angle))
        
        # Outer center is (0,0), outer angle is 0 for now
        side = min_side_for_container(inner_hexes, (0.0, 0.0), 0.0)
        return side

    # Initial honeycomb layout
    # Hexagonal grid spacing: for unit hexagons, center-to-center distance in honeycomb
    # is sqrt(3) for adjacent hexagons (flat-to-flat)
    # Let's use a 3-4-3-2 arrangement or similar
    # Actually, let's try a compact arrangement:
    # Row structure: 3, 4, 3, 2 = 12 hexagons
    # Or: 4, 4, 4 = 12 (rectangular grid of hexagons)
    
    # For a honeycomb, the horizontal spacing between columns is 1.5 (for pointy-top)
    # or sqrt(3) (for flat-top). Let's use flat-top hexagons.
    # Flat-top: width = 2*side, height = sqrt(3)*side
    # Horizontal spacing = 1.5 * side = 1.5
    # Vertical spacing = sqrt(3) * side = sqrt(3)
    # Odd columns shifted up by sqrt(3)/2
    
    # Let's use a simpler approach: place hexagons in a hexagonal lattice
    # The lattice vectors for a honeycomb of regular hexagons:
    # v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 3/2)
    # Wait, let me think again.
    
    # For unit regular hexagons (side=1), the distance between centers of adjacent
    # hexagons in a tiling is sqrt(3) (the flat-to-flat distance).
    # The 6 neighbors are at distance sqrt(3) in directions 0, 60, 120, 180, 240, 300 degrees.
    
    # Let's place them in a hexagonal pattern:
    # Center at origin, then 6 around it, then 5 more to make 12.
    # Or a 3x4 rectangular grid.
    
    # Try a 3x4 grid of hexagons in a honeycomb arrangement:
    # Columns: 4 columns, each with 3 hexagons
    # Horizontal spacing: sqrt(3)
    # Vertical spacing: sqrt(3)
    # Offset alternating columns by sqrt(3)/2
    
    sqrt3 = math.sqrt(3)
    x0 = 0.0
    y0 = 0.0
    
    inner_hexes = []
    # 4 columns, 3 rows each
    for col in range(4):
        for row in range(3):
            x = x0 + col * sqrt3
            y = y0 + row * sqrt3
            if col % 2 == 1:
                y += sqrt3 / 2
            inner_hexes.append((x, y, 0.0))
    
    # Center the arrangement
    xs = [h[0] for h in inner_hexes]
    ys = [h[1] for h in inner_hexes]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    inner_hexes = [(x - cx, y - cy, a) for x, y, a in inner_hexes]
    
    # Flatten to params
    params0 = []
    for x, y, a in inner_hexes:
        params0.extend([x, y, a])
    
    # Try optimization
    best_params = params0
    best_side = min_side_for_container(inner_hexes, (0.0, 0.0), 0.0)
    
    # Nelder-Mead optimization
    try:
        result = minimize(
            objective,
            params0,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        if result.fun < best_side:
            best_side = result.fun
            best_params = result.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 5.0:
        # Return best found so far
        inner = []
        for i in range(N):
            inner.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
        return inner, (0.0, 0.0), best_side, 0.0
    
    # Try a few more local optimizations with different initial configurations
    for restart in range(3):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb the initial layout
        np.random.seed(42 + restart)
        params_perturbed = list(params0)
        for i in range(len(params_perturbed)):
            if i % 3 != 2:  # Don't perturb angles too much
                params_perturbed[i] += np.random.normal(0, 0.1)
        
        try:
            result = minimize(
                objective,
                params_perturbed,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if result.fun < best_side:
                best_side = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Build final inner hexagons
    inner = []
    for i in range(N):
        inner.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    
    return inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END

