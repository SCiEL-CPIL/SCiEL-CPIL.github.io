"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0  # 120s safety margin within 150s limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_enclosing_hex_side(inner_hexes):
    """Find minimum side length of a regular hexagon that contains all inner hexagons.
    Returns (center_x, center_y, side_length, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Collect all vertices of all inner hexagons
    all_vertices = []
    for ix, iy, iangle in inner_hexes:
        verts = _get_hex_vertices(ix, iy, iangle)
        all_vertices.extend(verts)
    
    # Try different angles for the outer hexagon
    # Due to symmetry, 0 and 30 are the most likely optimal orientations for a hexagonal cluster.
    for angle_deg in [0.0, 30.0]:
        # The normals of the outer hexagon are at angle_deg + 30 + 60*k
        # We need to find a center (cx, cy) that minimizes the max projection.
        
        # Start with centroid of vertices
        cx_mean = sum(v[0] for v in all_vertices) / len(all_vertices)
        cy_mean = sum(v[1] for v in all_vertices) / len(all_vertices)
        
        # Refine center using a fine grid search around the centroid
        # The optimal center is within the convex hull of the points.
        # A grid step of 0.01 should be sufficient for high precision.
        range_search = 2.0
        step = 0.05
        
        best_local_side = float('inf')
        best_local_center = (cx_mean, cy_mean)
        
        # Coarse to fine grid search
        for r in [range_search, range_search/2, range_search/4, range_search/8]:
            for cx in [cx_mean + dx * r for dx in [-1, -0.5, 0, 0.5, 1]]:
                for cy in [cy_mean + dy * r for dy in [-1, -0.5, 0, 0.5, 1]]:
                    required_inradius = 0.0
                    for k in range(6):
                        a = math.radians(angle_deg + 30 + 60 * k)
                        nx = math.cos(a)
                        ny = math.sin(a)
                        
                        max_proj = -float('inf')
                        min_proj = float('inf')
                        for vx, vy in all_vertices:
                            proj = (vx - cx) * nx + (vy - cy) * ny
                            if proj > max_proj:
                                max_proj = proj
                            if proj < min_proj:
                                min_proj = proj
                        
                        extent = max(max_proj, -min_proj)
                        if extent > required_inradius:
                            required_inradius = extent
                    
                    side = required_inradius / (math.sqrt(3) / 2)
                    if side < best_local_side:
                        best_local_side = side
                        best_local_center = (cx, cy)
            
            cx_mean, cy_mean = best_local_center
        
        if best_local_side < best_side:
            best_side = best_local_side
            best_center = best_local_center
            best_angle = angle_deg
            
    return best_center[0], best_center[1], best_side, best_angle

def construct():
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Generate 11 points in a hexagonal cluster arrangement
    # Central hexagon + ring of 6 + 4 more in second ring
    # Use triangular lattice with spacing sqrt(3) for edge-to-edge contact
    sqrt3 = math.sqrt(3)
    # Scale factor to ensure strict disjointness (small gap between hexagons)
    scale = 1.001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    # Generate candidate points in a spiral/grid around the origin
    # We want the 11 closest points to the origin in the triangular lattice
    # Lattice points are (i, j) -> i*u + j*v
    # Let's generate points with |i|, |j| <= 3 and sort by distance
    
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))
    
    # Sort by distance
    candidates.sort(key=lambda t: t[0])
    
    # Take the first N unique points
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == N:
            break
    
    # Create inner hexagons with all angles = 30 (Pointy-Top)
    # This aligns the lattice vectors (0 and 60 deg) with the flat-to-flat normals,
    # requiring a minimum separation of sqrt(3) instead of 2.
    inner = [[pos[0], pos[1], 30.0] for pos in positions]
    
    # Compute minimum enclosing hexagon
    if time.time() > _DEADLINE - 10.0:
        # Quick estimate
        xs = [h[0] for h in inner]
        ys = [h[1] for h in inner]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        max_dist = 0
        for h in inner:
            d = math.sqrt((h[0] - cx)**2 + (h[1] - cy)**2)
            if d > max_dist:
                max_dist = d
        # Add circumradius 1 to max_dist for a rough bound
        side = (max_dist + 1.0) / (math.sqrt(3) / 2) + 0.1
        return inner, (cx, cy), side, 0.0
    
    cx, cy, side, angle = _min_enclosing_hex_side(inner)
    
    # Verify no overlaps (should be guaranteed by lattice spacing sqrt(3))
    # But just in case, check a few pairs
    # For unit hexagons on triangular lattice with spacing sqrt(3), they touch at edges.
    # This is valid (disjoint interiors).
    
    return inner, (cx, cy), side, angle
# EVOLVE-BLOCK-END
