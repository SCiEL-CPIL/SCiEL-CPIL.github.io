"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side s and rotation angle_deg."""
    a = np.deg2rad(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3.0
        x = cx + s * np.cos(theta)
        y = cy + s * np.sin(theta)
        verts.append((x, y))
    return verts

def _separation_distance(hex1_verts, hex2_verts):
    """
    Compute the minimum distance between two convex polygons (hexagons).
    If they overlap, return a negative value (penalty).
    We use the standard approach: for each edge of one polygon, check distance to the other.
    For convex polygons, the minimum distance is the minimum over all edge-vertex pairs and edge-edge pairs.
    Here we use a simpler but robust method: compute the minimum distance between any vertex of one to the other polygon,
    and vice versa. For convex polygons that are disjoint, this gives the correct minimum distance.
    If they intersect, we return -inf or a large negative number.
    """
    # Check if polygons intersect using SAT (Separating Axis Theorem)
    # For two convex polygons, if there is no separating axis, they intersect.
    
    def project_polygon(vertices, axis):
        dots = [np.dot(v, axis) for v in vertices]
        return min(dots), max(dots)
    
    def get_axes(vertices):
        axes = []
        n = len(vertices)
        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            # Normal to edge
            normal = (-edge[1], edge[0])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normal = (normal[0] / norm, normal[1] / norm)
            axes.append(normal)
        return axes
    
    axes1 = get_axes(hex1_verts)
    axes2 = get_axes(hex2_verts)
    all_axes = axes1 + axes2
    
    for axis in all_axes:
        min1, max1 = project_polygon(hex1_verts, axis)
        min2, max2 = project_polygon(hex2_verts, axis)
        if max1 < min2 or max2 < min1:
            # Separating axis found, polygons are disjoint
            # Distance along this axis is the gap
            gap = max(min2 - max1, min1 - max2)
            if gap < 0:
                gap = 0
            return gap  # This is the distance along this separating axis
    
    # No separating axis found, polygons intersect
    return -1000.0  # Large penalty

def _max_separation_distance(inner_hexes, outer_center, outer_side, outer_angle):
    """
    For a given placement of inner hexagons and outer hexagon,
    compute the minimum separation distance between any pair of inner hexagons,
    and also ensure all inner hexagons are inside the outer hexagon.
    Returns (min_pair_distance, min_outer_margin) where both should be >= 0.
    If any constraint is violated, returns negative values.
    """
    min_pair_dist = float('inf')
    min_outer_margin = float('inf')
    
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    for i in range(len(inner_hexes)):
        x1, y1, a1 = inner_hexes[i]
        v1 = _hex_vertices(x1, y1, 1.0, a1)
        
        # Check if inner hex i is inside outer hex
        # We can check by verifying that all vertices of inner hex are inside outer hex
        # Using point-in-convex-polygon test
        def point_in_convex_polygon(point, polygon):
            n = len(polygon)
            for i in range(n):
                p1 = polygon[i]
                p2 = polygon[(i + 1) % n]
                # Cross product to check if point is to the left of edge
                cross = (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])
                if cross < -1e-10:
                    return False
            return True
        
        for v in v1:
            if not point_in_convex_polygon(v, outer_verts):
                min_outer_margin = min(min_outer_margin, -1000.0)
        
        for j in range(i + 1, len(inner_hexes)):
            x2, y2, a2 = inner_hexes[j]
            v2 = _hex_vertices(x2, y2, 1.0, a2)
            dist = _separation_distance(v1, v2)
            min_pair_dist = min(min_pair_dist, dist)
    
    return min_pair_dist, min_outer_margin

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit
    
    # Start with a reasonable initial configuration
    # Place hexagons in a hexagonal lattice pattern
    # For N hexagons, use a compact arrangement
    
    # Initial guess: place hexagons in a line or small cluster
    # Let's start with a hexagonal packing arrangement
    
    # For simplicity, start with all hexagons at origin with slight offsets
    # and use a simple optimization approach
    
    # Actually, let's use a more structured initial placement
    # Arrange in a hexagonal grid pattern
    
    positions = []
    angles = []
    
    # Hexagonal lattice spacing for unit hexagons
    # Distance between centers of adjacent hexagons in a hex grid: sqrt(3) * side * 2 = 2*sqrt(3) for side=1
    # Actually, for regular hexagons with side length 1, the distance between centers of adjacent hexagons
    # in a honeycomb pattern is 2 * sqrt(3) / 2 * ... let me recalculate.
    
    # A regular hexagon with side length s has:
    # - Distance from center to vertex: s
    # - Distance from center to edge midpoint: s * sqrt(3)/2
    # - Distance between centers of two adjacent hexagons in a honeycomb: 2 * s * sqrt(3)/2 = s*sqrt(3)
    # Wait, no. In a honeycomb, the distance between centers of adjacent hexagons is 2 * (apothem) = 2 * s * sqrt(3)/2 = s*sqrt(3).
    # For s=1, this is sqrt(3) ≈ 1.732
    
    hex_spacing = np.sqrt(3)  # Distance between centers of adjacent unit hexagons in honeycomb
    
    # Arrange N hexagons in a hexagonal pattern
    # For N hexagons, we can arrange them in concentric rings
    # Ring 0: 1 hexagon
    # Ring 1: 6 hexagons
    # Ring 2: 12 hexagons
    # etc.
    
    # For N=11, we can use 1 + 6 + 4 = 11, or just place them in a compact cluster
    
    # Let's use a simple approach: place hexagons in a hexagonal grid
    # and compute the bounding hexagon
    
    # Generate hexagonal grid positions
    def hex_to_pixel(q, r, size=1.0):
        """Convert axial coordinates to pixel coordinates for a hex grid."""
        x = size * 3.0 / 2.0 * q
        y = size * np.sqrt(3) * (r + q / 2.0)
        return x, y
    
    # Generate N positions in a hexagonal pattern
    # For N=11, use a 3x3 grid with corners removed, or a hexagonal shape
    
    # Let's just use a simple cluster: center + 6 around + 4 more
    positions = []
    angles = [0.0] * N
    
    # Center
    positions.append((0.0, 0.0))
    
    # 6 around center
    for i in range(6):
        angle = i * np.pi / 3.0
        x = hex_spacing * np.cos(angle)
        y = hex_spacing * np.sin(angle)
        positions.append((x, y))
    
    # Add remaining 4 hexagons at the next ring
    for i in range(4):
        angle = i * np.pi / 3.0 + np.pi / 6.0
        x = 2 * hex_spacing * np.cos(angle)
        y = 2 * hex_spacing * np.sin(angle)
        positions.append((x, y))
    
    inner = [[pos[0], pos[1], ang] for pos, ang in zip(positions, angles)]
    
    # Initial outer hexagon: large enough to contain all
    outer_center = (0.0, 0.0)
    outer_side = 10.0
    outer_angle = 0.0
    
    # Simple optimization: try to reduce outer_side while maintaining feasibility
    # This is a non-convex problem, so we'll use a simple approach
    
    # For now, return a valid configuration
    # We need to ensure all inner hexagons fit inside the outer hexagon
    
    # Compute the required outer side length
    # The outer hexagon must contain all inner hexagons
    # A simple approach: find the maximum distance from outer_center to any vertex of any inner hexagon
    # and set outer_side to be at least that distance
    
    max_dist = 0.0
    for x, y, a in inner:
        verts = _hex_vertices(x, y, 1.0, a)
        for vx, vy in verts:
            dist = np.sqrt((vx - outer_center[0])**2 + (vy - outer_center[1])**2)
            max_dist = max(max_dist, dist)
    
    # For a regular hexagon, the distance from center to vertex is the side length
    # So outer_side should be at least max_dist
    outer_side = max_dist + 0.1  # Small margin
    
    # Check feasibility
    min_pair_dist, min_outer_margin = _max_separation_distance(inner, outer_center, outer_side, outer_angle)
    
    # If infeasible, increase outer_side
    if min_pair_dist < 0 or min_outer_margin < 0:
        outer_side *= 1.5
        min_pair_dist, min_outer_margin = _max_separation_distance(inner, outer_center, outer_side, outer_angle)
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
