"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.spatial import cKDTree

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Hexagon geometry: side length 1, flat-top orientation
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # Circumradius = 1, inradius = sqrt(3)/2
    # Width (flat-to-flat) = sqrt(3), Height (point-to-point) = 2
    
    # For two disjoint unit hexagons, the minimum center distance depends on relative rotation.
    # For same orientation, minimum distance is 2 (when aligned along a vertex direction) or sqrt(3) (along flat direction).
    # Actually, for same orientation, the minimum distance to avoid overlap is 2 if they're side-by-side horizontally (vertex touching).
    # Let's use a conservative minimum distance of 2.0 for same-orientation hexagons.
    
    # Strategy: Place hexagons in a honeycomb-like pattern but optimize the outer hexagon size.
    # A regular hexagon of side length S can contain unit hexagons arranged in a hexagonal grid.
    
    # For a hexagonal packing of unit hexagons with side length 1:
    # The distance between centers of adjacent hexagons (same orientation) is 2.
    # In a hexagonal lattice, the centers form a triangular lattice with spacing 2.
    
    # Let's try to fit N hexagons in the smallest possible outer hexagon.
    # We'll use a hexagonal arrangement and compute the required outer side length.
    
    def hexagon_vertices(cx, cy, angle_deg, side=1.0):
        angle_rad = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + math.radians(60 * i)
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices
    
    def point_in_hexagon(px, py, cx, cy, angle_deg, side=1.0):
        """Check if point (px,py) is inside hexagon centered at (cx,cy) with given angle."""
        # Transform to hexagon's local coordinate system
        dx = px - cx
        dy = py - cy
        # Rotate by -angle
        angle_rad = -math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        # In local coords, flat-top hexagon with side 1:
        # |x| <= 1, |y| <= sqrt(3)/2, and |x| + sqrt(3)*|y| <= 2
        # Actually for pointy-top: |y| <= 1, |x| <= sqrt(3)/2, sqrt(3)*|x| + |y| <= 2
        # For flat-top (angle 0 means vertex at right): vertices at 0,60,120,180,240,300
        # The hexagon is defined by: |x| <= 1, |y| <= sqrt(3)/2, and |x|/1 + |y|/(sqrt(3)/2) <= 2? No.
        # Let's use the standard: for a regular hexagon with circumradius R=1,
        # the distance from center to each side (inradius) is sqrt(3)/2.
        # In local coords, the hexagon is the intersection of 6 half-planes.
        # For flat-top: sides are horizontal and at ±60 degrees.
        # The constraints are:
        # |y| <= sqrt(3)/2
        # |x| <= 1
        # |x|/1 + |y|/(sqrt(3)/2) <= 2 is wrong.
        
        # Correct: for a regular hexagon with circumradius 1, vertices at (cos(k*pi/3), sin(k*pi/3))
        # The supporting lines are at distance sqrt(3)/2 from center.
        # The 6 normals are at angles 30, 90, 150, 210, 270, 330 degrees (for flat-top with vertex at 0)
        # Wait, if vertices are at 0,60,...,300, then the sides are perpendicular to directions 30,90,...,330.
        # So the constraints are: for each normal direction theta_k = 30 + 60*k degrees,
        # (lx, ly) dot (cos(theta_k), sin(theta_k)) <= sqrt(3)/2
        
        max_dot = 0
        for k in range(6):
            theta = math.radians(30 + 60 * k)
            dot = lx * math.cos(theta) + ly * math.sin(theta)
            if dot > max_dot:
                max_dot = dot
        return max_dot <= math.sqrt(3) / 2 + 1e-9
    
    def outer_hexagon_contains_inner(px, py, angle_deg, ocx, ocy, oangle_deg, oside):
        """Check if inner hexagon at (px,py) with angle is inside outer hexagon."""
        # Check all 6 vertices of inner hexagon
        verts = hexagon_vertices(px, py, angle_deg)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, ocx, ocy, oangle_deg, oside):
                return False
        return True
    
    def check_valid(inner, ocx, ocy, oside, oangle_deg):
        """Check if all inner hexagons are inside outer and don't overlap."""
        # Check containment
        for x, y, a in inner:
            if not outer_hexagon_contains_inner(x, y, a, ocx, ocy, oangle_deg, oside):
                return False
        
        # Check non-overlap using minimum distance
        # For simplicity, use distance > 2.0 - epsilon for same orientation
        # For different orientations, this is an approximation
        n = len(inner)
        for i in range(n):
            for j in range(i + 1, n):
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                # Conservative: require distance >= 2.0
                if dist < 2.0 - 1e-6:
                    return False
        return True
    
    # Generate candidate positions in a hexagonal grid pattern
    # Center the pattern at origin
    positions = []
    # Hexagonal grid with spacing 2.0
    # Row r, col c: x = 2*c + (r%2)*1, y = r * sqrt(3)
    # Actually for triangular lattice with spacing 2:
    # x = 2*c + (r % 2) * 1.0
    # y = r * sqrt(3)
    
    max_radius = 15  # Search radius
    for r in range(-max_radius, max_radius + 1):
        for c in range(-max_radius, max_radius + 1):
            x = 2.0 * c + (r % 2) * 1.0
            y = r * math.sqrt(3)
            if x * x + y * y <= (max_radius * 2) ** 2:
                positions.append((x, y))
    
    # Sort by distance from origin
    positions.sort(key=lambda p: p[0] * p[0] + p[1] * p[1])
    
    best_result = None
    best_score = float('inf')
    
    # Try different numbers of hexagons and arrangements
    # For N hexagons, try to find the smallest outer hexagon
    
    # Simple approach: try placing N hexagons in a compact hexagonal cluster
    # and compute the required outer hexagon size
    
    def compute_outer_size(positions_subset, angle_deg=0.0):
        """Compute the minimum outer hexagon side length to contain all positions."""
        # The outer hexagon should be centered at the centroid
        xs = [p[0] for p in positions_subset]
        ys = [p[1] for p in positions_subset]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        
        # For a regular hexagon with circumradius R (side = R), 
        # the maximum distance from center to any point inside is R.
        # But we need to account for the inner hexagons having size.
        # The inner hexagon vertices extend up to distance 1 from its center.
        # So we need outer_circumradius >= max_dist + 1
        
        max_dist = 0
        for x, y in positions_subset:
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            max_dist = max(max_dist, d)
        
        # This is a lower bound; the actual required size depends on orientation
        # For now, use this as an estimate
        return max_dist + 1.0, (cx, cy)
    
    # Try compact clusters of N hexagons
    # A hexagonal cluster of N hexagons: 1, 7, 19, 37, 61, ... (centered hexagonal numbers)
    # For N=11, we can use a 7-hex cluster plus 4 more
    
    # Let's just try to greedily place hexagons and compute the bounding hexagon
    import random
    random.seed(42)
    
    # Try a few random perturbations of the hexagonal grid
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Create a compact arrangement
        # Start with a central hexagon and add neighbors
        placed = [(0.0, 0.0)]
        
        # Add neighbors in order of distance
        candidate_positions = positions.copy()
        random.shuffle(candidate_positions)
        
        for x, y in candidate_positions:
            if len(placed) >= N:
                break
            # Check if this position is far enough from all placed
            valid = True
            for px, py in placed:
                d = math.sqrt((x - px) ** 2 + (y - py) ** 2)
                if d < 2.0 - 1e-6:
                    valid = False
                    break
            if valid:
                placed.append((x, y))
        
        if len(placed) < N:
            continue
        
        # Compute outer hexagon
        oside, (ocx, ocy) = compute_outer_size(placed)
        
        # Verify
        inner = [[x, y, 0.0] for x, y in placed]
        if check_valid(inner, ocx, ocy, oside, 0.0):
            if oside < best_score:
                best_score = oside
                best_result = (inner, (ocx, ocy), oside, 0.0)
    
    if best_result is None:
        # Fallback: use the original simple arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END
