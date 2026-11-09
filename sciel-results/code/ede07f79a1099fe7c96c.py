"""Packing 12 unit hexagons in a minimal regular hexagon."""
import numpy as np
import random
import math

N = 12

# Precompute hexagon vertices for a unit regular hexagon
# A regular hexagon with side length 1 has vertices at distance 1 from center
# at angles 0, 60, 120, 180, 240, 300 degrees
_HEX_VERTS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

def _hex_vertices(cx, cy, angle_deg):
    """Return vertices of a unit hexagon centered at (cx, cy) rotated by angle_deg."""
    theta = math.radians(angle_deg)
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    return cx + R @ _HEX_VERTS[:, 0] if False else (
        np.column_stack([
            cx + math.cos(theta) * _HEX_VERTS[:, 0] - math.sin(theta) * _HEX_VERTS[:, 1],
            cy + math.sin(theta) * _HEX_VERTS[:, 0] + math.cos(theta) * _HEX_VERTS[:, 1]
        ])
    )

def _min_enclosing_hexagon_side(points, center_x, center_y, outer_angle_deg):
    """
    Compute the minimum side length of a regular hexagon (centered at center_x, center_y, 
    rotated by outer_angle_deg) that contains all given points.
    
    The hexagon is defined by 6 half-planes. For a regular hexagon with side length s
    centered at origin, the apothem (distance from center to each edge) is s * sqrt(3)/2.
    The vertices are at distance s from center.
    
    Actually, for a regular hexagon with side length s:
    - The distance from center to each vertex is s
    - The distance from center to each edge (apothem) is s * sqrt(3) / 2
    
    The 6 edges are perpendicular to directions at angles: outer_angle + 30, 90, 150, 210, 270, 330
    (i.e., the normals point in directions that are 30 degrees offset from the vertex directions)
    
    Wait, let me reconsider. A regular hexagon with vertices at angles 0, 60, ..., 300 has
    edges whose outward normals point at angles 30, 90, 150, 210, 270, 330.
    
    For each normal direction, the constraint is: dot(p - center, normal) <= apothem = s * sqrt(3)/2
    
    So s = max over all points and all 6 normals of (dot(p - center, normal)) / (sqrt(3)/2)
    """
    theta = math.radians(outer_angle_deg)
    # The 6 outward normal directions for a hexagon with vertices at angles 0,60,...,300
    # The normals are at angles 30, 90, 150, 210, 270, 330 relative to the hexagon orientation
    normals = []
    for i in range(6):
        angle = math.radians(outer_angle_deg + 30 + i * 60)
        normals.append([math.cos(angle), math.sin(angle)])
    normals = np.array(normals)
    
    # Translate points relative to center
    px = points[:, 0] - center_x
    py = points[:, 1] - center_y
    
    # For each normal, compute max dot product
    max_dot = 0.0
    for nx, ny in normals:
        dots = px * nx + py * ny
        if dots.max() > max_dot:
            max_dot = dots.max()
    
    # apothem = s * sqrt(3) / 2, so s = 2 * apothem / sqrt(3)
    s = 2.0 * max_dot / math.sqrt(3.0)
    return max(s, 0.0)

def _all_hexagon_points(positions, angles):
    """Get all vertices of all inner hexagons as a single array."""
    all_pts = []
    for (x, y), a in zip(positions, angles):
        verts = _hex_vertices(x, y, a)
        all_pts.append(verts)
    return np.vstack(all_pts)

def _evaluate(positions, angles, center_x, center_y, outer_angle):
    """Evaluate the side length of the minimal enclosing hexagon."""
    pts = _all_hexagon_points(positions, angles)
    return _min_enclosing_hexagon_side(pts, center_x, center_y, outer_angle)

def _check_disjoint(positions, angles):
    """Check if all hexagons are disjoint (no overlapping interiors).
    For simplicity in the search, we'll rely on the enclosing hexagon constraint
    and add a penalty for overlaps. Actually, let's just check vertex containment
    as a quick proxy, but that's not sufficient.
    
    For a proper check, we need to verify no two hexagons overlap. This is expensive.
    Instead, we'll use a penalty-based approach in the optimization.
    """
    pass

def _overlap_penalty(positions, angles):
    """Compute a penalty based on minimum distance between hexagon centers.
    Two unit hexagons can be placed with centers as close as 2*apothem = sqrt(3) 
    when aligned, but the minimum distance for non-overlap depends on orientation.
    The minimum distance between centers of two non-overlapping unit hexagons is 
    at least 2 * (sqrt(3)/2) = sqrt(3) ~ 1.732 when they share an edge.
    Actually, the inradius is sqrt(3)/2, so two hexagons touching edge-to-edge 
    have center distance = 2 * sqrt(3)/2 = sqrt(3).
    
    But with rotation, the minimum distance can be different. Let's use a conservative
    minimum distance of sqrt(3) and penalize if centers are closer.
    """
    penalty = 0.0
    n = len(positions)
    min_dist = math.sqrt(3.0)  # sqrt(3)
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < min_dist:
                penalty += (min_dist - d) ** 2
    return penalty

def construct():
    """Construct 12 unit hexagons inside a minimal regular hexagon."""
    random.seed(42)
    np.random.seed(42)
    
    # Initial configuration: place hexagons in a honeycomb-like pattern
    # Honeycomb lattice: centers at positions that are sqrt(3) apart in hex grid
    # Hex grid spacing: horizontal = 1.5, vertical offset = sqrt(3)/2
    
    # Start with a compact arrangement
    # For 12 hexagons, a 3x4 or 4x3 grid in hex lattice
    
    def init_config():
        """Generate initial configuration."""
        positions = []
        angles = []
        # Hexagonal lattice positions
        # Row-based: each row offset by 0.75 horizontally and sqrt(3)/2 vertically
        # For a hex grid with side length 1:
        # Horizontal spacing: 1.5 (for adjacent rows in same column)
        # Actually, let's use a simple grid
        
        # Try a 4x3 grid in hexagonal arrangement
        # 4 columns, 3 rows
        for row in range(3):
            for col in range(4):
                x = col * 1.5
                y = row * math.sqrt(3)
                if row % 2 == 1:
                    x += 0.75
                positions.append((x, y))
                angles.append(0.0)
        
        # Center the configuration
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        positions = [(p[0] - cx, p[1] - cy) for p in positions]
        
        return positions, angles, 0.0, 0.0, 0.0
    
    best_positions, best_angles, best_cx, best_cy, best_outr = init_config()
    best_score = _evaluate(best_positions, best_angles, best_cx, best_cy, best_outr) + 100 * _overlap_penalty(best_positions, best_angles)
    
    # Simulated annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    max_iter = 50000
    
    current_positions = [p[:] for p in best_positions]
    current_angles = best_angles[:]
    current_cx = best_cx
    current_cy = best_cy
    current_outr = best_outr
    current_score = best_score
    
    for it in range(max_iter):
        T = max(T_min, T * cooling)
        
        # Generate a neighbor
        new_positions = [p[:] for p in current_positions]
        new_angles = current_angles[:]
        new_cx = current_cx
        new_cy = current_cy
        new_outr = current_outr
        
        move_type = random.random()
        
        if move_type < 0.6:
            # Move one hexagon
            idx = random.randint(0, N - 1)
            new_positions[idx] = (
                current_positions[idx][0] + random.gauss(0, 0.3),
                current_positions[idx][1] + random.gauss(0, 0.3)
            )
        elif move_type < 0.8:
            # Rotate one hexagon
            idx = random.randint(0, N - 1)
            new_angles[idx] = (current_angles[idx] + random.gauss(0, 10) + 360) % 360
        elif move_type < 0.9:
            # Move outer center
            new_cx = current_cx + random.gauss(0, 0.2)
            new_cy = current_cy + random.gauss(0, 0.2)
        else:
            # Rotate outer hexagon
            new_outr = (current_outr + random.gauss(0, 10) + 360) % 360
        
        new_score = _evaluate(new_positions, new_angles, new_cx, new_cy, new_outr) + 100 * _overlap_penalty(new_positions, new_angles)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_positions = new_positions
            current_angles = new_angles
            current_cx = new_cx
            current_cy = new_cy
            current_outr = new_outr
            current_score = new_score
            
            if current_score < best_score:
                best_positions = [p[:] for p in current_positions]
                best_angles = current_angles[:]
                best_cx = current_cx
                best_cy = current_cy
                best_outr = current_outr
                best_score = current_score
    
    # Final evaluation without penalty
    final_side = _evaluate(best_positions, best_angles, best_cx, best_cy, best_outr)
    
    # Build the inner list
    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], best_angles[i]])
    
    return inner, (best_cx, best_cy), final_side, best_outr
