"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 12

# EVOLVE-BLOCK-START
import math

def hex_vertices(side=1.0, rotation=0.0):
    """Return 6 vertices of a regular hexagon with given side length and rotation (radians)."""
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + rotation
    return side * np.column_stack([np.cos(angles), np.sin(angles)])

def point_in_hexagon(px, py, L, outer_rotation=0.0):
    """Check if point (px, py) is inside a regular hexagon of side L centered at origin, rotated by outer_rotation."""
    # Transform point to hexagon's local frame
    cos_r = math.cos(outer_rotation)
    sin_r = math.sin(outer_rotation)
    lx = cos_r * px - sin_r * py
    ly = sin_r * px + cos_r * py
    
    # For a regular hexagon with side L centered at origin, flat-to-flat distance = L*sqrt(3)
    # The hexagon is the intersection of 6 half-planes
    # In local frame, the 6 normals are at angles 0, 60, 120, 180, 240, 300 degrees
    # Each edge is at distance L*sqrt(3)/2 from center
    d = L * math.sqrt(3) / 2.0
    for i in range(6):
        angle = i * math.pi / 3.0
        nx = math.cos(angle)
        ny = math.sin(angle)
        if nx * lx + ny * ly > d:
            return False
    return True

def max_hex_dist(px, py, outer_rotation=0.0):
    """Return the minimum L such that point (px,py) is inside hexagon of side L centered at origin with given rotation."""
    cos_r = math.cos(outer_rotation)
    sin_r = math.sin(outer_rotation)
    lx = cos_r * px - sin_r * py
    ly = sin_r * px + cos_r * py
    
    # The required L is max over 6 directions of (n·p) / (sqrt(3)/2)
    # where n are the 6 unit normals
    max_val = 0.0
    for i in range(6):
        angle = i * math.pi / 3.0
        nx = math.cos(angle)
        ny = math.sin(angle)
        val = (nx * lx + ny * ly) * 2.0 / math.sqrt(3)
        if val > max_val:
            max_val = val
    return max_val

def required_L_for_hexagon(center, rotation, outer_rotation=0.0):
    """Return the minimum outer hexagon side L needed to contain a unit hexagon at given center and rotation."""
    verts = hex_vertices(1.0, rotation)
    max_d = 0.0
    for vx, vy in verts:
        px = center[0] + vx
        py = center[1] + vy
        d = max_hex_dist(px, py, outer_rotation)
        if d > max_d:
            max_d = d
    return max_d

def optimal_inner_rotation(center, outer_rotation=0.0):
    """Find the rotation of a unit hexagon at 'center' that minimizes the required outer L."""
    best_L = float('inf')
    best_rot = 0.0
    # Search over 0 to pi/3 (60 degrees) since hexagon has 6-fold symmetry
    for trial in range(12):
        rot = trial * (math.pi / 3.0) / 12.0
        L = required_L_for_hexagon(center, rot, outer_rotation)
        if L < best_L:
            best_L = L
            best_rot = rot
    # Refine around best
    best_rot = best_rot
    for _ in range(2):
        step = (math.pi / 3.0) / 12.0 / 3.0
        for trial in range(3):
            rot = best_rot + (trial - 1) * step
            L = required_L_for_hexagon(center, rot, outer_rotation)
            if L < best_L:
                best_L = L
                best_rot = rot
    return best_rot, best_L

def evaluate_arrangement(positions, outer_rotation=0.0):
    """Evaluate an arrangement: return (max_L, optimal_rotations)."""
    max_L = 0.0
    rotations = []
    for pos in positions:
        rot, L = optimal_inner_rotation(pos, outer_rotation)
        rotations.append(rot)
        if L > max_L:
            max_L = L
    return max_L, rotations

def search():
    """Search for optimal packing."""
    deadline = time.time() + 120.0
    best_L = float('inf')
    best_config = None
    
    # Initial arrangement: 12 hexagons in a 4x3 grid pattern or centered arrangement
    # Let's try several base arrangements
    
    # Arrangement 1: Centered in a 4x3 grid
    # Hexagon spacing: to avoid overlap, center-to-center distance must be >= 2 (for same orientation)
    # For optimal packing, we can use honeycomb-like arrangement
    
    # Let's try a centered arrangement with hexagonal symmetry
    # 12 hexagons: 1 center + 6 around + 5 more? Or 4 rows of 3?
    
    # Try: 4 rows, 3 columns, with hexagonal offset
    base_arrangements = []
    
    # Arrangement: 4 rows x 3 cols, honeycomb
    s = 1.0  # side length of unit hexagons
    # Distance between centers in honeycomb: 2*s for same orientation, but with rotation we can pack tighter
    # For flat-top hexagons, horizontal spacing = 1.5 * 2s = 3s? No...
    # Actually for unit hexagons, the minimum center-to-center distance depends on relative rotation.
    # For same orientation, it's 2 (vertex to vertex) or sqrt(3) (flat to flat).
    # Let's use a reasonable spacing and optimize.
    
    # Let's try different spacings and arrangements
    for row_count in [3, 4]:
        if row_count == 3:
            cols = [4, 4, 4]
        else:
            cols = [3, 4, 4, 3]  # centered
        for spacing in [1.8, 2.0, 2.2, 2.5]:
            positions = []
            for i, n_cols in enumerate(cols):
                y = (i - (len(cols) - 1) / 2.0) * spacing * math.sqrt(3) / 2.0
                for j in range(n_cols):
                    x = (j - (n_cols - 1) / 2.0) * spacing
                    # Add hexagonal offset for odd rows
                    if i % 2 == 1:
                        x += spacing / 2.0
                    positions.append([x, y])
            if len(positions) == N:
                base_arrangements.append(positions)
    
    # Also try a more compact centered arrangement
    # 1 in center, 6 around, 5 more
    positions_centered = [[0.0, 0.0]]
    for i in range(6):
        angle = i * math.pi / 3.0
        positions_centered.append([2.0 * math.cos(angle), 2.0 * math.sin(angle)])
    # Add 5 more
    for i in range(5):
        angle = i * 2 * math.pi / 5.0
        positions_centered.append([3.5 * math.cos(angle), 3.5 * math.sin(angle)])
    base_arrangements.append(positions_centered)
    
    # Search over outer rotation angles
    best_overall_L = float('inf')
    best_overall_config = None
    
    for outer_rot in np.linspace(0, math.pi / 3.0, 13):
        if time.time() > deadline - 10.0:
            break
        for positions in base_arrangements:
            if time.time() > deadline - 10.0:
                break
            L, rotations = evaluate_arrangement(positions, outer_rot)
            if L < best_overall_L:
                best_overall_L = L
                best_overall_config = (positions, rotations, outer_rot)
    
    # Refine the best configuration
    if best_overall_config is not None:
        positions, rotations, outer_rot = best_overall_config
        # Fine-tune outer rotation
        for _ in range(3):
            step = (math.pi / 3.0) / 13.0 / 3.0
            improved = False
            for trial in range(3):
                new_outer_rot = outer_rot + (trial - 1) * step
                L, new_rotations = evaluate_arrangement(positions, new_outer_rot)
                if L < best_overall_L:
                    best_overall_L = L
                    outer_rot = new_outer_rot
                    rotations = new_rotations
                    improved = True
            if not improved:
                break
    
    if best_overall_config is None:
        # Fallback
        positions = [[2.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return positions, (0.0, 0.0), 40.0, 0.0
    
    positions, rotations, outer_rot = best_overall_config
    # Convert rotations to degrees
    inner = [[positions[i][0], positions[i][1], math.degrees(rotations[i])] for i in range(N)]
    return inner, (0.0, 0.0), best_overall_L, math.degrees(outer_rot)

def construct():
    return search()
# EVOLVE-BLOCK-END

