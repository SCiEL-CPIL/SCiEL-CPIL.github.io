"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import linprog

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # For unit regular hexagons, the width (distance between opposite sides) is sqrt(3).
    # The width across corners (distance between opposite vertices) is 2.
    
    # Let's start with a simple grid arrangement and then optimize.
    # A regular hexagon of side length S has:
    # - Width across flats: S * sqrt(3)
    # - Width across corners: 2 * S
    
    # For N unit hexagons, we need to pack them.
    # Let's try a few known good configurations and optimize their positions.
    
    # First, let's define the vertices of a unit hexagon with given angle
    def hexagon_vertices(angle_deg=0.0):
        angle = np.radians(angle_deg)
        # Unit hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
        angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + angle
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Check if two hexagons overlap
    def hexagons_overlap(v1, v2):
        """Check if two convex polygons overlap using separating axis theorem."""
        vertices = [v1, v2]
        for i, poly in enumerate(vertices):
            for j in range(len(poly)):
                # Get edge normal
                p1 = poly[j]
                p2 = poly[(j + 1) % len(poly)]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                normal = normal / np.linalg.norm(normal)
                
                # Project both polygons onto this normal
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # Check for separation
                if max1 <= min2 or max2 <= min1:
                    return False
        
        return True
    
    # Try a simple horizontal line arrangement first
    # For N hexagons in a line, the minimum outer hexagon side length
    # depends on the arrangement.
    
    # Let's try to use a grid-like arrangement.
    # For regular hexagons, a honeycomb-like arrangement is efficient.
    
    # Start with a simple configuration: all hexagons in a row
    # Each unit hexagon has width sqrt(3) across flats, 2 across corners.
    # If we align them in a row with 0-degree rotation, the width along x is 2 (across corners)
    # but if we rotate by 30 degrees, the width along x is sqrt(3).
    
    # Let's try different arrangements and find the minimum outer hexagon side.
    
    def compute_outer_hexagon_side(positions, angles, center=(0.0, 0.0)):
        """Compute the minimum side length of a regular hexagon containing all inner hexagons."""
        # Get all vertices of all inner hexagons
        all_vertices = []
        for pos, ang in zip(positions, angles):
            verts = hexagon_vertices(ang) + np.array(pos)
            all_vertices.append(verts)
        
        # We need to find the minimum side length S of a regular hexagon
        # (with given orientation) that contains all these points.
        
        # For a regular hexagon with side S and orientation theta, centered at (cx, cy):
        # The support function in direction phi is S * (cos(phi - theta) + cos(phi - theta + 60°) + ...)
        # Actually, for a regular hexagon with side S, the distance from center to a side (apothem) is S * sqrt(3)/2
        # The distance from center to a vertex is S.
        
        # A point (x, y) is inside a regular hexagon centered at (cx, cy) with side S and orientation theta
        # if for all 6 directions, the projection is within bounds.
        
        # The 6 outward normals of a regular hexagon with orientation theta are at angles theta, theta+60, ..., theta+300
        # The distance from center to each side is S * sqrt(3)/2 (apothem)
        
        # For a point to be inside: |dot(point - center, normal_i)| <= S * sqrt(3)/2 for all i
        
        # So S >= 2/sqrt(3) * max_i |dot(point - center, normal_i)| for all points
        
        # We want to minimize S over center and orientation.
        
        # For simplicity, let's fix the orientation of the outer hexagon and optimize center.
        # Then we can try a few orientations.
        
        best_S = float('inf')
        
        # Try different orientations for the outer hexagon
        for outer_angle in [0, 15, 30, 45, 60]:
            outer_angle_rad = np.radians(outer_angle)
            # The 6 outward normals
            normals = []
            for k in range(6):
                ang = outer_angle_rad + k * np.pi / 3
                normals.append(np.array([np.cos(ang), np.sin(ang)]))
            
            # For a given center (cx, cy), the constraint for each point p is:
            # |dot(p - center, normal_i)| <= S * sqrt(3)/2 for all i
            # This means: -S*sqrt(3)/2 <= dot(p - center, normal_i) <= S*sqrt(3)/2
            
            # Let's use linear programming to find the minimum S.
            # Variables: cx, cy, S
            # Constraints: For each point p and each normal n:
            # dot(p, n) - dot(center, n) <= S * sqrt(3)/2
            # -dot(p, n) + dot(center, n) <= S * sqrt(3)/2
            
            # This is: dot(p, n) - n_x*cx - n_y*cy - sqrt(3)/2*S <= 0
            # and: -dot(p, n) + n_x*cx + n_y*cy - sqrt(3)/2*S <= 0
            
            # Let's collect all points
            points = np.vstack(all_vertices)
            
            n_points = len(points)
            n_normals = 6
            
            # Variables: [cx, cy, S]
            # Objective: minimize S -> c = [0, 0, 1]
            c = [0.0, 0.0, 1.0]
            
            A_ub = []
            b_ub = []
            
            for p in points:
                for n in normals:
                    # dot(p, n) - n_x*cx - n_y*cy - sqrt(3)/2*S <= 0
                    A_ub.append([n[0], n[1], -np.sqrt(3)/2])
                    b_ub.append(np.dot(p, n))
                    
                    # -dot(p, n) + n_x*cx + n_y*cy - sqrt(3)/2*S <= 0
                    A_ub.append([-n[0], -n[1], -np.sqrt(3)/2])
                    b_ub.append(-np.dot(p, n))
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            # S >= 0
            bounds = [(None, None), (None, None), (0, None)]
            
            try:
                result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if result.success:
                    S = result.x[2]
                    if S < best_S:
                        best_S = S
            except Exception:
                pass
        
        return best_S
    
    # Now let's try different arrangements of the N hexagons
    
    best_result = None
    best_score = float('inf')
    
    # Try arrangement 1: All in a horizontal line
    for line_angle in [0, 30, 60, 90]:
        positions = []
        angles = []
        for k in range(-(N // 2), N - N // 2):
            # Place hexagons along a line
            # If line_angle is 0, place along x-axis
            # The spacing should be such that hexagons don't overlap
            # For unit hexagons with 0-degree rotation, the width along x is 2 (across corners)
            # But if we want them to be tangent, the center-to-center distance should be sqrt(3) if aligned properly
            # Actually, for two unit hexagons with 0-degree rotation to be tangent horizontally,
            # the distance between centers should be 2 (since the width across corners is 2)
            # But if we rotate them, the required distance changes.
            
            # Let's use a spacing that ensures no overlap.
            # For safety, let's use spacing = 2.0 (which is the maximum width of a unit hexagon)
            # This guarantees no overlap but might not be optimal.
            
            # Better: compute the minimum distance for non-overlap based on orientation.
            # For simplicity, let's use spacing = sqrt(3) + epsilon for now, and adjust.
            
            # Actually, let's just try a few spacings.
            for spacing in [1.8, 1.9, 2.0, 2.1]:
                pos = np.array([k * spacing, 0.0])
                # Rotate the position by line_angle
                theta = np.radians(line_angle)
                pos_rot = np.array([pos[0] * np.cos(theta) - pos[1] * np.sin(theta),
                                    pos[0] * np.sin(theta) + pos[1] * np.cos(theta)])
                positions.append(pos_rot)
                angles.append(0.0)  # Inner hexagons at 0 degrees
        
        # Center the arrangement
        if positions:
            mean_pos = np.mean(positions, axis=0)
            positions = [p - mean_pos for p in positions]
        
        score = compute_outer_hexagon_side(positions, angles)
        if score < best_score:
            best_score = score
            best_result = (positions, angles, mean_pos if positions else (0.0, 0.0), score)
        
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        if best_result is not None:
            positions, angles, center, S = best_result
            inner = [[pos[0], pos[1], ang] for pos, ang in zip(positions, angles)]
            return inner, (center[0], center[1]), S, 0.0
        else:
            # Fallback
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0
    
    # Try arrangement 2: Honeycomb-like grid
    # For a honeycomb, rows are offset.
    row_height = np.sqrt(3)  # Vertical distance between rows in honeycomb
    for row_spacing in [row_height * 0.9, row_height, row_height * 1.1]:
        for col_spacing in [1.8, 1.9, 2.0, 2.1]:
            positions = []
            angles = []
            # Determine number of rows
            # For N hexagons, use roughly sqrt(N) rows
            n_rows = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
            n_cols = int(np.ceil(N / n_rows))
            
            count = 0
            for r in range(n_rows):
                for c in range(n_cols):
                    if count >= N:
                        break
                    x = c * col_spacing
                    if r % 2 == 1:
                        x += col_spacing / 2
                    y = r * row_spacing
                    positions.append(np.array([x, y]))
                    angles.append(0.0)
                    count += 1
            
            if positions:
                mean_pos = np.mean(positions, axis=0)
                positions = [p - mean_pos for p in positions]
            
            score = compute_outer_hexagon_side(positions, angles)
            if score < best_score:
                best_score = score
                best_result = (positions, angles, mean_pos if positions else (0.0, 0.0), score)
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if best_result is not None:
        positions, angles, center, S = best_result
        inner = [[pos[0], pos[1], ang] for pos, ang in zip(positions, angles)]
        return inner, (center[0], center[1]), S, 0.0
    else:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
