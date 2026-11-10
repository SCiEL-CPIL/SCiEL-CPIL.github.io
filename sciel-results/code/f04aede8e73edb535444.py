# EVOLVE-BLOCK-START
import numpy as np
import time

def _get_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(np.arange(0, 360, 60))
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack((xs, ys))

def _check_separation(hex1, hex2):
    # hex1, hex2 are (cx, cy, angle)
    # Get vertices
    v1 = _get_vertices(hex1[0], hex1[1], 1.0, hex1[2])
    v2 = _get_vertices(hex2[0], hex2[1], 1.0, hex2[2])
    
    # Get edge normals for both
    # Edge i is v[i] to v[(i+1)%6]
    # Normal is perpendicular to edge, pointing out
    # Angle of edge i is a + 60i + 30
    # Normal angle is a + 60i + 30 - 90 = a + 60i - 60
    a1 = np.deg2rad(hex1[2])
    a2 = np.deg2rad(hex2[2])
    
    normals = []
    for i in range(6):
        ang1 = a1 + np.deg2rad(60*i - 60)
        normals.append((np.cos(ang1), np.sin(ang1)))
    for i in range(6):
        ang2 = a2 + np.deg2rad(60*i - 60)
        normals.append((np.cos(ang2), np.sin(ang2)))
        
    normals = np.array(normals)
    
    # Project vertices onto normals
    # If any normal has disjoint intervals, they are separated
    for n in normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check if [min1, max1] and [min2, max2] are disjoint
        # Apart by more than 1e-6
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True # Separated
            
    return False # Not separated (intersecting or touching)

def _check_outer_constraint(inner_hexes, outer_side, outer_center, outer_angle):
    # Outer hexagon vertices
    v_outer = _get_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Check each inner vertex
    for hex in inner_hexes:
        v_inner = _get_vertices(hex[0], hex[1], 1.0, hex[2])
        for p in v_inner:
            # Check if p is inside v_outer
            # Using cross product check for each edge
            inside = True
            for i in range(6):
                p1 = v_outer[i]
                p2 = v_outer[(i+1)%6]
                # Cross product (p2 - p1) x (p - p1)
                # Should be >= -1e-6 for CCW polygon
                # v_outer is generated CCW?
                # cos/sin with increasing angle is CCW.
                cp = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cp < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    return True

def _is_valid(inner_hexes, outer_side, outer_center, outer_angle):
    # Check finite
    if not np.all(np.isfinite(inner_hexes)):
        return False
    if outer_side < np.sqrt(11):
        return False
        
    # Check pair separation
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i+1, n):
            if not _check_separation(inner_hexes[i], inner_hexes[j]):
                return False
                
    # Check outer constraint
    if not _check_outer_constraint(inner_hexes, outer_side, outer_center, outer_angle):
        return False
        
    # Check edge lengths
    for hex in inner_hexes:
        v = _get_vertices(hex[0], hex[1], 1.0, hex[2])
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1)%6]
            dist = np.linalg.norm(p2 - p1)
            if abs(dist - 1.0) > 1e-8:
                return False
    return True

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600
    
    # Initial guess
    # 1 center, 6 ring, 4 extra
    # We will try to optimize S
    best_score = 0
    best_result = None
    
    # Try a range of S
    # Lower bound sqrt(11) ~ 3.317
    # Upper bound 5.0
    
    # We'll use a simple local search for a fixed S
    # Then binary search or linear search for best S
    
    # Since we have time, let's do a few random restarts for a few S values
    
    def run_local_search(S, num_restarts=10, max_iters=500):
        best_inner = None
        best_validity = -1
        
        for r in range(num_restarts):
            # Initialize
            inner = np.zeros((11, 3))
            inner[0] = [0, 0, 0]
            # Ring 1
            for i in range(6):
                ang = np.deg2rad(60 * i)
                inner[i+1] = [np.sqrt(3)*np.cos(ang), np.sqrt(3)*np.sin(ang), 0]
            # Ring 2 (4 hexagons)
            # Place at corners of the S hexagon?
            # Let's place them at distance 2*sqrt(3) in directions 30, 90, 150, 210
            angles_extra = [30, 90, 150, 210]
            for i, ang_deg in enumerate(angles_extra):
                ang = np.deg2rad(ang_deg)
                inner[7+i] = [2*np.sqrt(3)*np.cos(ang), 2*np.sqrt(3)*np.sin(ang), 0]
                
            # Local search
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                    
                # Calculate forces
                forces = np.zeros_like(inner)
                
                # Separation
                for i in range(11):
                    for j in range(i+1, 11):
                        if not _check_separation(inner[i], inner[j]):
                            # Push apart
                            dx = inner[j][0] - inner[i][0]
                            dy = inner[j][1] - inner[i][1]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist > 1e-6:
                                fx = dx / dist
                                fy = dy / dist
                            else:
                                fx, fy = 1, 0
                                
                            forces[i] += [-fx, -fy, 0]
                            forces[j] += [fx, fy, 0]
                            
                # Outer constraint
                outer_center = np.array([0, 0])
                outer_angle = 0
                v_outer = _get_vertices(0, 0, S, 0)
                
                for i in range(11):
                    v_inner = _get_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
                    for p in v_inner:
                        # Check if outside
                        inside = True
                        for k in range(6):
                            p1 = v_outer[k]
                            p2 = v_outer[(k+1)%6]
                            cp = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                            if cp < 0:
                                inside = False
                                # Push inwards
                                # Normal to edge
                                edge_vec = p2 - p1
                                norm_vec = np.array([-edge_vec[1], edge_vec[0]])
                                norm_vec = norm_vec / np.linalg.norm(norm_vec)
                                # Ensure normal points inwards
                                # Center is (0,0). p1 is on boundary.
                                # Check dot product with vector from p1 to center
                                if np.dot(norm_vec, -p1) < 0:
                                    norm_vec = -norm_vec
                                    
                                # Add force to all vertices? Or just center?
                                # Adding to center is simpler
                                forces[i] += [norm_vec[0], norm_vec[1], 0]
                                break
                        if not inside:
                            break
                            
                # Update positions
                # Learning rate
                lr = 0.1
                inner[:, :2] += lr * forces[:, :2]
                
                # Check validity
                if _is_valid(inner, S, outer_center, outer_angle):
                    return inner, S
                    
        return None, None

    # Try S values
    for S in [3.5, 3.6, 3.7, 3.8, 4.0]:
        if time.time() > deadline - 60:
            break
        inner, side = run_local_search(S, num_restarts=5, max_iters=200)
        if inner is not None:
            # This S is valid. We can try smaller S.
            # But for now, record it.
            # To maximize 1/S, we want smallest S.
            # Let's just return the first valid one found in descending order?
            # No, we want the smallest S.
            # So we should search from low to high?
            # But low S might be hard to solve.
            # Let's try 3.4, 3.5, 3.6...
            
            # Actually, if we find a valid packing for S, it's a good candidate.
            # We can try to optimize S further.
            # For simplicity, let's just return the best found so far.
            # Since we iterate S ascending, the first valid one is the best.
            
            outer = np.array([0, 0, 0])
            return inner, outer, S

    # Fallback to seed
    inner_hex_data = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ]
    )
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
