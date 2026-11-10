# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side s and angle a_deg."""
    a = math.radians(a_deg)
    verts = []
    for i in range(6):
        ang = a + math.radians(60 * i)
        verts.append((cx + s * math.cos(ang), cy + s * math.sin(ang)))
    return np.array(verts)


def _get_edge_normals(verts):
    """Return 6 outward normals (unit vectors) for a convex hexagon given its 6 vertices in order."""
    normals = np.zeros((6, 2))
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        edge = p2 - p1
        # Outward normal: rotate edge by -90 degrees (for CCW polygon)
        # For a CCW polygon, the outward normal to edge (p2-p1) is (dy, -dx) normalized?
        # Let's check. Center is (0,0). Vertex 0 at (1,0), Vertex 1 at (0.5, sqrt(3)/2).
        # Edge 0-1: (-0.5, sqrt(3)/2).
        # Outward normal should point away from center.
        # Midpoint of edge 0-1 is (0.75, sqrt(3)/4).
        # Vector from center to midpoint is (0.75, sqrt(3)/4).
        # Rotate edge (-0.5, 0.866) by -90 deg: (0.866, 0.5). This points outward.
        # So normal = (edge[1], -edge[0]).
        n = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-12:
            normals[i] = n / norm
        else:
            normals[i] = np.array([1.0, 0.0])
    return normals


def _check_separation(hex1_verts, hex2_verts, hex1_normals, hex2_normals):
    """
    Check if two hexagons are separated.
    Returns True if separated (valid), False if intersecting (invalid).
    Separation condition: on at least one of the 12 edge normals, the projected intervals are apart by more than 1e-6.
    """
    eps = 1e-6
    all_normals = np.vstack([hex1_normals, hex2_normals])
    
    for n in all_normals:
        # Project both hexagons onto this normal
        proj1 = hex1_verts @ n
        proj2 = hex2_verts @ n
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check if intervals are apart
        # Case 1: hex1 is completely to the left of hex2
        if max1 + eps < min2:
            return True
        # Case 2: hex2 is completely to the left of hex1
        if max2 + eps < min1:
            return True
            
    return False


def _is_valid_packing(inner, outer, side):
    """
    Validate the packing.
    inner: (11, 3) array of (x, y, angle_deg)
    outer: (3,) array of (x, y, angle_deg)
    side: float
    """
    # 1. Check finite values
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(side):
        return False
    
    # 2. Check side >= sqrt(11)
    if side < math.sqrt(11):
        return False
        
    # 3. Compute vertices for all inner hexagons
    inner_verts = []
    inner_normals = []
    for i in range(11):
        x, y, a = inner[i]
        verts = _hex_vertices(x, y, 1.0, a)
        inner_verts.append(verts)
        inner_normals.append(_get_edge_normals(verts))
        
    # 4. Check separation between all pairs of inner hexagons
    for i in range(11):
        for j in range(i + 1, 11):
            if not _check_separation(inner_verts[i], inner_verts[j], inner_normals[i], inner_normals[j]):
                return False
                
    # 5. Check containment in outer hexagon
    ox, oy, oa = outer
    outer_verts = _hex_vertices(ox, oy, side, oa)
    
    # Shift coordinates by outer center for cross product check?
    # "Coordinates are first shifted by the outer center"
    # This implies we should translate everything so outer center is (0,0).
    # But the containment check is: every inner vertex p satisfies cross(p2 - p1, p - p1) >= -1e-6 for each outer edge.
    # This is a standard point-in-convex-polygon test.
    
    # Let's translate everything by -outer_center to make outer center (0,0)
    # This matches the "shifted by outer center" description.
    shift = np.array([ox, oy])
    
    # Outer vertices shifted
    outer_verts_shifted = outer_verts - shift
    # Inner vertices shifted
    inner_verts_shifted = [v - shift for v in inner_verts]
    
    # Check each inner vertex against outer edges
    for iv in inner_verts_shifted:
        for k in range(6):
            p1 = outer_verts_shifted[k]
            p2 = outer_verts_shifted[(k + 1) % 6]
            # cross(p2 - p1, p - p1)
            v1 = p2 - p1
            v2 = iv - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if cross < -1e-6:
                return False
                
    # 6. Check edge lengths of inner hexagons (after shift? No, shift doesn't change length)
    # "shifted inner hexagons must keep edge lengths within 1e-8 of 1"
    # This is just a numerical stability check.
    for iv in inner_verts:
        for k in range(6):
            p1 = iv[k]
            p2 = iv[(k + 1) % 6]
            length = np.linalg.norm(p2 - p1)
            if abs(length - 1.0) > 1e-8:
                return False

    return True


def _objective(inner, outer, side):
    """
    Calculate the objective value. Lower side is better.
    We want to minimize side.
    """
    if not _is_valid_packing(inner, outer, side):
        return float('inf')
    return side


def hexagon_packing_11():
    start_time = time.time()
    deadline = 630.0
    safety_margin = 10.0

    # Initial packing: 4-3-4 honeycomb grid, scaled slightly to ensure separation
    h = math.sqrt(3) / 2
    coords = []
    for i in range(4):
        coords.append([i * 3.0, 0.0])
    for i in range(3):
        coords.append([1.5 + i * 3.0, 2 * h])
    for i in range(4):
        coords.append([i * 3.0, 4 * h])

    # Center the configuration
    cx = np.mean([c[0] for c in coords])
    cy = np.mean([c[1] for c in coords])
    for c in coords:
        c[0] -= cx
        c[1] -= cy

    # Scale to create gaps > 1e-6
    # The gap in a perfect honeycomb is 0.
    # Scaling by (1 + delta) increases the distance between centers by (1+delta).
    # The "gap" between edges is approximately delta * (distance between centers) / 2?
    # Actually, for two touching hexagons, the distance between centers is 2 * apothem = sqrt(3).
    # If we scale coordinates by (1+delta), the new distance is sqrt(3)*(1+delta).
    # The sum of apothems is still sqrt(3).
    # So the gap is sqrt(3)*delta.
    # We need gap > 1e-6.
    # sqrt(3)*delta > 1e-6 => delta > 1e-6 / 1.732 ~ 5.77e-7.
    # Let's use delta = 1e-4 to be safe and allow for optimization.
    scale = 1.0 + 1e-4
    for c in coords:
        c[0] *= scale
        c[1] *= scale

    inner = np.zeros((11, 3))
    for i, c in enumerate(coords):
        inner[i, 0] = c[0]
        inner[i, 1] = c[1]
        inner[i, 2] = 0.0

    outer = np.array([0.0, 0.0, 0.0])

    # Helper to compute minimal side for given inner and outer angle
    def compute_min_side(inner_data, outer_angle_deg):
        outer_angle_rad = math.radians(outer_angle_deg)
        # Normals of outer hexagon
        # Vertices at outer_angle + 60k
        # Normals at outer_angle + 30 + 60k
        normal_angles = [outer_angle_deg + 30 + 60 * k for k in range(6)]
        normals = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in normal_angles])
        
        max_proj = 0.0
        base_angles = np.radians(60 * np.arange(6))
        base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
        
        for i in range(11):
            x, y, a = inner_data[i]
            # Rotate base vertices by angle a
            ca = math.cos(math.radians(a))
            sa = math.sin(math.radians(a))
            rot = np.array([[ca, -sa], [sa, ca]])
            verts = base_verts @ rot.T + np.array([x, y])
            
            projs = verts @ normals.T
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
                
        side = 2.0 * max_proj / math.sqrt(3)
        return side

    # 1. Optimize outer angle
    best_side = float('inf')
    best_outer_angle = 0.0
    
    # Sample angles 0 to 30
    for outer_angle_deg in np.arange(0, 30.01, 0.5):
        if time.time() - start_time > deadline - safety_margin:
            break
        side = compute_min_side(inner, outer_angle_deg)
        if side < best_side:
            best_side = side
            best_outer_angle = outer_angle_deg

    outer[2] = best_outer_angle
    current_side = best_side

    # 2. Local optimization of inner positions
    # We want to minimize the side.
    # Side is determined by the max projection of inner vertices onto outer normals.
    # We can try to move inner hexagons to reduce this max projection.
    # This is a non-convex problem, but local search might help.
    
    # Gradient-free local search (Nelder-Mead style or simple random perturbations)
    # Let's try simple random perturbations with cooling.
    
    best_inner = inner.copy()
    best_side_val = current_side
    best_outer_ang = best_outer_angle
    
    # Precompute base vertices for speed
    base_angles = np.radians(60 * np.arange(6))
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    def fast_side(inner_data, outer_angle_deg):
        outer_angle_rad = math.radians(outer_angle_deg)
        normal_angles = [outer_angle_deg + 30 + 60 * k for k in range(6)]
        normals = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in normal_angles])
        
        max_proj = 0.0
        for i in range(11):
            x, y, a = inner_data[i]
            ca = math.cos(math.radians(a))
            sa = math.sin(math.radians(a))
            rot = np.array([[ca, -sa], [sa, ca]])
            verts = base_verts @ rot.T + np.array([x, y])
            projs = verts @ normals.T
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        return 2.0 * max_proj / math.sqrt(3)

    # Optimization loop
    step_size = 0.1
    temp = 1.0
    
    while time.time() - start_time < deadline - safety_margin:
        # Try a random perturbation
        candidate_inner = best_inner.copy()
        idx = np.random.randint(0, 11)
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        candidate_inner[idx, 0] += dx
        candidate_inner[idx, 1] += dy
        
        # Also try perturbing the outer angle slightly
        candidate_outer_angle = best_outer_ang + np.random.normal(0, 0.5)
        
        cand_side = fast_side(candidate_inner, candidate_outer_angle)
        
        # Check validity (expensive, so do it only if side is better or close)
        # Actually, we must ensure validity.
        # If the perturbation is small, validity is likely preserved.
        # But we must check.
        
        # Optimization: Only check full validity if cand_side < best_side_val + small_epsilon
        # Or just check validity always? 11 hexes, 55 pairs, 12 normals each = 660 checks.
        # Plus containment 66 vertices * 6 edges = 396 checks.
        # Total ~1000 cross products. Should be fast enough in Python if vectorized?
        # Let's vectorize the validity check part.
        
        if cand_side < best_side_val + 0.01: # Only validate promising candidates
            # Reconstruct outer for validation
            cand_outer = np.array([0.0, 0.0, candidate_outer_angle])
            if _is_valid_packing(candidate_inner, cand_outer, cand_side):
                if cand_side < best_side_val:
                    best_side_val = cand_side
                    best_inner = candidate_inner
                    best_outer_ang = candidate_outer_angle
                    temp *= 0.95 # Cool down
                else:
                    temp *= 0.99
        else:
            temp *= 0.99
            
        # Anneal step size
        step_size = max(0.001, step_size * 0.99)
        
        # Periodic restart or larger jumps?
        # For now, keep it simple.

    # Final result
    outer[2] = best_outer_ang
    return best_inner, outer, best_side_val

# EVOLVE-BLOCK-END
