# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _hex_edge_normals(cx, cy, s, a_deg):
    """Return 12 edge normals (6 edges × 2 directions) and their projected intervals."""
    verts = _hex_vertices(cx, cy, s, a_deg)
    normals = []
    intervals = []
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        edge = p2 - p1
        # Perpendicular (normal)
        n = np.array([-edge[1], edge[0]])
        n = n / (np.linalg.norm(n) + 1e-15)
        # Project both endpoints onto normal
        proj1 = np.dot(p1, n)
        proj2 = np.dot(p2, n)
        lo = min(proj1, proj2)
        hi = max(proj1, proj2)
        normals.append(n)
        intervals.append((lo, hi))
        # Also add opposite direction
        normals.append(-n)
        intervals.append((-hi, -lo))
    return np.array(normals), np.array(intervals)


def _hexagons_separated(cx1, cy1, a1, cx2, cy2, a2, s=1.0, tol=1e-6):
    """Check if two unit hexagons are separated (non-intersecting)."""
    n1, i1 = _hex_edge_normals(cx1, cy1, s, a1)
    n2, i2 = _hex_edge_normals(cx2, cy2, s, a2)
    all_normals = np.vstack([n1, n2])
    all_intervals1 = np.vstack([i1, i2])  # For hexagon 1, use its own normals
    # Actually, we need to project both hexagons onto each normal
    # For each of the 12 normals, check if projections are disjoint
    for j in range(12):
        n = all_normals[j]
        # Project hexagon 1
        v1 = _hex_vertices(cx1, cy1, s, a1)
        proj1 = v1 @ n
        lo1, hi1 = proj1.min(), proj1.max()
        # Project hexagon 2
        v2 = _hex_vertices(cx2, cy2, s, a2)
        proj2 = v2 @ n
        lo2, hi2 = proj2.min(), proj2.max()
        # Check if intervals are disjoint with gap > tol
        if hi1 + tol < lo2 or hi2 + tol < lo1:
            return True
    return False


def _hexagon_in_outer(cx, cy, a_deg, outer_cx, outer_cy, outer_s, outer_a_deg, tol=1e-6):
    """Check if inner hexagon is inside outer hexagon."""
    v_inner = _hex_vertices(cx, cy, 1.0, a_deg)
    v_outer = _hex_vertices(outer_cx, outer_cy, outer_s, outer_a_deg)
    # For each edge of outer hexagon, check all inner vertices are on the inside
    for i in range(6):
        p1 = v_outer[i]
        p2 = v_outer[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        n = n / (np.linalg.norm(n) + 1e-15)
        # The interior is on the side where center is
        center_proj = np.dot(outer_cy - p1[1], n[1]) + np.dot(outer_cx - p1[0], n[0])
        # Actually, let's compute cross product for each inner vertex
        for v in v_inner:
            cross = np.cross(p2 - p1, v - p1)
            # All should be on same side (same sign as cross for center)
            center_cross = np.cross(p2 - p1, np.array([outer_cx, outer_cy]) - p1)
            if center_cross * cross < -tol:
                return False
    return True


def _objective(x, outer_s):
    """
    Objective: maximize the minimum separation between all pairs of inner hexagons,
    subject to all being inside the outer hexagon.
    x: flattened array of 11*(x, y, angle_deg) = 33 values
    """
    n = 11
    centers = x[:2*n].reshape(n, 2)
    angles = x[2*n:]
    
    # Check all pairs are separated
    min_sep = np.inf
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1 = centers[i]
            cx2, cy2 = centers[j]
            a1, a2 = angles[i], angles[j]
            # Use center distance as a proxy for separation (for same orientation, min dist = 2*sqrt(3))
            d = np.linalg.norm(centers[i] - centers[j])
            min_sep = min(min_sep, d)
    
    # Check containment in outer hexagon (centered at origin, side=outer_s, angle=0)
    # For a regular hexagon centered at origin with side S and angle 0:
    # Vertices at (S*cos(60k), S*sin(60k)) for k=0..5
    # The constraint is that all vertices of each inner hexagon must be inside
    
    # Simple check: distance from origin to any vertex of inner hexagon <= S (circumradius)
    # But that's not sufficient for a hexagon; we need edge constraints
    # For now, use a soft constraint via distance to edges
    
    # Compute max "violation" of containment
    max_violation = 0.0
    v_outer = _hex_vertices(0, 0, outer_s, 0)
    for i in range(n):
        v_inner = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for j in range(6):
            p1 = v_outer[j]
            p2 = v_outer[(j + 1) % 6]
            edge = p2 - p1
            n_edge = np.array([-edge[1], edge[0]])
            n_edge = n_edge / (np.linalg.norm(n_edge) + 1e-15)
            # Project inner vertices onto edge normal
            for v in v_inner:
                cross = np.cross(edge, v - p1)
                center_cross = np.cross(edge, np.array([0.0, 0.0]) - p1)
                if center_cross > 0:
                    # Interior is positive cross side
                    if cross < 0:
                        max_violation = max(max_violation, -cross / (np.linalg.norm(edge) + 1e-15))
                else:
                    if cross > 0:
                        max_violation = max(max_violation, cross / (np.linalg.norm(edge) + 1e-15))
    
    # Objective: minimize max_violation and penalize small separations
    # We want min_sep >= 2*sqrt(3) and max_violation <= 0
    penalty = 0.0
    if min_sep < 2 * np.sqrt(3):
        penalty += (2 * np.sqrt(3) - min_sep) ** 2 * 100
    if max_violation > 0:
        penalty += max_violation ** 2 * 100
    
    return penalty


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    n = 11
    min_center_dist = 2 * np.sqrt(3)  # Minimum center distance for non-intersecting unit hexagons
    
    # Initial guess: hexagonal lattice arrangement
    # 11 hexagons in a compact arrangement
    # Try: center + 6 around + 4 more in gaps
    # Hexagonal lattice spacing: 2*sqrt(3) between centers
    
    # Standard hex grid positions for 11 hexagons:
    # Row 0: 3 hexagons
    # Row 1: 5 hexagons  
    # Row 2: 3 hexagons
    # This gives a hexagonal shape of 11
    
    dx = 2 * np.sqrt(3)  # horizontal spacing
    dy = 3.0  # vertical spacing (dx * sin(60) = 2*sqrt(3)*sqrt(3)/2 = 3)
    
    positions = []
    # Row -1 (top): 3 hexagons
    for k in range(3):
        x = (k - 1) * dx
        y = -dy
        positions.append([x, y])
    # Row 0 (middle): 5 hexagons
    for k in range(5):
        x = (k - 2) * dx
        y = 0
        positions.append([x, y])
    # Row 1 (bottom): 3 hexagons
    for k in range(3):
        x = (k - 1) * dx
        y = dy
        positions.append([x, y])
    
    positions = np.array(positions)
    angles = np.zeros(n)
    
    # Center the arrangement
    center = positions.mean(axis=0)
    positions -= center
    
    # Initial outer hexagon side length
    # The bounding hexagon needs to contain all vertices
    # Max distance from origin to any vertex
    max_r = 0
    for i in range(n):
        v = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, 0)
        max_r = max(max_r, np.max(np.linalg.norm(v, axis=1)))
    outer_s = max_r + 0.5  # Add margin
    
    # Optimize: minimize outer_s by adjusting positions
    # Use scipy optimize
    
    best_result = None
    best_outer_s = outer_s
    
    # Try optimization with different initial outer_s
    for attempt in range(3):
        if time.time() > deadline - 60:
            break
        
        # Set up optimization
        x0 = np.concatenate([positions.flatten(), angles])
        
        def obj_func(x):
            if time.time() > deadline - 30:
                return 1e10
            n_local = 11
            centers = x[:2*n_local].reshape(n_local, 2)
            angles_local = x[2*n_local:]
            
            # Compute required outer hexagon side
            # For a regular hexagon centered at origin with angle 0:
            # A point (px, py) is inside if it satisfies 6 linear inequalities
            # The side length S is determined by the maximum "support" in each of 6 directions
            
            # For hexagon with vertices at (S*cos(60k), S*sin(60k)):
            # Edge k has normal direction at angle (60k + 90) degrees... 
            # Actually, let's compute it differently.
            
            # The hexagon can be defined as: for each of 6 edge normals n_k,
            # the support function h(n_k) = max over all vertices of (v · n_k)
            # must be <= S * (support of unit hexagon in direction n_k)
            
            # For a regular hexagon of side S, the support in direction theta is:
            # h(theta) = S * max_k cos(theta - 60k) for k=0..5... no
            # Actually for a regular hexagon with circumradius S (vertex distance S):
            # The inradius (distance to edge) is S * cos(30) = S * sqrt(3)/2
            # Support function: h(theta) = S * max(cos(theta - 60k) for k=0..5)
            
            # So the constraint is: for each inner vertex v, and each edge direction,
            # v · n_k <= S * h_unit(n_k)
            
            # Let's compute the minimum S needed
            # The 6 edge normals of a hexagon with angle 0 are at angles 90, 150, 210, 270, 330, 30 degrees
            # i.e., 30 + 60*k for k=0..5
            
            min_s = 0.0
            for i in range(n_local):
                v_inner = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles_local[i])
                for v in v_inner:
                    # For each of 6 edge directions
                    for k in range(6):
                        theta = np.radians(30 + 60 * k)
                        n_dir = np.array([np.cos(theta), np.sin(theta)])
                        proj = np.dot(v, n_dir)
                        # The support of unit hexagon in this direction
                        # For unit hexagon (side 1), circumradius = 1, inradius = sqrt(3)/2
                        # Support in direction theta: max over vertices of (vertex · n_dir)
                        # Vertices of unit hex at angles 0, 60, ..., 300
                        max_proj_unit = max(np.cos(np.radians(60 * m) - theta) for m in range(6))
                        # Actually: vertex at angle phi has dot with n_dir = cos(phi - theta)
                        # Max over phi = 0, 60, ..., 300 of cos(phi - theta)
                        # This equals cos(min_k |theta - 60k| mod 360)
                        # Which is cos(min over k of |theta - 60k|) where we take the acute angle
                        # The max is cos(delta) where delta is the minimum angle from theta to any multiple of 60
                        # delta is in [0, 30]
                        if max_proj_unit > 1e-10:
                            required_s = proj / max_proj_unit
                            if required_s > min_s:
                                min_s = required_s
            
            # Also check pairwise separation
            sep_penalty = 0.0
            for i in range(n_local):
                for j in range(i+1, n_local):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d < min_center_dist:
                        sep_penalty += (min_center_dist - d) ** 2 * 1000
            
            return min_s + sep_penalty
        
        try:
            res = minimize(obj_func, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            
            if res.fun < best_outer_s and res.fun < 1e8:
                best_outer_s = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    # Use best result or initial
    if best_result is not None:
        x_final = best_result
    else:
        x_final = np.concatenate([positions.flatten(), angles])
    
    n_local = 11
    centers = x_final[:2*n_local].reshape(n_local, 2)
    angles_final = x_final[2*n_local:]
    
    # Compute actual outer hexagon side
    min_s = 0.0
    for i in range(n_local):
        v_inner = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles_final[i])
        for v in v_inner:
            for k in range(6):
                theta = np.radians(30 + 60 * k)
                n_dir = np.array([np.cos(theta), np.sin(theta)])
                proj = np.dot(v, n_dir)
                max_proj_unit = max(np.cos(np.radians(60 * m) - theta) for m in range(6))
                if max_proj_unit > 1e-10:
                    required_s = proj / max_proj_unit
                    if required_s > min_s:
                        min_s = required_s
    
    # Add small margin for numerical safety
    outer_s_final = min_s + 1e-5
    
    inner_hex_data = np.column_stack([centers[:, 0], centers[:, 1], angles_final])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, outer_s_final


# EVOLVE-BLOCK-END
