# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _hex_edge_normals(vertices):
    """Return unit edge normals for a hexagon (6, 2) array."""
    edges = np.roll(vertices, -1, axis=0) - vertices
    normals = np.column_stack([-edges[:, 1], edges[:, 0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / norms
    center = vertices.mean(axis=0)
    mids = (vertices + np.roll(vertices, -1, axis=0)) / 2
    dots = np.sum(normals * (center - mids), axis=1)
    normals[dots < 0] *= -1
    return normals

def _sat_overlap_penalty(v1, v2, normals1, normals2, tol=1e-6):
    """Compute smooth penalty for overlap between two hexagons using SAT."""
    all_normals = np.vstack([normals1, normals2])
    penalty = 0.0
    for n in all_normals:
        p1 = v1 @ n
        p2 = v2 @ n
        min1, max1 = p1.min(), p1.max()
        min2, max2 = p2.min(), p2.max()
        gap = min(max1, max2) - max(min1, min2)
        if gap < 0:
            # Smooth penalty: quadratic in overlap
            penalty += gap * gap
    return penalty

def _point_outside_hex_penalty(points, outer_verts):
    """Compute penalty for points outside the outer hexagon."""
    penalty = 0.0
    n = len(outer_verts)
    for i in range(n):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % n]
        edge = p2 - p1
        # Normal pointing inward
        normal = np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)
        # Check if center is on the correct side
        center = np.mean(outer_verts, axis=0)
        if np.dot(normal, center - p1) < 0:
            normal = -normal
        # Distance from each point to the edge line
        dists = np.dot(points - p1, normal)
        # Points with dist < 0 are outside
        outside = dists < 0
        if np.any(outside):
            penalty += np.sum(dists[outside] ** 2)
    return penalty

def _required_s(vertices, angle_outer=0.0):
    """
    Calculate the minimum side length s of a regular hexagon (centered at origin, 
    with angle angle_outer) that contains all points in vertices (N, 2).
    """
    # Rotate vertices to align with axis-aligned hexagon
    a = np.radians(-angle_outer)
    cos_a, sin_a = np.cos(a), np.sin(a)
    x_rot = vertices[:, 0] * cos_a - vertices[:, 1] * sin_a
    y_rot = vertices[:, 0] * sin_a + vertices[:, 1] * cos_a
    
    # Constraints for axis-aligned hexagon with side s:
    # 1. |x| <= s
    # 2. |sqrt(3)*x + y| <= sqrt(3)*s
    # 3. |sqrt(3)*x - y| <= sqrt(3)*s
    sqrt3 = np.sqrt(3)
    s1 = np.max(np.abs(x_rot))
    s2 = np.max(np.abs(sqrt3 * x_rot + y_rot)) / sqrt3
    s3 = np.max(np.abs(sqrt3 * x_rot - y_rot)) / sqrt3
    
    return max(s1, s2, s3)

def _objective(x, angle_inner=0.0, angle_outer=0.0):
    """
    Objective function for optimization.
    x = [cx1, cy1, cx2, cy2, ..., cx12, cy12, s]
    Returns: s + penalty_terms
    """
    centers = x[:24].reshape(12, 2)
    s = x[24]
    
    # Compute all inner hexagon vertices
    all_vertices = []
    inner_verts = []
    inner_normals = []
    for i in range(12):
        v = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angle_inner)
        inner_verts.append(v)
        inner_normals.append(_hex_edge_normals(v))
        all_vertices.append(v)
    
    all_vertices = np.vstack(all_vertices)
    
    # Outer hexagon vertices
    outer_verts = _hex_vertices(0, 0, s, angle_outer)
    
    # Penalty for points outside outer hexagon
    pen_outer = _point_outside_hex_penalty(all_vertices, outer_verts)
    
    # Penalty for overlaps between inner hexagons
    pen_overlap = 0.0
    for i in range(12):
        for j in range(i + 1, 12):
            dist_sq = np.sum((centers[i] - centers[j]) ** 2)
            if dist_sq > 9.0:
                continue
            pen_overlap += _sat_overlap_penalty(inner_verts[i], inner_verts[j], 
                                                 inner_normals[i], inner_normals[j])
    
    return s + 10 * pen_outer + 10 * pen_overlap

def _is_feasible(centers, s, angle_inner=0.0, angle_outer=0.0, tol=1e-6):
    """Check if the configuration is feasible."""
    # Compute all inner hexagon vertices
    all_vertices = []
    inner_verts = []
    inner_normals = []
    for i in range(12):
        v = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angle_inner)
        inner_verts.append(v)
        inner_normals.append(_hex_edge_normals(v))
        all_vertices.append(v)
    
    all_vertices = np.vstack(all_vertices)
    
    # Check containment: all inner vertices must be inside outer hexagon
    outer_verts = _hex_vertices(0, 0, s, angle_outer)
    n = len(outer_verts)
    for i in range(n):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % n]
        edge = p2 - p1
        for v in all_vertices:
            cross = edge[0] * (v[1] - p1[1]) - edge[1] * (v[0] - p1[0])
            # For a convex polygon, all cross products should have the same sign
            # We need to check consistency
            # Actually, let's use the dot product method
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)
            center = np.mean(outer_verts, axis=0)
            if np.dot(normal, center - p1) < 0:
                normal = -normal
            dist = np.dot(v - p1, normal)
            if dist < -tol:
                return False
    
    # Check separation between inner hexagons
    for i in range(12):
        for j in range(i + 1, 12):
            dist_sq = np.sum((centers[i] - centers[j]) ** 2)
            if dist_sq > 9.0:
                continue
            all_normals = np.vstack([inner_normals[i], inner_normals[j]])
            for n_vec in all_normals:
                p1 = inner_verts[i] @ n_vec
                p2 = inner_verts[j] @ n_vec
                min1, max1 = p1.min(), p1.max()
                min2, max2 = p2.min(), p2.max()
                gap = min(max1, max2) - max(min1, min2)
                if gap < -tol:
                    return False
    
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = 620.0  # Safety margin
    
    best_score = 0.0
    best_result = None
    best_s = float('inf')
    
    angle_inner = 0.0
    angle_outer = 0.0
    sqrt3 = np.sqrt(3)
    
    # Multiple initial configurations to try
    initial_configs = []
    
    # Config 1: 13-hex cluster minus one (original)
    init1 = np.zeros((12, 2))
    init1[0] = [0, 0]
    for i in range(6):
        angle = np.radians(60 * i)
        init1[i+1] = [sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)]
    for i in range(5):
        angle = np.radians(30 + 60 * i)
        init1[i+7] = [3 * np.cos(angle), 3 * np.sin(angle)]
    initial_configs.append(init1)
    
    # Config 2: Different selection from second ring
    init2 = np.zeros((12, 2))
    init2[0] = [0, 0]
    for i in range(6):
        angle = np.radians(60 * i)
        init2[i+1] = [sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)]
    for i in range(5):
        angle = np.radians(0 + 60 * i)
        init2[i+7] = [3 * np.cos(angle), 3 * np.sin(angle)]
    initial_configs.append(init2)
    
    # Config 3: Symmetric arrangement
    init3 = np.zeros((12, 2))
    init3[0] = [0, 0]
    for i in range(6):
        angle = np.radians(60 * i)
        init3[i+1] = [sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)]
    for i in range(5):
        angle = np.radians(60 * i + 30)
        init3[i+7] = [3 * np.cos(angle), 3 * np.sin(angle)]
    initial_configs.append(init3)
    
    for init_centers in initial_configs:
        if time.time() > deadline:
            break
        
        # Compute initial s
        all_vertices = []
        for i in range(12):
            v = _hex_vertices(init_centers[i, 0], init_centers[i, 1], 1.0, angle_inner)
            all_vertices.append(v)
        all_vertices = np.vstack(all_vertices)
        s_init = _required_s(all_vertices, angle_outer) + 0.1  # Small buffer
        
        # Initial x for optimization
        x0 = np.zeros(25)
        x0[:24] = init_centers.flatten()
        x0[24] = s_init
        
        # Try L-BFGS-B optimization
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-12},
                             args=(angle_inner, angle_outer))
            
            if result.success or result.fun < float('inf'):
                x_opt = result.x
                centers_opt = x_opt[:24].reshape(12, 2)
                s_opt = x_opt[24]
                
                # Check feasibility
                if _is_feasible(centers_opt, s_opt, angle_inner, angle_outer):
                    score = 1.0 / s_opt
                    if score > best_score:
                        best_score = score
                        best_s = s_opt
                        inner_data = np.column_stack([centers_opt, np.full(12, angle_inner)])
                        outer_data = np.array([0, 0, angle_outer])
                        best_result = (inner_data, outer_data, s_opt)
        except Exception:
            pass
        
        # Also try simulated annealing from this initial config
        if time.time() > deadline:
            break
        
        current_centers = init_centers.copy()
        T = 0.5
        cooling_rate = 0.999
        T_end = 1e-8
        max_iter = 50000
        
        # Initial evaluation
        all_vertices = []
        for i in range(12):
            v = _hex_vertices(current_centers[i, 0], current_centers[i, 1], 1.0, angle_inner)
            all_vertices.append(v)
        all_vertices = np.vstack(all_vertices)
        curr_s = _required_s(all_vertices, angle_outer)
        
        best_centers_sa = current_centers.copy()
        best_s_sa = curr_s
        
        for iter_count in range(max_iter):
            if time.time() > deadline:
                break
            
            # Perturb
            idx = random.randint(0, 11)
            dx = random.gauss(0, 0.3)
            dy = random.gauss(0, 0.3)
            
            new_centers = current_centers.copy()
            new_centers[idx] = [current_centers[idx][0] + dx, current_centers[idx][1] + dy]
            
            # Compute new s
            all_vertices = []
            for i in range(12):
                v = _hex_vertices(new_centers[i, 0], new_centers[i, 1], 1.0, angle_inner)
                all_vertices.append(v)
            all_vertices = np.vstack(all_vertices)
            new_s = _required_s(all_vertices, angle_outer)
            
            # Compute overlap penalty
            pen = 0.0
            inner_verts = []
            inner_normals = []
            for i in range(12):
                v = _hex_vertices(new_centers[i, 0], new_centers[i, 1], 1.0, angle_inner)
                inner_verts.append(v)
                inner_normals.append(_hex_edge_normals(v))
            
            for i in range(12):
                for j in range(i + 1, 12):
                    dist_sq = np.sum((new_centers[i] - new_centers[j]) ** 2)
                    if dist_sq > 9.0:
                        continue
                    pen += _sat_overlap_penalty(inner_verts[i], inner_verts[j],
                                                inner_normals[i], inner_normals[j])
            
            curr_obj = curr_s + 10 * 0  # No outer penalty in SA
            new_obj = new_s + 10 * pen
            
            delta = new_obj - curr_obj
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_centers = new_centers
                curr_s = new_s
                
                if new_s < best_s_sa:
                    best_s_sa = new_s
                    best_centers_sa = new_centers.copy()
            
            T *= cooling_rate
        
        # Check SA result
        if _is_feasible(best_centers_sa, best_s_sa, angle_inner, angle_outer):
            score = 1.0 / best_s_sa
            if score > best_score:
                best_score = score
                best_s = best_s_sa
                inner_data = np.column_stack([best_centers_sa, np.full(12, angle_inner)])
                outer_data = np.array([0, 0, angle_outer])
                best_result = (inner_data, outer_data, best_s)
    
    if best_result is None:
        # Fallback: simple grid arrangement
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
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    return best_result

# EVOLVE-BLOCK-END
