import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + d * np.cos(angles),
        cy + d * np.sin(angles)
    ])

def _hex_edge_normals(angle_deg):
    """Compute the 6 outward unit edge normals for a regular hexagon."""
    a = np.radians(angle_deg)
    # Edge normals are at angles 30 + 60*k + a (for k=0..5)
    # Wait, let me think about this more carefully.
    # For a hexagon with vertex at angle a, the edges are between consecutive vertices.
    # The outward normal of the edge from vertex k to vertex k+1 is perpendicular to the edge.
    # The edge from vertex at angle (a + 60k) to vertex at angle (a + 60(k+1)) 
    # has direction that is tangent to the hexagon.
    # The outward normal points in the direction of the midpoint of the arc, which is at angle a + 60k + 30.
    normals_angles = a + np.radians(30) + np.radians(60) * np.arange(6)
    return np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])

def _check_separation(hex1, hex2, tol=1e-6):
    """Check if two hexagons are separated using SAT."""
    v1 = _hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
    v2 = _hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])
    
    # Get edge normals for both hexagons
    n1 = _hex_edge_normals(hex1[2])
    n2 = _hex_edge_normals(hex2[2])
    
    for normals in [n1, n2]:
        for normal in normals:
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            if max1 + tol < min2 or max2 + tol < min1:
                return True
    
    return False

def _vertex_inside_outer(v, outer_verts):
    """Check if vertex v is inside outer hexagon."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i+1) % 6]
        cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
        if cross < -1e-6:
            return False
    return True

def _validate(inner_data, outer_data, s):
    """Validate the packing."""
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    
    # Check outer vertices distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    # Check inner hexagons
    inner_verts_list = []
    for i in range(12):
        verts = _hex_vertices(inner_data[i,0], inner_data[i,1], 1.0, inner_data[i,2])
        inner_verts_list.append(verts)
        
        # Check unit edges
        for j in range(6):
            p1 = verts[j]
            p2 = verts[(j+1) % 6]
            if abs(np.linalg.norm(p2 - p1) - 1.0) > 1e-8:
                return False
        
        # Check inside outer
        for v in verts:
            if not _vertex_inside_outer(v, outer_verts):
                return False
    
    # Check separation
    for i in range(12):
        for j in range(i+1, 12):
            if not _check_separation(inner_data[i], inner_data[j]):
                return False
    
    return True

def _min_enclosing_hex_side(centers, inner_angles=None):
    """
    Find the minimum side length s of a regular hexagon (centered at origin, angle 0)
    that contains all inner hexagons.
    """
    # Outer hexagon (angle 0) has outward normals at 30, 90, 150, 210, 270, 330 degrees
    outer_normals = _hex_edge_normals(0.0)
    
    if inner_angles is None:
        inner_angles = np.zeros(len(centers))
    
    max_required_s = 0.0
    for i, center in enumerate(centers):
        inner_angle = inner_angles[i]
        # Compute support function of inner hexagon in each outer normal direction
        inner_verts = _hex_vertices(0, 0, 1.0, inner_angle)
        for n in outer_normals:
            support = np.max(inner_verts @ n)
            # The farthest point in direction n from origin is:
            #   dot(center, n) + support
            # This must be <= s * sqrt(3) / 2
            required_s = (center @ n + support) / (np.sqrt(3) / 2)
            if required_s > max_required_s:
                max_required_s = required_s
    
    return max_required_s

def _honeycomb_positions():
    """Generate 12 hexagon centers in a 3-4-3 honeycomb arrangement."""
    sqrt3 = np.sqrt(3)
    col_spacing = 1.5
    row_spacing = sqrt3
    
    positions = []
    for col in range(4):
        for row in range(3):
            x = col * col_spacing
            if col % 2 == 0:
                y = row * row_spacing
            else:
                y = (row + 0.5) * row_spacing
            positions.append([x, y, 0.0])
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    return np.array(positions)

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s kill
    
    sqrt3 = np.sqrt(3)
    
    # Initial placement in a honeycomb pattern
    inner_data = _honeycomb_positions()
    centers = inner_data[:, :2]
    inner_angles = np.zeros(12)
    
    # Compute minimum enclosing hexagon side
    s = _min_enclosing_hex_side(centers, inner_angles)
    # Add small margin
    s *= 1.001
    
    outer_data = np.array([0.0, 0.0, 0.0])
    
    # Validate initial solution
    if not _validate(inner_data, outer_data, s):
        # Fallback: use larger s
        s = 6.0
        outer_data = np.array([0.0, 0.0, 0.0])
    
    best_value = 1.0 / s
    best_result = (inner_data.copy(), outer_data.copy(), s)
    
    # Affine transformation approach: optimize rotation angle and scale of the rigid honeycomb cluster
    def affine_objective(params):
        # params: [rotation_deg, scale]
        if len(params) < 2:
            return 1e6
        rot_deg = params[0]
        scale = params[1]
        
        if scale < 0.5 or scale > 3.0:
            return 1e6
        
        # Apply rotation and scaling to the base honeycomb positions
        theta = np.radians(rot_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        
        scaled_centers = centers * scale
        rotated_centers = (R @ scaled_centers.T).T
        
        # Check if rotated/scaled cluster fits in a hexagon of side s
        # The outer hexagon is fixed at origin with some side length
        # We need to find the minimum s such that all inner hexagons fit
        
        # Compute the required s for this configuration
        s_required = _min_enclosing_hex_side(rotated_centers, np.zeros(12))
        
        # Penalty for separation violations
        penalty = 0.0
        outer_normals = _hex_edge_normals(0.0)
        
        for i in range(12):
            verts_i = _hex_vertices(rotated_centers[i,0], rotated_centers[i,1], scale, 0.0)
            
            # Containment check
            for n in outer_normals:
                max_proj = np.max(verts_i @ n)
                apothem = s_required * sqrt3 / 2
                if max_proj > apothem + 1e-9:
                    penalty += (max_proj - apothem) ** 2
            
            # Separation check
            for j in range(i+1, 12):
                verts_j = _hex_vertices(rotated_centers[j,0], rotated_centers[j,1], scale, 0.0)
                
                n1 = _hex_edge_normals(0.0)
                n2 = _hex_edge_normals(0.0)
                
                separated = False
                min_gap = np.inf
                for normals in [n1, n2]:
                    for normal in normals:
                        proj1 = verts_i @ normal
                        proj2 = verts_j @ normal
                        min1, max1 = proj1.min(), proj1.max()
                        min2, max2 = proj2.min(), proj2.max()
                        
                        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                            separated = True
                            break
                        
                        gap = min(min2 - max1, min1 - max2)
                        if gap < min_gap:
                            min_gap = gap
                
                if not separated and min_gap < 0:
                    penalty += min_gap ** 2
        
        return s_required + 1000 * penalty
    
    # Optimize the affine transformation
    affine_bounds = [(-180, 180), (0.5, 3.0)]
    
    try:
        affine_result = differential_evolution(
            affine_objective,
            affine_bounds,
            seed=42,
            maxiter=100,
            tol=1e-8,
            polish=False,
            workers=-1
        )
        
        if affine_result.fun < 100:
            rot_opt = affine_result.x[0]
            scale_opt = affine_result.x[1]
            
            theta = np.radians(rot_opt)
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
            
            scaled_centers = centers * scale_opt
            rotated_centers = (R @ scaled_centers.T).T
            
            s_opt = _min_enclosing_hex_side(rotated_centers, np.zeros(12)) * 1.0001
            
            inner_opt = np.zeros((12, 3))
            inner_opt[:, 0] = rotated_centers[:, 0]
            inner_opt[:, 1] = rotated_centers[:, 1]
            inner_opt[:, 2] = 0.0
            outer_opt = np.array([0.0, 0.0, 0.0])
            
            if _validate(inner_opt, outer_opt, s_opt):
                value = 1.0 / s_opt
                if value > best_value:
                    best_value = value
                    best_result = (inner_opt, outer_opt, s_opt)
    except Exception as e:
        pass
    
    # Full coordinate optimization with Nelder-Mead
    def objective(params):
        # params: [x1,y1, x2,y2, ..., x12,y12, s]
        inner_xy = params[:24].reshape(12, 2)
        s_val = params[24]
        
        if s_val < np.sqrt(12):
            return 1e6
        
        # Build inner data with all angles 0
        inner = np.zeros((12, 3))
        inner[:, 0] = inner_xy[:, 0]
        inner[:, 1] = inner_xy[:, 1]
        inner[:, 2] = 0.0  # all angle 0
        
        outer = np.array([0.0, 0.0, 0.0])
        
        # Check validity and compute penalty
        outer_verts = _hex_vertices(0, 0, s_val, 0)
        outer_normals = _hex_edge_normals(0.0)
        
        penalty = 0.0
        
        # Check each inner hexagon is inside outer
        for i in range(12):
            verts = _hex_vertices(inner[i,0], inner[i,1], 1.0, 0.0)
            
            # Use support function approach for containment check
            for n in outer_normals:
                max_proj = np.max(verts @ n)
                apothem = s_val * sqrt3 / 2
                if max_proj > apothem + 1e-9:
                    penalty += (max_proj - apothem) ** 2
            
            # Check separation from other hexagons
            for j in range(i+1, 12):
                verts2 = _hex_vertices(inner[j,0], inner[j,1], 1.0, 0.0)
                
                # SAT check using edge normals
                n1 = _hex_edge_normals(0.0)
                n2 = _hex_edge_normals(0.0)
                
                separated = False
                min_gap = np.inf
                for normals in [n1, n2]:
                    for normal in normals:
                        proj1 = verts @ normal
                        proj2 = verts2 @ normal
                        min1, max1 = proj1.min(), proj1.max()
                        min2, max2 = proj2.min(), proj2.max()
                        
                        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                            separated = True
                            break
                        
                        # Track minimum gap for penalty
                        gap = min(min2 - max1, min1 - max2)
                        if gap < min_gap:
                            min_gap = gap
                
                if not separated and min_gap < 0:
                    penalty += min_gap ** 2
        
        return s_val + 1000 * penalty
    
    # Initial params from best result so far
    init_params = np.zeros(25)
    init_params[:24] = best_result[0][:, :2].flatten()
    init_params[24] = best_result[2]
    
    # Optimize with multiple restarts
    n_restarts = 15
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            params0 = init_params.copy()
        else:
            # Perturb initial guess
            perturbation = 0.3 * np.random.randn(24)
            params0 = init_params.copy()
            params0[:24] += perturbation
            params0[24] = best_result[2] + 0.3 * np.random.randn()
        
        try:
            result = minimize(
                objective,
                params0,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-7,
                    'fatol': 1e-7,
                    'adaptive': True
                }
            )
            
            if result.fun < 100:
                final_params = result.x
                inner_xy_final = final_params[:24].reshape(12, 2)
                s_final = final_params[24]
                
                if s_final > np.sqrt(12):
                    inner_final = np.zeros((12, 3))
                    inner_final[:, 0] = inner_xy_final[:, 0]
                    inner_final[:, 1] = inner_xy_final[:, 1]
                    inner_final[:, 2] = 0.0
                    outer_final = np.array([0.0, 0.0, 0.0])
                    
                    # Recompute s to be exact
                    s_exact = _min_enclosing_hex_side(inner_xy_final, inner_angles)
                    s_final = s_exact * 1.0001
                    
                    if _validate(inner_final, outer_final, s_final):
                        value = 1.0 / s_final
                        if value > best_value:
                            best_value = value
                            best_result = (inner_final, outer_final, s_final)
        except Exception as e:
            pass
    
    # Final high-precision polish pass
    if time.time() < deadline - 30:
        current_inner = best_result[0].copy()
        current_outer = best_result[1].copy()
        current_s = best_result[2]
        
        polish_params = np.zeros(25)
        polish_params[:24] = current_inner[:, :2].flatten()
        polish_params[24] = current_s
        
        try:
            polish_result = minimize(
                objective,
                polish_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-9,
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            
            if polish_result.fun < 100:
                final_params = polish_result.x
                inner_xy_final = final_params[:24].reshape(12, 2)
                s_final = final_params[24]
                
                if s_final > np.sqrt(12):
                    inner_final = np.zeros((12, 3))
                    inner_final[:, 0] = inner_xy_final[:, 0]
                    inner_final[:, 1] = inner_xy_final[:, 1]
                    inner_final[:, 2] = 0.0
                    outer_final = np.array([0.0, 0.0, 0.0])
                    
                    # Recompute s to be exact
                    s_exact = _min_enclosing_hex_side(inner_xy_final, inner_angles)
                    s_final = s_exact * 1.0001
                    
                    if _validate(inner_final, outer_final, s_final):
                        value = 1.0 / s_final
                        if value > best_value:
                            best_value = value
                            best_result = (inner_final, outer_final, s_final)
        except Exception as e:
            pass
    
    return best_result
