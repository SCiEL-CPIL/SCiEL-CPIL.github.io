import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

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
    normals_angles = a + np.radians(30) + np.radians(60) * np.arange(6)
    return np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])

def _check_separation(hex1, hex2, tol=1e-6):
    """Check if two hexagons are separated using SAT."""
    v1 = _hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
    v2 = _hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])
    
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
    
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    inner_verts_list = []
    for i in range(12):
        verts = _hex_vertices(inner_data[i,0], inner_data[i,1], 1.0, inner_data[i,2])
        inner_verts_list.append(verts)
        
        for j in range(6):
            p1 = verts[j]
            p2 = verts[(j+1) % 6]
            if abs(np.linalg.norm(p2 - p1) - 1.0) > 1e-8:
                return False
        
        for v in verts:
            if not _vertex_inside_outer(v, outer_verts):
                return False
    
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
    outer_normals = _hex_edge_normals(0.0)
    
    if inner_angles is None:
        inner_angles = np.zeros(len(centers))
    
    max_required_s = 0.0
    for i, center in enumerate(centers):
        inner_angle = inner_angles[i]
        inner_verts = _hex_vertices(0, 0, 1.0, inner_angle)
        for n in outer_normals:
            support = np.max(inner_verts @ n)
            required_s = (center @ n + support) / (np.sqrt(3) / 2)
            if required_s > max_required_s:
                max_required_s = required_s
    
    return max_required_s

def _honeycomb_positions():
    """Generate initial honeycomb positions for 12 hexagons."""
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
    deadline = start_time + 610  # safety margin before 630s kill
    
    sqrt3 = np.sqrt(3)
    
    # Initial placement
    inner_data = _honeycomb_positions()
    centers = inner_data[:, :2]
    inner_angles = np.zeros(12)
    
    s = _min_enclosing_hex_side(centers, inner_angles)
    s *= 1.0005
    
    outer_data = np.array([0.0, 0.0, 0.0])
    
    if not _validate(inner_data, outer_data, s):
        s = 6.0
        outer_data = np.array([0.0, 0.0, 0.0])
    
    best_value = 1.0 / s
    best_result = (inner_data.copy(), outer_data.copy(), s)
    
    def objective(params):
        """Objective function: minimize s + penalty for constraint violations."""
        inner_xy = params[:36].reshape(12, 3)
        s_val = params[36]
        
        if s_val < np.sqrt(12):
            return 1e6
        
        outer_normals = _hex_edge_normals(0.0)
        
        penalty = 0.0
        
        verts_list = []
        for i in range(12):
            verts_list.append(_hex_vertices(inner_xy[i,0], inner_xy[i,1], 1.0, inner_xy[i,2]))
        
        # Check each inner hexagon is inside outer
        apothem = s_val * sqrt3 / 2
        for i in range(12):
            verts = verts_list[i]
            for n in outer_normals:
                max_proj = np.max(verts @ n)
                if max_proj > apothem + 1e-9:
                    penalty += (max_proj - apothem) ** 2
        
        # Check separation
        for i in range(12):
            verts1 = verts_list[i]
            n1 = _hex_edge_normals(inner_xy[i,2])
            for j in range(i+1, 12):
                verts2 = verts_list[j]
                n2 = _hex_edge_normals(inner_xy[j,2])
                
                separated = False
                min_gap = np.inf
                for normals in [n1, n2]:
                    for normal in normals:
                        proj1 = verts1 @ normal
                        proj2 = verts2 @ normal
                        min1, max1 = proj1.min(), proj1.max()
                        min2, max2 = proj2.min(), proj2.max()
                        
                        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                            separated = True
                            break
                        
                        gap = min(min2 - max1, min1 - max2)
                        if gap < min_gap:
                            min_gap = gap
                    if separated:
                        break
                
                if not separated and min_gap < 0:
                    penalty += min_gap ** 2
        
        return s_val + 1000 * penalty
    
    # Phase 1: Differential evolution for global search
    if time.time() < deadline - 120:
        # Define bounds for each parameter
        # x, y in [-4, 4], angle in [-180, 180], s in [sqrt(12), 8]
        bounds = [(-4, 4), (-4, 4), (-180, 180)] * 12 + [(np.sqrt(12), 8)]
        
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            
            if de_result.fun < 100:
                final_params = de_result.x
                inner_final = final_params[:36].reshape(12, 3)
                s_final = final_params[36]
                
                if s_final > np.sqrt(12):
                    inner_final[:, 2] = np.mod(inner_final[:, 2] + 180, 360) - 180
                    outer_final = np.array([0.0, 0.0, 0.0])
                    
                    s_exact = _min_enclosing_hex_side(inner_final[:, :2], inner_final[:, 2])
                    s_final = s_exact * 1.0001
                    
                    if _validate(inner_final, outer_final, s_final):
                        value = 1.0 / s_final
                        if value > best_value:
                            best_value = value
                            best_result = (inner_final, outer_final, s_final)
        except Exception:
            pass
    
    # Phase 2: Local optimization with Nelder-Mead from multiple starting points
    n_restarts = 15
    for restart in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            params0 = np.zeros(37)
            params0[:36] = best_result[0].flatten()
            params0[36] = best_result[2]
        elif restart == 1:
            # Use the DE result if available
            params0 = np.zeros(37)
            params0[:36] = best_result[0].flatten()
            params0[36] = best_result[2]
        else:
            # Perturb from best
            perturbation = 0.2 * np.random.randn(36)
            perturbation[2::3] = np.mod(perturbation[2::3] + 180, 360) - 180
            params0 = best_result[0].flatten().copy()
            params0 += perturbation
            params0 = np.concatenate([params0, [best_result[2] + 0.2 * np.random.randn()]])
        
        try:
            result = minimize(
                objective,
                params0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if result.fun < 50:
                final_params = result.x
                inner_final = final_params[:36].reshape(12, 3)
                s_final = final_params[36]
                
                if s_final > np.sqrt(12):
                    inner_final[:, 2] = np.mod(inner_final[:, 2] + 180, 360) - 180
                    outer_final = np.array([0.0, 0.0, 0.0])
                    
                    s_exact = _min_enclosing_hex_side(inner_final[:, :2], inner_final[:, 2])
                    s_final = s_exact * 1.0001
                    
                    if _validate(inner_final, outer_final, s_final):
                        value = 1.0 / s_final
                        if value > best_value:
                            best_value = value
                            best_result = (inner_final, outer_final, s_final)
        except Exception:
            pass
    
    # Phase 3: High-precision polish
    if time.time() < deadline - 30:
        polish_params = np.zeros(37)
        polish_params[:36] = best_result[0].flatten()
        polish_params[36] = best_result[2]
        
        try:
            polish_result = minimize(
                objective,
                polish_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            if polish_result.fun < 10:
                final_params = polish_result.x
                inner_final = final_params[:36].reshape(12, 3)
                s_final = final_params[36]
                
                if s_final > np.sqrt(12):
                    inner_final[:, 2] = np.mod(inner_final[:, 2] + 180, 360) - 180
                    outer_final = np.array([0.0, 0.0, 0.0])
                    
                    s_exact = _min_enclosing_hex_side(inner_final[:, :2], inner_final[:, 2])
                    s_final = s_exact * 1.00005
                    
                    if _validate(inner_final, outer_final, s_final):
                        value = 1.0 / s_final
                        if value > best_value:
                            best_value = value
                            best_result = (inner_final, outer_final, s_final)
        except Exception:
            pass
    
    # Phase 4: Final micro-polish if time permits
    if time.time() < deadline - 15:
        micro_params = np.zeros(37)
        micro_params[:36] = best_result[0].flatten()
        micro_params[36] = best_result[2]
        
        try:
            micro_result = minimize(
                objective,
                micro_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 20000,
                    'xatol': 1e-12,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if micro_result.fun < 5:
                final_params = micro_result.x
                inner_final = final_params[:36].reshape(12, 3)
                s_final = final_params[36]
                
                if s_final > np.sqrt(12):
                    inner_final[:, 2] = np.mod(inner_final[:, 2] + 180, 360) - 180
                    outer_final = np.array([0.0, 0.0, 0.0])
                    
                    s_exact = _min_enclosing_hex_side(inner_final[:, :2], inner_final[:, 2])
                    s_final = s_exact * 1.00002
                    
                    if _validate(inner_final, outer_final, s_final):
                        value = 1.0 / s_final
                        if value > best_value:
                            best_value = value
                            best_result = (inner_final, outer_final, s_final)
        except Exception:
            pass
    
    return best_result
