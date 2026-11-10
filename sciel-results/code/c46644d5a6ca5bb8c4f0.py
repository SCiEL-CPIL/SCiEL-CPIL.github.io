import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + d * np.cos(angles),
        cy + d * np.sin(angles)
    ])

def _check_separation(hex1, hex2, tol=1e-6):
    """Check if two hexagons are separated using SAT."""
    v1 = _hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
    v2 = _hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])
    
    for verts in [v1, v2]:
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i+1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-15)
            
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

def _min_enclosing_hex_side(centers, circumradius=1.0):
    """
    Find the minimum side length s of a regular hexagon (centered at origin, angle 0)
    that contains all points at distance circumradius from each center.
    """
    # For a flat-top hexagon (angle 0) centered at origin with side s:
    # The apothem (distance from center to edge midpoint) is s*sqrt(3)/2
    # The vertices are at distance s from center
    # 
    # A point (x, y) is inside the hexagon if:
    # |x| <= s (for the left/right edges) -- actually this is for pointy-top
    # For flat-top (vertices at 0, 60, 120, 180, 240, 300 degrees):
    # The six edges have normals at 30, 90, 150, 210, 270, 330 degrees
    # Edge i has normal direction n_i = (cos(30+60i), sin(30+60i))
    # The constraint is: dot(point, n_i) <= s * sqrt(3) / 2 for all i
    # Wait, let me reconsider.
    #
    # For a regular hexagon with side s, centered at origin, with one vertex at angle 0:
    # Vertices: s*(cos(k*60), sin(k*60)) for k=0..5
    # The apothem (inradius) = s * sqrt(3) / 2
    # The edges are perpendicular to directions at 30, 90, 150, 210, 270, 330 degrees
    # For each edge with outward normal n (unit vector), the constraint is:
    #   dot(point, n) <= apothem = s * sqrt(3) / 2
    #
    # So for a point p, the required s is:
    #   s >= max_i( dot(p, n_i) / (sqrt(3)/2) )
    # where n_i are the 6 outward normals.
    #
    # But we need to contain the entire hexagon (all vertices of inner hexagons),
    # not just centers. Each inner hexagon has circumradius 1, so we need to
    # consider all points at distance up to 1 from each center.
    #
    # The worst case for a given direction n is:
    #   max over centers c of (dot(c, n) + 1)  [since the inner hexagon extends 1 in all directions]
    # Wait, that's not quite right. The inner hexagon is also a hexagon, so its
    # extent in direction n depends on the angle. But for a conservative bound,
    # we can use the circumradius = 1.
    #
    # Actually, for a regular hexagon of side 1, the support function in direction n
    # is: h(n) = max over vertices v of dot(v, n)
    # For a unit hexagon centered at origin with vertex at angle 0:
    # The support function is: h(n) = 1 if n is at vertex direction, sqrt(3)/2 if n is at edge normal direction
    # More generally, for angle theta of n, h(n) = cos(theta - k*60) for the appropriate k
    #
    # For simplicity and to be safe, let's use circumradius = 1 as the extent in all directions.
    # This gives a slightly larger hexagon but is simpler.
    
    normals = []
    for k in range(6):
        angle = np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(angle), np.sin(angle)]))
    normals = np.array(normals)
    
    # For each center, the worst-case dot product with normal n is:
    #   dot(center, n) + 1  (using circumradius as upper bound on support)
    # Actually, the support function of a unit hexagon in direction n is at most 1.
    # Let's compute it properly.
    
    # For a unit hexagon with vertex at angle 0, the vertices are at:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # The support function h(n) = max(dot(v, n)) over these 6 vertices
    
    unit_verts = _hex_vertices(0, 0, 1.0, 0.0)
    
    max_required_s = 0.0
    for center in centers:
        for n in normals:
            # Support of the inner hexagon in direction n:
            support = np.max(unit_verts @ n)
            # The farthest point in direction n from origin is:
            #   dot(center, n) + support
            # This must be <= s * sqrt(3) / 2
            required_s = (center @ n + support) / (np.sqrt(3) / 2)
            if required_s > max_required_s:
                max_required_s = required_s
    
    return max_required_s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s kill
    
    sqrt3 = np.sqrt(3)
    
    # Initial placement in a honeycomb pattern
    # 4 columns, 3 rows
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
    
    inner_data = np.array(positions)
    centers = inner_data[:, :2]
    
    # Compute minimum enclosing hexagon side
    s = _min_enclosing_hex_side(centers)
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
    
    # Optimize to minimize s
    def objective(params):
        # params: [x1,y1, x2,y2, ..., x12,y12, s]
        inner_xy = params[:24].reshape(12, 2)
        s_val = params[24]
        
        if s_val < np.sqrt(12):
            return 1e6
        
        # Build inner data
        inner = np.zeros((12, 3))
        inner[:, 0] = inner_xy[:, 0]
        inner[:, 1] = inner_xy[:, 1]
        inner[:, 2] = 0.0  # all angle 0
        
        outer = np.array([0.0, 0.0, 0.0])
        
        # Check validity and compute penalty
        outer_verts = _hex_vertices(0, 0, s_val, 0)
        
        penalty = 0.0
        
        # Check each inner hexagon
        for i in range(12):
            verts = _hex_vertices(inner[i,0], inner[i,1], 1.0, 0.0)
            
            # Check inside outer
            for v in verts:
                for j in range(6):
                    p1 = outer_verts[j]
                    p2 = outer_verts[(j+1) % 6]
                    cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                    if cross < 0:
                        penalty += abs(cross)
            
            # Check separation from other hexagons
            for j in range(i+1, 12):
                verts2 = _hex_vertices(inner[j,0], inner[j,1], 1.0, 0.0)
                
                # SAT check
                separated = False
                for verts in [verts, verts2]:
                    for k in range(6):
                        p1 = verts[k]
                        p2 = verts[(k+1) % 6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-15:
                            normal = normal / norm
                            proj1 = verts @ normal
                            proj2 = verts2 @ normal
                            min1, max1 = proj1.min(), proj1.max()
                            min2, max2 = proj2.min(), proj2.max()
                            if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                                separated = True
                                break
                    if separated:
                        break
                
                if not separated:
                    # Compute overlap penalty
                    min_gap = np.inf
                    for verts in [verts, verts2]:
                        for k in range(6):
                            p1 = verts[k]
                            p2 = verts[(k+1) % 6]
                            edge = p2 - p1
                            normal = np.array([-edge[1], edge[0]])
                            norm = np.linalg.norm(normal)
                            if norm > 1e-15:
                                normal = normal / norm
                                proj1 = verts @ normal
                                proj2 = verts2 @ normal
                                min1, max1 = proj1.min(), proj1.max()
                                min2, max2 = proj2.min(), proj2.max()
                                gap = min(min2 - max1, min1 - max2)
                                if gap < 0:
                                    min_gap = min(min_gap, -gap)
                    
                    if min_gap < np.inf:
                        penalty += min_gap ** 2
        
        return s_val + 1000 * penalty
    
    # Initial params
    init_params = np.zeros(25)
    init_params[:24] = inner_data[:, :2].flatten()
    init_params[24] = s
    
    # Optimize
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
            params0[24] = s + 0.3 * np.random.randn()
        
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
                    s_exact = _min_enclosing_hex_side(inner_xy_final)
                    s_final = s_exact * 1.0001
                    
                    if _validate(inner_final, outer_final, s_final):
                        value = 1.0 / s_final
                        if value > best_value:
                            best_value = value
                            best_result = (inner_final, outer_final, s_final)
        except Exception as e:
            pass
    
    return best_result
