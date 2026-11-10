# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_enclosing_hex_side(centers, angles):
    """
    Given centers (N,2) and angles (N,) in degrees for unit hexagons,
    find the minimum side length s and optimal outer hexagon angle theta
    such that all inner hexagons fit inside the outer hexagon of side s
    centered at origin.
    
    Returns (s, theta_deg) where theta is the outer hexagon's angle.
    """
    N = len(centers)
    # For each inner hexagon, compute the 6 vertex positions relative to origin
    # For a hexagon with center c, side 1, angle a, vertices are c + cos(a+60i), sin(a+60i)
    
    # We need to find theta (outer angle) and s such that all vertices are inside
    # The outer hexagon with center at origin, side s, angle theta has vertices at
    # s*cos(theta+60i), s*sin(theta+60i)
    
    # A point p is inside the outer hexagon iff for all 6 edge normals n_i,
    # dot(p, n_i) <= s * dot(outer_vertex_0, n_i) ... actually let me think more carefully.
    
    # For a regular hexagon centered at origin with side s and angle theta:
    # The support function in direction n is h(n) = s * max_i dot(v_i, n) where v_i are unit vertices
    # Actually, the inradius (distance from center to edge midpoint) is s * sqrt(3)/2
    # The support in direction n is s * sqrt(3)/2 * max over the 6 edge normals of dot(n, e_i)
    
    # Simpler approach: for a given theta, compute the required s as the max over all
    # inner vertices of the "radial" measure.
    
    # For a regular hexagon with side s, angle theta, the boundary in direction phi is:
    # r(phi) = (s * sqrt(3)/2) / cos(phi - theta - 30k) for the appropriate sector
    
    # Let me use a different approach: the outer hexagon contains point p iff
    # for each of the 6 edges, the point is on the correct side.
    # Edge i connects vertex i to vertex i+1. The outward normal of edge i is at angle theta + 60*i + 90 degrees? 
    # Let me just compute it directly.
    
    # Outer hexagon vertices: V_i = s * (cos(theta + 60i), sin(theta + 60i)) for i=0..5
    # Edge from V_i to V_{i+1}. The inward normal (pointing toward center) is:
    # edge = V_{i+1} - V_i
    # normal_inward = rotate edge by -90 deg, normalized, then negate if pointing outward
    
    # Actually, let me just parametrize by theta and for each theta compute the minimum s.
    
    def _required_s_for_theta(theta_deg):
        theta = np.radians(theta_deg)
        # Outer hexagon unit vertices (s=1)
        unit_outer_verts = np.column_stack([np.cos(theta + np.arange(6) * np.pi/3), 
                                             np.sin(theta + np.arange(6) * np.pi/3)])
        
        # For each edge, compute the inward normal and the distance from center to edge (inradius)
        # Edge i: from unit_outer_verts[i] to unit_outer_verts[(i+1)%6]
        max_s = 0.0
        for i in range(6):
            p1 = unit_outer_verts[i]
            p2 = unit_outer_verts[(i+1) % 6]
            edge = p2 - p1
            # Inward normal: perpendicular to edge, pointing toward origin
            # Rotate edge by 90 degrees CCW: (-edge[1], edge[0])
            normal = np.array([-edge[1], edge[0]])
            # Check if this points toward origin: dot(normal, -p1) should be positive
            if np.dot(normal, -p1) < 0:
                normal = -normal
            norm = np.linalg.norm(normal)
            normal /= norm
            
            # The inradius (distance from center to edge) for unit hexagon
            inradius = np.dot(p1, normal)  # should be sqrt(3)/2
            
            # For all inner vertices, we need dot(vertex, normal) <= s * inradius
            # i.e., s >= dot(vertex, normal) / inradius
            for k in range(N):
                # Compute all 6 vertices of inner hexagon k
                a_k = np.radians(angles[k])
                c_k = centers[k]
                for j in range(6):
                    vj = c_k + np.array([np.cos(a_k + j * np.pi/3), np.sin(a_k + j * np.pi/3)])
                    s_req = np.dot(vj, normal) / inradius
                    if s_req > max_s:
                        max_s = s_req
        
        return max_s
    
    # Search over theta in [0, 60) degrees (periodicity of hexagon)
    # Use fine grid + local optimization
    best_s = np.inf
    best_theta = 0
    
    # Coarse grid
    theta_vals = np.linspace(0, 60, 121)
    for t in theta_vals:
        s_val = _required_s_for_theta(t)
        if s_val < best_s:
            best_s = s_val
            best_theta = t
    
    # Local refinement around best
    def _obj(theta_arr):
        return _required_s_for_theta(theta_arr[0])
    
    result = minimize(_obj, [best_theta], method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
    if result.fun < best_s:
        best_s = result.fun
        best_theta = result.x[0]
    
    # Ensure theta is in [0, 60)
    best_theta = best_theta % 60
    
    return best_s, best_theta

def _check_valid(inner, outer, s, tol=1e-6):
    """Check if the packing is valid."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    
    # Check outer vertices distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-6:
                return False
    
    # Shift by outer center
    shift = np.array([outer[0], outer[1]])
    
    # Check inner edges
    inner_verts_list = []
    for i in range(12):
        x, y, a = inner[i]
        verts = _hex_vertices(x, y, 1.0, a)
        inner_verts_list.append(verts)
        for k in range(6):
            d = np.linalg.norm(verts[(k+1)%6] - verts[k])
            if abs(d - 1.0) > 1e-8:
                return False
    
    # Check no overlap (SAT)
    for i in range(12):
        for j in range(i+1, 12):
            v1 = inner_verts_list[i]
            v2 = inner_verts_list[j]
            separated = False
            for poly in [v1, v2]:
                for k in range(6):
                    p1 = poly[k]
                    p2 = poly[(k+1)%6]
                    edge = p2 - p1
                    normal = np.array([edge[1], -edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-12:
                        continue
                    normal /= norm
                    proj1 = np.dot(v1, normal)
                    proj2 = np.dot(v2, normal)
                    if proj1.max() < proj2.min() - 1e-8 or proj2.max() < proj1.min() - 1e-8:
                        separated = True
                        break
                if separated:
                    break
            if not separated:
                return False
    
    # Check containment
    shifted_outer_verts = outer_verts - shift
    for i in range(12):
        for v in inner_verts_list[i]:
            v_shifted = v - shift
            inside = True
            for k in range(6):
                p1 = shifted_outer_verts[k]
                p2 = shifted_outer_verts[(k+1)%6]
                cross = (p2[0]-p1[0])*(v_shifted[1]-p1[1]) - (p2[1]-p1[1])*(v_shifted[0]-p1[0])
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True

def _honeycomb_343():
    """3-4-3 honeycomb arrangement, all angles 0."""
    dy = 1.5 * np.sqrt(3)
    inner = []
    for i in range(3):
        x = (i - 1) * 2.0
        inner.append([x, 0, 0])
    for i in range(4):
        x = (i - 1.5) * 2.0
        inner.append([x, dy, 0])
    for i in range(3):
        x = (i - 1) * 2.0
        inner.append([x, 2*dy, 0])
    return np.array(inner)

def _honeycomb_2442():
    """2-4-4-2 arrangement."""
    dy = 1.5 * np.sqrt(3)
    inner = []
    for i in range(2):
        x = (i - 0.5) * 2.0
        inner.append([x, 0, 0])
    for i in range(4):
        x = (i - 1.5) * 2.0
        inner.append([x, dy, 0])
    for i in range(4):
        x = (i - 1.5) * 2.0
        inner.append([x, 2*dy, 0])
    for i in range(2):
        x = (i - 0.5) * 2.0
        inner.append([x, 3*dy, 0])
    return np.array(inner)

def _circular_arrangement():
    """1 center + 6 ring + 5 outer."""
    inner = [[0, 0, 0]]
    for i in range(6):
        a = np.radians(60 * i)
        x = 2.0 * np.cos(a)
        y = 2.0 * np.sin(a)
        inner.append([x, y, 0])
    for i in range(5):
        a = np.radians(72 * i + 30)
        x = 3.86 * np.cos(a)
        y = 3.86 * np.sin(a)
        inner.append([x, y, 0])
    return np.array(inner)

def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon, minimizing outer side length.
    Returns (inner, outer, s) where inner is (12,3), outer is (3,), s is float.
    """
    start_time = time.time()
    deadline = start_time + 610  # 20s safety margin
    
    best_s = np.inf
    best_inner = None
    best_outer = None
    
    # Strategy 1: Rigid honeycomb cluster - optimize global rotation and outer hex angle
    # For a fixed topology, we can try different rotations of the whole cluster
    initial_configs = [_honeycomb_343(), _honeycomb_2442(), _circular_arrangement()]
    
    for config_idx, inner_init in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        # Center the configuration
        centers = inner_init[:, :2].copy()
        angles = inner_init[:, 2].copy()
        cx = np.mean(centers[:, 0])
        cy = np.mean(centers[:, 1])
        centers -= np.array([cx, cy])
        
        # Try to find best outer hex by varying rotation of cluster and outer hex angle
        # The key insight: rotate the entire cluster and find best enclosing hex
        
        best_local_s = np.inf
        best_local_rotation = 0
        best_local_outer_angle = 0
        
        # Coarse search over cluster rotation (0 to 60 degrees)
        for rot_deg in np.linspace(0, 60, 61):
            if time.time() > deadline:
                break
            rot = np.radians(rot_deg)
            R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
            rotated_centers = centers @ R.T
            
            # Find best enclosing hex for this rotated configuration
            s_val, outer_angle = _min_enclosing_hex_side(rotated_centers, angles)
            if s_val < best_local_s:
                best_local_s = s_val
                best_local_rotation = rot_deg
                best_local_outer_angle = outer_angle
        
        # Local refinement of rotation
        def _rot_obj(params):
            rot = np.radians(params[0])
            R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
            rotated_centers = centers @ R.T
            s_val, _ = _min_enclosing_hex_side(rotated_centers, angles)
            return s_val
        
        if time.time() < deadline:
            res = minimize(_rot_obj, [best_local_rotation], method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_local_s:
                best_local_s = res.fun
                best_local_rotation = res.x[0] % 60
        
        # Compute final configuration
        rot = np.radians(best_local_rotation)
        R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        final_centers = centers @ R.T
        s_val, outer_angle = _min_enclosing_hex_side(final_centers, angles)
        
        # Build result
        inner_result = np.column_stack([final_centers, angles])
        outer_result = np.array([0.0, 0.0, outer_angle])
        
        # Validate
        if _check_valid(inner_result, outer_result, s_val):
            if s_val < best_s:
                best_s = s_val
                best_inner = inner_result.copy()
                best_outer = outer_result.copy()
    
    # Strategy 2: Full optimization with Nelder-Mead on all parameters
    if time.time() < deadline - 30:
        def _full_cost(params):
            s = params[0]
            ox, oy, oa = params[1], params[2], params[3]
            inner_data = params[4:].reshape(12, 3)
            
            outer_verts = _hex_vertices(ox, oy, s, oa)
            penalty = 0.0
            
            # Outer vertices distinct
            for i in range(6):
                for j in range(i+1, 6):
                    if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-6:
                        penalty += 1000.0
            
            inner_verts_list = []
            for i in range(12):
                x, y, a = inner_data[i]
                verts = _hex_vertices(x, y, 1.0, a)
                inner_verts_list.append(verts)
            
            # No overlap
            for i in range(12):
                for j in range(i+1, 12):
                    v1, v2 = inner_verts_list[i], inner_verts_list[j]
                    overlap = True
                    for poly in [v1, v2]:
                        for k in range(6):
                            p1, p2 = poly[k], poly[(k+1)%6]
                            edge = p2 - p1
                            normal = np.array([edge[1], -edge[0]])
                            norm = np.linalg.norm(normal)
                            if norm < 1e-12:
                                continue
                            normal /= norm
                            proj1 = np.dot(v1, normal)
                            proj2 = np.dot(v2, normal)
                            if proj1.max() < proj2.min() - 1e-8 or proj2.max() < proj1.min() - 1e-8:
                                overlap = False
                                break
                        if not overlap:
                            break
                    if overlap:
                        penalty += 10.0
            
            # Containment
            for i in range(12):
                for v in inner_verts_list[i]:
                    inside = True
                    for k in range(6):
                        p1 = outer_verts[k]
                        p2 = outer_verts[(k+1)%6]
                        cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                        if cross < -1e-6:
                            inside = False
                            break
                    if not inside:
                        penalty += 10.0
            
            return s + penalty
        
        # Use best result as initial guess
        if best_inner is not None:
            init_inner = best_inner.copy()
            init_outer = best_outer.copy()
            init_s = best_s
        else:
            init_inner = _honeycomb_343()
            init_outer = np.array([0, 0, 0])
            init_s = 5.0
        
        params_init = np.concatenate([[init_s, init_outer[0], init_outer[1], init_outer[2]], 
                                       init_inner.flatten()])
        
        # Multiple restarts
        n_restarts = 5
        for r in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            if r == 0:
                params = params_init.copy()
            else:
                params = params_init + np.random.normal(0, 0.05, size=params_init.shape)
            
            try:
                result = minimize(_full_cost, params, method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            except Exception:
                continue
            
            s_try = result.x[0]
            ox, oy, oa = result.x[1], result.x[2], result.x[3]
            inner_try = result.x[4:].reshape(12, 3)
            outer_try = np.array([ox, oy, oa])
            
            if _check_valid(inner_try, outer_try, s_try):
                if s_try < best_s:
                    best_s = s_try
                    best_inner = inner_try.copy()
                    best_outer = outer_try.copy()
    
    # Fallback
    if best_inner is None:
        inner = _honeycomb_343()
        outer = np.array([0, 0, 0])
        s = 8.0
        return inner, outer, s
    
    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END
