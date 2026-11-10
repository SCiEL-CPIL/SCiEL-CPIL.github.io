# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, linprog

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _sat_overlap(v1, v2, tol=1e-6):
    for poly in [v1, v2]:
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n_len = np.linalg.norm(n)
            if n_len < 1e-12:
                continue
            n = n / n_len
            proj1 = v1 @ n
            proj2 = v2 @ n
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 - tol or max2 < min1 - tol:
                return False
    return True

def _min_enclosing_hex(points, angle_deg):
    """Find minimum side length s and center (cx,cy) of regular hexagon with given angle containing all points.
    Returns (s, cx, cy) or None if infeasible."""
    a = np.deg2rad(angle_deg)
    # Hexagon vertices: cx + s*cos(a+60k), cy + s*sin(a+60k)
    # Edge i: from vertex i to vertex i+1
    # For point p to be inside: cross(v_{i+1} - v_i, p - v_i) >= 0 for all i
    
    # Let's set up LP: variables x = [cx, cy, s]
    # For each point p and each edge i:
    # v_i = (cx + s*cos(a_i), cy + s*sin(a_i)) where a_i = a + 60*i
    # v_{i+1} = (cx + s*cos(a_{i+1}), cy + s*sin(a_{i+1}))
    # edge = v_{i+1} - v_i = (s*(cos(a_{i+1}) - cos(a_i)), s*(sin(a_{i+1}) - sin(a_i)))
    # p - v_i = (px - cx - s*cos(a_i), py - cy - s*sin(a_i))
    # cross = edge_x * (py - cy - s*sin(a_i)) - edge_y * (px - cx - s*cos(a_i)) >= 0
    
    # This is linear in (cx, cy, s)!
    # Let ca_i = cos(a_i), sa_i = sin(a_i)
    # ca_{i+1} = cos(a_{i+1}), sa_{i+1} = sin(a_{i+1})
    # edge_x = s*(ca_{i+1} - ca_i), edge_y = s*(sa_{i+1} - sa_i)
    # cross = s*(ca_{i+1}-ca_i)*(py - cy - s*sa_i) - s*(sa_{i+1}-sa_i)*(px - cx - s*ca_i)
    # = s*(ca_{i+1}-ca_i)*py - s*(ca_{i+1}-ca_i)*cy - s^2*(ca_{i+1}-ca_i)*sa_i
    #   - s*(sa_{i+1}-sa_i)*px + s*(sa_{i+1}-sa_i)*cx + s^2*(sa_{i+1}-sa_i)*ca_i
    # This has s^2 terms, so it's NOT linear in s!
    
    # So LP won't work directly. Let me use a different approach.
    # For fixed s, the constraints are linear in (cx, cy).
    # So I can binary search on s and for each s, solve LP for (cx, cy).
    
    # Actually, let me just use a simpler approach: for given points and angle,
    # the minimum enclosing hexagon can be found by optimization.
    
    # Let me use a grid search + refinement approach
    # Or just use scipy minimize
    
    from scipy.optimize import minimize
    
    def objective(params):
        cx, cy, s = params
        if s < 0.1:
            return 1e10
        verts = _hex_vertices(cx, cy, s, angle_deg)
        # Check all points inside
        max_violation = 0
        for p in points:
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i + 1) % 6]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < 0:
                    max_violation = max(max_violation, -cross)
        return s + 100 * max_violation
    
    # Initial guess
    px_min, px_max = points[:, 0].min(), points[:, 0].max()
    py_min, py_max = points[:, 1].min(), points[:, 1].max()
    cx0 = (px_min + px_max) / 2
    cy0 = (py_min + py_max) / 2
    s0 = max(px_max - px_min, py_max - py_min) / 2 + 1
    
    res = minimize(objective, [cx0, cy0, s0], method='Nelder-Mead', 
                   options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7})
    
    cx, cy, s = res.x
    if s < 0.1:
        return None
    
    # Verify
    verts = _hex_vertices(cx, cy, s, angle_deg)
    for p in points:
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
            if cross < -1e-5:
                return None
    
    return (s, cx, cy)

def _evaluate_full(inner_positions, outer_params):
    """Evaluate full packing. outer_params = (cx, cy, angle_deg, s)"""
    cx, cy, angle_deg, s = outer_params
    
    # Inner hexagon vertices
    inner_verts = []
    for i in range(12):
        px, py = inner_positions[i]
        verts = _hex_vertices(px, py, 1.0, 0.0)
        inner_verts.append(verts)
    
    # Outer hexagon vertices
    outer_verts = _hex_vertices(cx, cy, s, angle_deg)
    
    # Check containment
    for verts in inner_verts:
        for v in verts:
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return (s + 1000, False)
    
    # Check overlap
    for i in range(12):
        for j in range(i + 1, 12):
            if _sat_overlap(inner_verts[i], inner_verts[j]):
                return (s + 1000, False)
    
    return (s, True)

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600
    
    sqrt3 = np.sqrt(3)
    
    # Initial arrangement
    initial_positions = np.array([
        [-2.0, 0.0],
        [0.0, 0.0],
        [2.0, 0.0],
        [-3.0, sqrt3],
        [-1.0, sqrt3],
        [1.0, sqrt3],
        [3.0, sqrt3],
        [-2.0, 2*sqrt3],
        [0.0, 2*sqrt3],
        [2.0, 2*sqrt3],
        [-1.0, 3*sqrt3],
        [1.0, 3*sqrt3],
    ])
    
    # Find optimal outer hexagon for initial positions
    all_vertices = []
    for pos in initial_positions:
        verts = _hex_vertices(pos[0], pos[1], 1.0, 0.0)
        all_vertices.extend(verts.tolist())
    all_vertices = np.array(all_vertices)
    
    outer_result = _min_enclosing_hex(all_vertices, 0.0)
    if outer_result:
        s0, cx0, cy0 = outer_result
        initial_outer = (cx0, cy0, 0.0, s0)
    else:
        initial_outer = (0.0, 0.0, 0.0, 5.0)
    
    obj0, valid0 = _evaluate_full(initial_positions, initial_outer)
    
    best_positions = initial_positions.copy()
    best_outer = initial_outer
    best_s = obj0 if valid0 else 1e10
    
    # Differential evolution over inner positions only
    # Outer hexagon will be optimized separately for each candidate
    
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            return 1e10
        
        positions = params.reshape(12, 2)
        
        # Get all inner vertices
        all_verts = []
        for pos in positions:
            verts = _hex_vertices(pos[0], pos[1], 1.0, 0.0)
            all_verts.extend(verts.tolist())
        all_verts = np.array(all_verts)
        
        # Find min enclosing hexagon (angle 0)
        outer_res = _min_enclosing_hex(all_verts, 0.0)
        if outer_res is None:
            return 1e10
        s, cx, cy = outer_res
        outer = (cx, cy, 0.0, s)
        
        # Check validity
        obj, valid = _evaluate_full(positions, outer)
        return obj
    
    bounds = [(-5.0, 5.0)] * 24
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        try:
            result = differential_evolution(
                objective,
                bounds,
                x0=initial_positions.flatten(),
                maxiter=100,
                popsize=12,
                tol=1e-4,
                seed=42,
                polish=False,
                maxfev=3000
            )
            
            if result.fun < best_s:
                positions = result.x.reshape(12, 2)
                all_verts = []
                for pos in positions:
                    verts = _hex_vertices(pos[0], pos[1], 1.0, 0.0)
                    all_verts.extend(verts.tolist())
                all_verts = np.array(all_verts)
                outer_res = _min_enclosing_hex(all_verts, 0.0)
                if outer_res:
                    s, cx, cy = outer_res
                    outer = (cx, cy, 0.0, s)
                    obj, valid = _evaluate_full(positions, outer)
                    if valid:
                        best_positions = positions.copy()
                        best_outer = outer
                        best_s = obj
        except Exception:
            pass
    
    # Local refinement
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        from scipy.optimize import minimize
        try:
            res = minimize(objective, best_positions.flatten(), method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7})
            if res.fun < best_s:
                positions = res.x.reshape(12, 2)
                all_verts = []
                for pos in positions:
                    verts = _hex_vertices(pos[0], pos[1], 1.0, 0.0)
                    all_verts.extend(verts.tolist())
                all_verts = np.array(all_verts)
                outer_res = _min_enclosing_hex(all_verts, 0.0)
                if outer_res:
                    s, cx, cy = outer_res
                    outer = (cx, cy, 0.0, s)
                    obj, valid = _evaluate_full(positions, outer)
                    if valid:
                        best_positions = positions.copy()
                        best_outer = outer
                        best_s = obj
        except Exception:
            pass
    
    cx, cy, angle, s = best_outer
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_positions[i, 0]
        inner_hex_data[i, 1] = best_positions[i, 1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([cx, cy, angle])
    
    return inner_hex_data, outer_hex_data, s

# EVOLVE-BLOCK-END
