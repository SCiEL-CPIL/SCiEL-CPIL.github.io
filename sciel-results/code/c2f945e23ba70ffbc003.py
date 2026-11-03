import math
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _point_in_hex_dist_sq(px, py, cx, cy, s, angle_deg):
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    cos_a = math.cos(-a)
    sin_a = math.sin(-a)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    
    apothem = s * math.sqrt(3) / 2
    sum_sq_viol = 0.0
    for i in range(3):
        theta_n = math.radians(30 + i * 60)
        proj = rx * math.cos(theta_n) + ry * math.sin(theta_n)
        if proj > apothem:
            sum_sq_viol += (proj - apothem)**2
        if proj < -apothem:
            sum_sq_viol += (-apothem - proj)**2
    return sum_sq_viol

def _pairwise_clearance(p1, p2):
    c1x, c1y, a1 = p1
    c2x, c2y, a2 = p2
    
    dx = c1x - c2x
    dy = c1y - c2y
    if dx*dx + dy*dy > 4.5: 
        return 0.0
        
    v1 = _hex_vertices(c1x, c1y, 1.0, a1)
    v2 = _hex_vertices(c2x, c2y, 1.0, a2)
    
    penalty = 0.0
    for pt in v1:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c2x, c2y, 1.0, a2)
    for pt in v2:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c1x, c1y, 1.0, a1)
        
    return penalty

def _min_enclosing_hexagon_fast(inner_hexes):
    """
    Finds the minimum side length of a regular hexagon enclosing the given inner hexagons.
    Returns (cx, cy, side, angle_deg).
    """
    points = []
    for h in inner_hexes:
        verts = _hex_vertices(h[0], h[1], 1.0, h[2])
        points.extend(verts)
    
    points = np.array(points)
    if len(points) == 0:
        return 0, 0, 1.0, 0.0

    best_side = float('inf')
    best_params = (0, 0, 1.0, 0.0)
    
    # Scan angles 0-60 degrees (symmetry)
    # Step 1: Coarse scan
    angles = np.arange(0, 60, 1.0)
    
    for angle_deg in angles:
        a = math.radians(angle_deg)
        # Normals for the 3 pairs of parallel sides
        # The sides of a regular hexagon are perpendicular to directions 30, 90, 150 deg relative to the hex orientation?
        # If vertices are at 0, 60..., sides are at 30, 90...
        # So normals are at 30, 90, 150.
        normals = []
        for k in range(3):
            theta_n = a + math.pi/6 + k * math.pi/3
            normals.append((math.cos(theta_n), math.sin(theta_n)))
        
        A = []
        b = []
        
        for nx, ny in normals:
            for px, py in points:
                A.append([nx, ny, -math.sqrt(3)/2])
                b.append(nx*px + ny*py)
                A.append([-nx, -ny, -math.sqrt(3)/2])
                b.append(-nx*px - ny*py)
        
        bounds = [(-100, 100), (-100, 100), (0, None)]
        
        try:
            res = linprog(c=[0, 0, 1], A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if res.success:
                s_val = res.x[2]
                if s_val < best_side:
                    best_side = s_val
                    best_params = (res.x[0], res.x[1], s_val, angle_deg)
        except Exception:
            pass
            
    # Refine around best angle
    if best_params[3] > 0 and best_params[3] < 60:
        best_ang = best_params[3]
        # Fine scan around the best coarse angle
        angles_fine = np.arange(best_ang - 1.0, best_ang + 1.0, 0.1)
        
        for angle_deg in angles_fine:
            a = math.radians(angle_deg)
            normals = []
            for k in range(3):
                theta_n = a + math.pi/6 + k * math.pi/3
                normals.append((math.cos(theta_n), math.sin(theta_n)))
                
            A = []
            b = []
            for nx, ny in normals:
                for px, py in points:
                    A.append([nx, ny, -math.sqrt(3)/2])
                    b.append(nx*px + ny*py)
                    A.append([-nx, -ny, -math.sqrt(3)/2])
                    b.append(-nx*px - ny*py)
                    
            bounds = [(-100, 100), (-100, 100), (0, None)]
            try:
                res = linprog(c=[0, 0, 1], A_ub=A, b_ub=b, bounds=bounds, method='highs')
                if res.success:
                    s_val = res.x[2]
                    if s_val < best_side:
                        best_side = s_val
                        best_params = (res.x[0], res.x[1], s_val, angle_deg)
            except Exception:
                pass
                
    return best_params

def _objective(params):
    """
    Objective function for optimization.
    params: flattened array of [x0, y0, a0, x1, y1, a1, ...]
    Returns: Outer side length + Penalty for overlaps.
    """
    positions = params.reshape(N, 3)
    
    # 1. Calculate Overlap Penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            pen = _pairwise_clearance(tuple(positions[i]), tuple(positions[j]))
            if pen > 0:
                penalty += pen * 100.0 # High weight to enforce disjointness
    
    # 2. Calculate Enclosing Hexagon Side Length
    # To make this faster and differentiable-ish, we can use the LP result directly.
    # However, LP is not differentiable. Nelder-Mead handles non-smooth functions well.
    # But calling LP inside every Nelder-Mead evaluation might be slow.
    # Let's estimate the side length using the maximum projection in the 6 normal directions of a "canonical" hexagon
    # and then refine the angle separately?
    
    # Actually, for the optimization phase, we can fix the outer hexagon orientation to 0 (or optimize it later)
    # and just minimize the extent in the 3 normal directions.
    # Extent in direction n is max(p.n) - min(p.n).
    # Side length S is related to extent W by S = W / (sqrt(3)/2) * ... wait.
    # If the hexagon is oriented such that normals are at 0, 60, 120 (flat top/bottom? no, pointy top?).
    # Let's stick to the LP approach but maybe make it cheaper or less frequent?
    # No, N=11 is small. LP with 66 constraints is very fast.
    
    inner_list = [tuple(p) for p in positions]
    _, _, side_len, _ = _min_enclosing_hexagon_fast(inner_list)
    
    return side_len + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial Layout: Honeycomb Cluster
    # 1 center, 6 ring, 4 outer
    rest_len = math.sqrt(3)
    
    pos = np.zeros((N, 3))
    pos[0] = [0, 0, 0]
    
    angles_neighbor = [30, 90, 150, 210, 270, 330]
    for i in range(1, 7):
        ang_rad = math.radians(angles_neighbor[i-1])
        pos[i] = [rest_len * math.cos(ang_rad), rest_len * math.sin(ang_rad), 0]
        
    for idx, neighbor_idx in enumerate([1, 2, 3, 4]):
        node_idx = 7 + idx
        base_pos = pos[neighbor_idx]
        ang_rad = math.radians(angles_neighbor[neighbor_idx-1])
        pos[node_idx] = [
            base_pos[0] + rest_len * math.cos(ang_rad),
            base_pos[1] + rest_len * math.sin(ang_rad),
            0
        ]
        
    x0 = pos.flatten()
    
    best_params = x0
    best_score = float('inf')
    
    # Optimization 1: Nelder-Mead to minimize (Side Length + Overlap Penalty)
    # This directly optimizes the target metric.
    try:
        res = minimize(_objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
        
        if res.success:
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
    except Exception:
        pass
    
    if time.time() > deadline - 20:
        # Fallback if too slow
        pass
    else:
        # Optimization 2: Refine positions only (fix angles to 0) using L-BFGS-B
        # This helps settle any slight overlaps or positional inefficiencies
        # while keeping the shape stable.
        
        def objective_xy(xy_flat):
            p = np.zeros((N, 3))
            for i in range(N):
                p[i][0] = xy_flat[2*i]
                p[i][1] = xy_flat[2*i+1]
                p[i][2] = 0.0 # Fix angle to 0 for this refinement
            return _objective(p.flatten())
            
        # Extract current best xy
        current_pos = best_params.reshape(N, 3)
        x0_xy = current_pos[:, :2].flatten()
        
        try:
            res2 = minimize(objective_xy, x0_xy, method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-12})
            if res2.success:
                if res2.fun < best_score:
                    best_score = res2.fun
                    # Update best_params with new xy and 0 angles
                    new_pos = np.zeros((N, 3))
                    new_pos[:, :2] = res2.x.reshape(N, 2)
                    best_params = new_pos.flatten()
        except Exception:
            pass
            
        # Optimization 3: Allow angles to vary slightly?
        # The honeycomb lattice usually works best with aligned hexagons.
        # But maybe slight rotations help?
        # Let's try one more Nelder-Mead pass allowing angles, starting from the refined xy.
        
        if time.time() < deadline - 10:
            try:
                res3 = minimize(_objective, best_params, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
                if res3.success:
                    if res3.fun < best_score:
                        best_score = res3.fun
                        best_params = res3.x
            except Exception:
                pass

    # Final Result Construction
    final_pos = best_params.reshape(N, 3)
    
    # Verify disjointness strictly
    # If there is any overlap, we might need to push them apart, but the penalty should have handled it.
    # Just in case, do a final check and if invalid, try a simple repulsion.
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if _pairwise_clearance(tuple(final_pos[i]), tuple(final_pos[j])) > 1e-6:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Simple repulsion repair
        for _ in range(50):
            if time.time() > deadline: break
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    pen = _pairwise_clearance(tuple(final_pos[i]), tuple(final_pos[j]))
                    if pen > 1e-6:
                        dx = final_pos[i][0] - final_pos[j][0]
                        dy = final_pos[i][1] - final_pos[j][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 1e-6:
                            dx, dy = 1.0, 0.0
                            dist = 1.0
                        force = 0.05 * pen
                        final_pos[i][0] += force * dx / dist
                        final_pos[i][1] += force * dy / dist
                        final_pos[j][0] -= force * dx / dist
                        final_pos[j][1] -= force * dy / dist
                        moved = True
            if not moved: break

    inner_list = [tuple(p) for p in final_pos]
    ocx, ocy, os, oang = _min_enclosing_hexagon_fast(inner_list)
    
    inner = []
    for p in final_pos:
        inner.append([p[0], p[1], p[2]])
        
    return inner, (ocx, ocy), os, oang

if __name__ == "__main__":
    r = construct()
    print(r)
