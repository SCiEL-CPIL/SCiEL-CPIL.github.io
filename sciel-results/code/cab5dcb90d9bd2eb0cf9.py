import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

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

def _min_enclosing_hexagon(inner_hexes):
    points = []
    for h in inner_hexes:
        verts = _hex_vertices(h[0], h[1], 1.0, h[2])
        points.extend(verts)
    
    points = np.array(points)
    if len(points) == 0:
        return 0, 0, 1.0, 0.0

    best_side = float('inf')
    best_params = (0, 0, 1.0, 0.0)
    
    # Coarse scan over outer hexagon orientation
    for angle_deg in np.arange(0, 60, 0.5):
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
            
    # Refine around best angle
    if best_params[3] > 0 and best_params[3] < 60:
        best_ang = best_params[3]
        for angle_deg in np.arange(best_ang - 0.5, best_ang + 0.5, 0.05):
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

def _energy(params, positions, edges, k_spring=500.0, k_overlap=5000.0, k_compress=50.0):
    energy = 0.0
    n = len(positions)
    
    # Spring forces
    for i, j, rest_len in edges:
        p1 = positions[i]
        p2 = positions[j]
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 1e-6:
            dist = 1e-6
        diff = dist - rest_len
        energy += k_spring * diff * diff
        
    # Compression penalty: penalize distances that are too small (overlap)
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            d_sq = dx*dx + dy*dy
            if d_sq < 4.5:  # Only check close pairs
                dist = math.sqrt(d_sq)
                min_dist = 2.0  # minimum center distance for non-overlapping unit hexes (roughly)
                if dist < min_dist:
                    energy += k_compress * (min_dist - dist)**2
    
    # Overlap penalty (strong)
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            if dx*dx + dy*dy < 9.0:
                pen = _pairwise_clearance(tuple(positions[i]), tuple(positions[j]))
                if pen > 0:
                    energy += k_overlap * pen
    return energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Define honeycomb topology
    rest_len = math.sqrt(3)
    edges = []
    # Center to ring 1
    for i in range(1, 7):
        edges.append((0, i, rest_len))
    # Ring 1 to Ring 1
    for i in range(1, 7):
        j = (i % 6) + 1
        edges.append((i, j, rest_len))
    # Outer ring attachments
    for idx, neighbor_idx in enumerate([1, 2, 3, 4]):
        node_idx = 7 + idx
        edges.append((neighbor_idx, node_idx, rest_len))
    
    # Initial positions: ideal honeycomb
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
    
    # Multi-start optimization
    best_pos = pos.copy()
    best_score = float('inf')
    
    starts = [0, 0, 30, 30, 15, 45]  # different initial rotations
    for start_angle in starts:
        if time.time() > deadline - 10:
            break
            
        trial_pos = pos.copy()
        # Rotate all inner hexes by start_angle
        for i in range(N):
            trial_pos[i][2] = start_angle
            
        def objective(xy_flat):
            p = trial_pos.copy()
            for i in range(N):
                p[i][0] = xy_flat[2*i]
                p[i][1] = xy_flat[2*i+1]
            return _energy(None, p, edges)
            
        x0 = trial_pos[:, :2].flatten()
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success:
                for i in range(N):
                    trial_pos[i][0] = res.x[2*i]
                    trial_pos[i][1] = res.x[2*i+1]
                    
                # Check validity
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if _pairwise_clearance(tuple(trial_pos[i]), tuple(trial_pos[j])) > 1e-3:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    inner_list = [tuple(p) for p in trial_pos]
                    ocx, ocy, os, oang = _min_enclosing_hexagon(inner_list)
                    if os < best_score:
                        best_score = os
                        best_pos = trial_pos.copy()
        except Exception:
            pass
    
    # If no valid solution found, use the initial placement
    if best_score == float('inf'):
        best_pos = pos
    
    # Final refinement: optimize positions with current best as start
    if time.time() < deadline - 15:
        def objective_final(xy_flat):
            p = best_pos.copy()
            for i in range(N):
                p[i][0] = xy_flat[2*i]
                p[i][1] = xy_flat[2*i+1]
            return _energy(None, p, edges)
            
        x0 = best_pos[:, :2].flatten()
        try:
            res = minimize(objective_final, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
            if res.success:
                for i in range(N):
                    best_pos[i][0] = res.x[2*i]
                    best_pos[i][1] = res.x[2*i+1]
                    
                # Verify still valid
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if _pairwise_clearance(tuple(best_pos[i]), tuple(best_pos[j])) > 1e-3:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    # Revert
                    best_pos = pos.copy()
                    for i in range(N):
                        best_pos[i][2] = 0
        except Exception:
            pass
    
    # Compute final enclosing hexagon
    inner_list = [tuple(p) for p in best_pos]
    ocx, ocy, os, oang = _min_enclosing_hexagon(inner_list)
    
    inner = []
    for p in best_pos:
        inner.append([p[0], p[1], p[2]])
        
    return inner, (ocx, ocy), os, oang

if __name__ == "__main__":
    r = construct()
    print(r)
