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
    
    # Sample angles 0-360
    # For a hexagon, 0-60 is sufficient due to symmetry, but let's scan a bit more for safety
    # Actually, the set of normals rotates. The problem is invariant under 60 deg rotation of the outer hex.
    # So we can just scan 0-60.
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

def _energy(params, positions, edges, k_spring=100.0, k_overlap=1000.0):
    """
    params: not used, just for minimize interface
    positions: array of shape (N, 3) [x, y, angle]
    edges: list of (i, j, rest_length)
    """
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
        
    # Overlap penalty
    for i in range(n):
        for j in range(i+1, n):
            # Only check if they are close
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            if dx*dx + dy*dy < 9.0: # 3^2
                pen = _pairwise_clearance(tuple(positions[i]), tuple(positions[j]))
                if pen > 0:
                    energy += k_overlap * pen
    return energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Define the graph: 11 hexagons in a honeycomb-like cluster
    # Center: 0
    # Ring 1: 1-6 (surrounding 0)
    # Ring 2: 7-10 (attached to 1, 3, 5, or similar to form a compact shape)
    # Let's use a specific compact topology:
    # 0 at center.
    # 1-6 around 0.
    # 7 attached to 1 (opposite side from 0)
    # 8 attached to 3 (opposite side from 0)
    # 9 attached to 5 (opposite side from 0)
    # 10 attached to 2? or 4? 
    # To keep it compact, maybe 10 attached to 1 as well? No, 7 is there.
    # Let's try: 7 on 1, 8 on 2, 9 on 3, 10 on 4. 
    # This creates a "crescent" or "arc" of 4 outer hexes.
    
    # Rest lengths:
    # Edge-to-edge distance for unit hexagons is sqrt(3) (distance between parallel sides)
    # Wait, if they are edge-to-edge, the distance between centers is 2 * apothem = sqrt(3).
    # Yes, apothem = sqrt(3)/2. Distance = 2 * sqrt(3)/2 = sqrt(3).
    rest_len = math.sqrt(3)
    
    edges = []
    # Center to ring 1
    for i in range(1, 7):
        edges.append((0, i, rest_len))
    # Ring 1 to Ring 1 (adjacent)
    for i in range(1, 7):
        j = (i % 6) + 1
        edges.append((i, j, rest_len))
    
    # Outer ring attachments
    # 7 attached to 1
    edges.append((1, 7, rest_len))
    # 8 attached to 2
    edges.append((2, 8, rest_len))
    # 9 attached to 3
    edges.append((3, 9, rest_len))
    # 10 attached to 4
    edges.append((4, 10, rest_len))
    
    # Initial positions based on ideal honeycomb lattice
    # Angles for directions from center to ring 1 vertices: 0, 60, 120, 180, 240, 300
    # Wait, the "direction" to the center of a neighbor is perpendicular to the shared edge.
    # If hex 0 is at (0,0) with angle 0, its vertices are at 0, 60, ...
    # The neighbor sharing the edge between 0 and 60 deg vertices is in the direction 30 deg.
    # Distance is sqrt(3).
    
    pos = np.zeros((N, 3))
    pos[0] = [0, 0, 0]
    
    # Directions to neighbors 1-6
    # Neighbor 1 shares edge at 0-60 deg? No, let's say neighbor 1 is at angle 30 deg relative to hex 0's orientation.
    # If hex 0 is angle 0, neighbors are at angles 30, 90, 150, 210, 270, 330.
    # And they should have the same orientation (0) if they are in a lattice.
    
    angles_neighbor = [30, 90, 150, 210, 270, 330]
    for i in range(1, 7):
        ang_rad = math.radians(angles_neighbor[i-1])
        pos[i] = [rest_len * math.cos(ang_rad), rest_len * math.sin(ang_rad), 0]
        
    # Outer ring
    # 7 attached to 1. 1 is at angle 30 from 0. 
    # The direction from 0 to 1 is 30 deg.
    # The direction from 1 to 7 should be the same (30 deg) to continue the line? 
    # Or 90 deg relative to 1?
    # In a honeycomb, the "next" layer continues in the same radial direction.
    # So 7 is at 1 + (rest_len * cos(30), rest_len * sin(30)).
    # 8 is at 2 + (rest_len * cos(90), rest_len * sin(90)).
    # 9 is at 3 + (rest_len * cos(150), rest_len * sin(150)).
    # 10 is at 4 + (rest_len * cos(210), rest_len * sin(210)).
    
    for idx, neighbor_idx in enumerate([1, 2, 3, 4]):
        node_idx = 7 + idx
        base_pos = pos[neighbor_idx]
        ang_rad = math.radians(angles_neighbor[neighbor_idx-1])
        pos[node_idx] = [
            base_pos[0] + rest_len * math.cos(ang_rad),
            base_pos[1] + rest_len * math.sin(ang_rad),
            0
        ]
        
    # Optimize positions
    # We fix angles to 0 for simplicity first, just optimize x, y?
    # The prompt allows rotation. But for a lattice, 0 is best.
    # Let's allow x, y to vary.
    
    def objective(xy_flat):
        p = pos.copy()
        for i in range(N):
            p[i][0] = xy_flat[2*i]
            p[i][1] = xy_flat[2*i+1]
        return _energy(None, p, edges)
        
    x0 = pos[:, :2].flatten()
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
        if res.success:
            for i in range(N):
                pos[i][0] = res.x[2*i]
                pos[i][1] = res.x[2*i+1]
    except Exception:
        pass
        
    # Check validity and compute enclosing hex
    # It's possible the spring system allowed slight overlaps or didn't minimize the bounding box.
    # Let's do a quick local refinement of the bounding box orientation.
    
    # Actually, the spring system minimizes internal distances, which usually correlates with compactness.
    # But it doesn't directly minimize the outer hex side length.
    # The outer hex side length is determined by the "width" of the cluster in 3 directions.
    # Minimizing internal energy tends to make the cluster as compact as possible (minimal diameter).
    
    # One refinement step: try rotating the whole cluster? No, the relative positions are what matter.
    # The enclosing hexagon function handles the orientation.
    
    # Verify disjointness
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if _pairwise_clearance(tuple(pos[i]), tuple(pos[j])) > 1e-4:
                # If overlap, we have a problem. 
                # We could try to push them apart, but for now, let's assume the spring constant is high enough.
                # If invalid, we might need to restart or accept penalty.
                valid = False
    
    # If invalid, we can try a simple repulsion step
    if not valid:
        for _ in range(100):
            if time.time() > deadline: break
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    pen = _pairwise_clearance(tuple(pos[i]), tuple(pos[j]))
                    if pen > 1e-4:
                        # Push apart
                        dx = pos[i][0] - pos[j][0]
                        dy = pos[i][1] - pos[j][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 1e-6:
                            dx, dy = 1.0, 0.0
                            dist = 1.0
                        force = 0.01 * pen
                        pos[i][0] += force * dx / dist
                        pos[i][1] += force * dy / dist
                        pos[j][0] -= force * dx / dist
                        pos[j][1] -= force * dy / dist
                        moved = True
            if not moved: break
            
    # Final check
    # Re-run spring optimization with current positions as start?
    # Maybe just one more minimize call to settle.
    x0 = pos[:, :2].flatten()
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
        if res.success:
            for i in range(N):
                pos[i][0] = res.x[2*i]
                pos[i][1] = res.x[2*i+1]
    except Exception:
        pass

    # Calculate min enclosing hexagon
    inner_list = [tuple(p) for p in pos]
    ocx, ocy, os, oang = _min_enclosing_hexagon(inner_list)
    
    # Verify containment
    # The LP guarantees containment of vertices.
    
    inner = []
    for p in pos:
        inner.append([p[0], p[1], p[2]])
        
    return inner, (ocx, ocy), os, oang

if __name__ == "__main__":
    r = construct()
    print(r)
