import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    deadline = time.time() + 145.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Precompute unit hex vertices (flat-top: vertices at 0, 60, 120, 180, 240, 300)
    base_verts = [(math.cos(math.radians(60*i)), math.sin(math.radians(60*i))) for i in range(6)]

    def transform_verts(cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        return [(cx + S*(vx*ca - vy*sa), cy + S*(vx*sa + vy*ca)) for vx, vy in base_verts]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    def segments_intersect(p1, p2, p3, p4):
        d1 = cross(p3, p4, p1); d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3); d4 = cross(p1, p2, p4)
        if ((d1>0 and d2<0) or (d1<0 and d2>0)) and ((d3>0 and d4<0) or (d3<0 and d4>0)):
            return True
        return False

    def point_in_polygon(p, poly):
        x, y = p
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]; xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi) + xi):
                inside = not inside
            j = i
        return inside

    def hexagons_disjoint(va, vb):
        for v in va:
            if point_in_polygon(v, vb): return False
        for v in vb:
            if point_in_polygon(v, va): return False
        for i in range(6):
            if segments_intersect(va[i], va[(i+1)%6], vb[i], vb[(i+1)%6]):
                return False
        return True

    def compute_outer_hex(inner, ocx, ocy, oangle):
        inner_verts = [transform_verts(ix, iy, 1.0, ia) for ix, iy, ia in inner]
        a = math.radians(oangle)
        apothem_factor = math.sqrt(3) / 2
        max_S = 0.0
        for i in range(6):
            theta = a + math.radians(30 + 60*i)
            nx, ny = math.cos(theta), math.sin(theta)
            max_proj = 0.0
            for iv in inner_verts:
                for v in iv:
                    proj = (v[0]-ocx)*nx + (v[1]-ocy)*ny
                    if proj > max_proj: max_proj = proj
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S: max_S = S_needed
        return max_S

    def all_fit(inner, ocx, ocy, S, oangle):
        outer_verts = transform_verts(ocx, ocy, S, oangle)
        inner_verts = [transform_verts(ix, iy, 1.0, ia) for ix, iy, ia in inner]
        for iv in inner_verts:
            for v in iv:
                if not point_in_polygon(v, outer_verts):
                    return False
        for i in range(len(inner_verts)):
            for j in range(i+1, len(inner_verts)):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                    return False
        return True

    def update_best(inner, ocx, ocy, S, oangle):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        if S < best_S and all_fit(inner, ocx, ocy, S, oangle):
            best_S = S
            best_inner = [list(x) for x in inner]
            best_center = (ocx, ocy)
            best_angle = oangle
            best_valid = True
            return True
        return False

    # ---- Structural change: 1-D rigid-body optimization over 6-fold symmetry ----
    # We fix the internal arrangement of the 11 hexagons to a dense 3-4-3 cluster
    # on a hexagonal lattice. The only degrees of freedom are the cluster's
    # translation (tx, ty) and rotation (theta) relative to the outer container.
    # The outer container is centered at the origin with side length S and rotation 0.
    # We minimize S, which is determined by the support function of the rigid cluster.

    # Hexagonal lattice spacing for unit hexagons (side 1)
    sp = math.sqrt(3)
    a1 = (sp, 0)
    a2 = (sp/2, sp*math.sqrt(3)/2)

    # 3-4-3 cluster: rows of 3, 4, 3 hexagons
    # Row 0 (top): 3 hexagons
    # Row 1 (mid): 4 hexagons
    # Row 2 (bot): 3 hexagons
    # Coordinates relative to cluster center (approx)
    # Let's define the lattice points
    # Using axial coordinates (q, r) for hex grid
    # Point (q, r) -> x = sp * (q + r/2), y = sp * (sqrt(3)/2) * r
    # 3-4-3 cluster:
    # r=1: q=-1, 0, 1  (3)
    # r=0: q=-1, 0, 1, 2 (4) -> wait, standard 3-4-3 is:
    # r=1: q=0, 1, 2
    # r=0: q=-1, 0, 1, 2
    # r=-1: q=0, 1, 2
    # Let's check adjacency.
    # (0,1) neighbors: (0,0), (1,0), (1,1), (0,2), (-1,1), (-1,2)
    # (1,1) neighbors: (1,0), (2,0), (2,1), (1,2), (0,1), (0,2)
    # (2,1) neighbors: (2,0), (3,0), (3,1), (2,2), (1,1), (1,2)
    # Row r=0: q=-1,0,1,2
    # (0,0) neighbors: (0,-1), (1,-1), (1,0), (0,1), (-1,0), (-1,1)
    # So (0,1) is neighbor of (0,0). (1,1) is neighbor of (1,0). (2,1) is neighbor of (2,0).
    # Row r=-1: q=0,1,2
    # (0,-1) neighbor of (0,0) and (1,0)? (0,-1) neighbors: (0,0), (1,0), (1,-1), (0,-2), (-1,-1), (-1,0)
    # So (0,-1) touches (0,0) and (1,0).
    # (1,-1) touches (1,0) and (2,0).
    # (2,-1) touches (2,0).
    # This forms a compact cluster.
    
    cluster_points = []
    # r=1
    for q in [0, 1, 2]:
        x = sp * (q + 1/2)
        y = sp * (math.sqrt(3)/2) * 1
        cluster_points.append((x, y))
    # r=0
    for q in [-1, 0, 1, 2]:
        x = sp * (q + 1/2)
        y = sp * (math.sqrt(3)/2) * 0
        cluster_points.append((x, y))
    # r=-1
    for q in [0, 1, 2]:
        x = sp * (q + 1/2)
        y = sp * (math.sqrt(3)/2) * (-1)
        cluster_points.append((x, y))
        
    # Center the cluster
    cx = sum(p[0] for p in cluster_points) / N
    cy = sum(p[1] for p in cluster_points) / N
    cluster_points = [(p[0]-cx, p[1]-cy) for p in cluster_points]
    
    # Precompute cluster vertices in local frame
    local_verts = []
    for px, py in cluster_points:
        verts = [(px + vx, py + vy) for vx, vy in base_verts]
        local_verts.append(verts)
        
    # Flatten for faster projection
    all_local_pts = []
    for verts in local_verts:
        for v in verts:
            all_local_pts.append(v)
    all_local_pts = np.array(all_local_pts)

    def rigid_objective(params):
        if time.time() > deadline:
            return 1e10
        tx, ty, theta_deg = params
        theta = math.radians(theta_deg)
        ca, sa = math.cos(theta), math.sin(theta)
        
        # Rotate and translate points
        # p' = R * p + t
        # x' = ca*x - sa*y + tx
        # y' = sa*x + ca*y + ty
        pts = all_local_pts
        x = pts[:, 0]
        y = pts[:, 1]
        tx_arr = np.full_like(x, tx)
        ty_arr = np.full_like(y, ty)
        x_new = ca * x - sa * y + tx_arr
        y_new = sa * x + ca * y + ty_arr
        
        # Compute max projection onto 6 normals of outer hexagon (centered at origin, rot 0)
        max_S = 0.0
        apothem_factor = math.sqrt(3) / 2
        for i in range(6):
            theta_n = math.radians(30 + 60*i)
            nx, ny = math.cos(theta_n), math.sin(theta_n)
            proj = x_new * nx + y_new * ny
            max_proj = np.max(proj)
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S: max_S = S_needed
        return max_S

    # Optimization
    # Initial guess: tx=0, ty=0, theta=0
    # Bounds: tx, ty in [-4, 4], theta in [0, 60]
    bounds = [(-4, 4), (-4, 4), (0, 60)]
    
    # Multi-start Nelder-Mead
    best_params = None
    best_obj = float('inf')
    
    # Try several initial rotations
    for theta0 in [0, 10, 20, 30, 40, 50]:
        if time.time() > deadline:
            break
        x0 = [0, 0, theta0]
        try:
            res = minimize(rigid_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is not None:
        tx, ty, theta = best_params
        # Construct inner list
        theta_rad = math.radians(theta)
        ca, sa = math.cos(theta_rad), math.sin(theta_rad)
        inner = []
        for px, py in cluster_points:
            x_new = ca * px - sa * py + tx
            y_new = sa * px + ca * py + ty
            inner.append([x_new, y_new, theta])
        
        S = best_obj
        # Verify validity
        if all_fit(inner, 0.0, 0.0, S, 0.0):
            update_best(inner, 0.0, 0.0, S, 0.0)

    # Fallback if rigid optimization failed or is worse than expected
    if not best_valid or best_inner is None:
        # Use the lattice blob fallback from the original program
        sp = math.sqrt(3) * 1.001
        a1 = (sp, 0)
        a2 = (sp/2, sp*math.sqrt(3)/2)
        points = []
        for i in range(-8, 9):
            for j in range(-8, 9):
                x = i*a1[0] + j*a2[0]
                y = i*a1[1] + j*a2[1]
                points.append((x, y))
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        selected = points[:N]
        cx = sum(p[0] for p in selected)/N
        cy = sum(p[1] for p in selected)/N
        inner = [[p[0]-cx, p[1]-cy, 0.0] for p in selected]
        S = compute_outer_hex(inner, 0.0, 0.0, S, 0.0)
        if all_fit(inner, 0.0, 0.0, S, 0.0):
            return inner, (0.0, 0.0), S, 0.0
        for _ in range(200):
            inner = [[p[0]*1.02, p[1]*1.02, p[2]] for p in inner]
            S = compute_outer_hex(inner, 0.0, 0.0, S, 0.0)
            if all_fit(inner, 0.0, 0.0, S, 0.0):
                return inner, (0.0, 0.0), S, 0.0

    if best_valid and best_inner is not None:
        return best_inner, best_center, best_S, best_angle

    # Absolute fallback
    sp = math.sqrt(3)
    inner = [[(k - N//2)*sp, 0.0, 0.0] for k in range(N)]
    S = compute_outer_hex(inner, 0.0, 0.0, S, 0.0)
    for _ in range(200):
        if all_fit(inner, 0.0, 0.0, S, 0.0):
            return inner, (0.0, 0.0), S, 0.0
        inner = [[p[0]*1.05, p[1]*1.05, p[2]] for p in inner]
        S = compute_outer_hex(inner, 0.0, 0.0, S, 0.0)
    return inner, (0.0, 0.0), S, 0.0
