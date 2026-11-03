import numpy as np
from scipy.optimize import minimize
import time

N = 12

def get_hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    angles = np.arange(6) * np.pi / 3 + angle_rad
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def min_outer_side_for_vertices(verts, outer_angle):
    # verts: (M, 2)
    # outer_angle: radians
    phi = outer_angle
    normals = np.column_stack([np.cos(np.arange(6)*np.pi/3 + phi), 
                               np.sin(np.arange(6)*np.pi/3 + phi)])
    proj = verts @ normals.T
    max_proj = np.max(proj)
    return 2.0 / np.sqrt(3.0) * max_proj

def overlap_penalty(inner):
    # inner: list of (x, y, angle_deg)
    # Returns sum of squared penetration depths (approx)
    penalty = 0.0
    n = len(inner)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, a1 = inner[i]
            x2, y2, a2 = inner[j]
            # Check overlap
            # Use SAT
            v1 = get_hex_vertices(x1, y1, a1)
            v2 = get_hex_vertices(x2, y2, a2)
            
            # Axes
            axes = []
            for v in [v1, v2]:
                for k in range(6):
                    p1 = v[k]
                    p2 = v[(k+1)%6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    normal = normal / np.linalg.norm(normal)
                    axes.append(normal)
            
            # Project
            min_dist = np.inf
            for ax in axes:
                proj1 = v1 @ ax
                proj2 = v2 @ ax
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # Overlap on this axis?
                # If max1 < min2 or max2 < min1, no overlap on this axis
                # Gap = min(0, max1 - min2, max2 - min1)
                gap1 = max1 - min2
                gap2 = max2 - min1
                gap = min(gap1, gap2)
                if gap < 0:
                    # Overlapping on this axis
                    # The minimal separation is the minimum gap over all axes
                    # If any axis has positive gap, they are separated.
                    # So if we find an axis with positive gap, return 0 for this pair.
                    # If all axes have negative gap, they overlap.
                    # The penalty is the max penetration depth?
                    # Actually, if they overlap, the "distance" is negative.
                    # We want to push them apart.
                    # The most constraining axis is the one with the smallest gap (most negative).
                    min_dist = min(min_dist, gap)
                else:
                    # Separated on this axis
                    min_dist = 0.0
                    break
            
            if min_dist < 0:
                penalty += min_dist**2
                
    return penalty

def construct():
    start_time = time.time()
    deadline = 150.0
    
    sqrt3 = np.sqrt(3)
    
    # Topology 1: 2x6
    centers_2x6 = []
    for i in range(6):
        x = (i - 2.5) * sqrt3
        y = -0.75
        centers_2x6.append((x, y, 0.0))
        x = (i - 2.5) * sqrt3 + sqrt3/2
        y = 0.75
        centers_2x6.append((x, y, 0.0))
        
    # Topology 2: 3x4
    centers_3x4 = []
    for i in range(4):
        centers_3x4.append(((i - 1.5) * sqrt3, -1.5, 0.0))
        centers_3x4.append(((i - 1.5) * sqrt3 + sqrt3/2, 0.0, 0.0))
        centers_3x4.append(((i - 1.5) * sqrt3, 1.5, 0.0))
        
    best_score = 1e10
    best_inner = None
    best_outer_angle = 0.0
    best_topology = None
    
    for name, centers in [("2x6", centers_2x6), ("3x4", centers_3x4)]:
        # 1. Optimize rotation
        def obj_rot(params):
            outer_angle = params[0]
            inner_angle = params[1]
            inner = [(x, y, inner_angle) for x, y, _ in centers]
            verts = []
            for x, y, ang in inner:
                verts.append(get_hex_vertices(x, y, ang))
            verts = np.vstack(verts)
            return min_outer_side_for_vertices(verts, np.deg2rad(outer_angle))
            
        x0 = [0.0, 0.0]
        try:
            res_rot = minimize(obj_rot, x0, method='Nelder-Mead', options={'maxiter': 100})
            best_rot = res_rot.x
            score_rot = res_rot.fun
        except:
            best_rot = [0.0, 0.0]
            score_rot = obj_rot(x0)
            
        # 2. Local search on positions
        # Variables: x1..x12, y1..y12, th1..th12, outer_angle
        # Total 37
        x0_local = []
        for x, y, _ in centers:
            x0_local.extend([x, y, 0.0])
        x0_local.append(best_rot[0])
        
        def obj_local(params):
            if time.time() > deadline - 5:
                return 1e10
            inner = []
            for k in range(12):
                x = params[3*k]
                y = params[3*k+1]
                th = params[3*k+2]
                inner.append((x, y, th))
            outer_angle = params[36]
            
            # Overlap penalty
            pen = overlap_penalty(inner)
            
            # Side length
            verts = []
            for x, y, ang in inner:
                verts.append(get_hex_vertices(x, y, ang))
            verts = np.vstack(verts)
            side = min_outer_side_for_vertices(verts, np.deg2rad(outer_angle))
            
            return side + 10.0 * pen
            
        try:
            res_local = minimize(obj_local, x0_local, method='Nelder-Mead', 
                                  options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            if res_local.success:
                # Check validity
                inner_final = []
                for k in range(12):
                    x = res_local.x[3*k]
                    y = res_local.x[3*k+1]
                    th = res_local.x[3*k+2]
                    inner_final.append([x, y, th])
                outer_angle_final = res_local.x[36]
                
                pen = overlap_penalty(inner_final)
                if pen < 1e-6:
                    verts = []
                    for x, y, ang in inner_final:
                        verts.append(get_hex_vertices(x, y, ang))
                    verts = np.vstack(verts)
                    side_final = min_outer_side_for_vertices(verts, np.deg2rad(outer_angle_final))
                    
                    if side_final < best_score:
                        best_score = side_final
                        best_inner = inner_final
                        best_outer_angle = outer_angle_final
                        best_topology = name
        except Exception as e:
            pass
            
    # If no local search worked, use the rotation-only result
    if best_inner is None:
        # Use 2x6 with best rotation
        inner = [(x, y, best_rot[1]) for x, y, _ in centers_2x6]
        best_inner = [[x, y, th] for x, y, th in inner]
        best_outer_angle = best_rot[0]
        verts = []
        for x, y, ang in best_inner:
            verts.append(get_hex_vertices(x, y, ang))
        verts = np.vstack(verts)
        best_score = min_outer_side_for_vertices(verts, np.deg2rad(best_outer_angle))

    return best_inner, (0.0, 0.0), best_score, best_outer_angle
