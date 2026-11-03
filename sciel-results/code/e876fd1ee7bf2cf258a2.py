import numpy as np
from scipy.optimize import minimize
import time

N = 12

def get_hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    angles = np.arange(6) * np.pi / 3 + angle_rad
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def min_outer_side_for_vertices(verts, outer_angle):
    # Center the vertices at the origin to allow the outer hexagon to be centered at (0,0)
    center = np.mean(verts, axis=0)
    verts_centered = verts - center
    
    # outer_angle: radians
    phi = outer_angle
    normals = np.column_stack([np.cos(np.arange(6)*np.pi/3 + phi), 
                               np.sin(np.arange(6)*np.pi/3 + phi)])
    proj = verts_centered @ normals.T
    max_proj = np.max(proj)
    return 2.0 / np.sqrt(3.0) * max_proj

def overlap_penalty(inner, eps=1e-9):
    # inner: list of (x, y, angle_deg)
    # Returns sum of squared penetration depths
    penalty = 0.0
    n = len(inner)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, a1 = inner[i]
            x2, y2, a2 = inner[j]
            v1 = get_hex_vertices(x1, y1, a1)
            v2 = get_hex_vertices(x2, y2, a2)
            
            axes = []
            for v in [v1, v2]:
                for k in range(6):
                    p1 = v[k]
                    p2 = v[(k+1)%6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-10:
                        normal = normal / norm
                        axes.append(normal)
            
            min_dist = np.inf
            for ax in axes:
                proj1 = v1 @ ax
                proj2 = v2 @ ax
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                gap1 = max1 - min2
                gap2 = max2 - min1
                gap = min(gap1, gap2)
                
                if gap < -eps:
                    min_dist = min(min_dist, gap)
                else:
                    min_dist = 0.0
                    break
            
            if min_dist < 0:
                penalty += min_dist**2
                
    return penalty

def construct():
    start_time = time.time()
    deadline = 145.0
    
    sqrt3 = np.sqrt(3)
    
    # Generate a valid initial topology with small gaps to ensure disjointness
    # Using a honeycomb-like structure but scaled up slightly
    # Base honeycomb vectors for side 1:
    # a1 = (1.5, sqrt(3)/2), a2 = (1.5, -sqrt(3)/2)
    # To create gaps, we scale the positions by a factor > 1
    
    def generate_topology(rows, cols, scale=1.1):
        centers = []
        for r in range(rows):
            for c in range(cols):
                # Honeycomb coordinates
                x = c * 3.0 + (r % 2) * 1.5
                y = r * sqrt3 / 2.0
                # Center the cluster roughly
                # Approximate center for 3x4
                if rows == 3 and cols == 4:
                    cx_offset = 3.0
                    cy_offset = sqrt3
                elif rows == 2 and cols == 6:
                    cx_offset = 7.5
                    cy_offset = sqrt3/2
                else:
                    cx_offset = 0
                    cy_offset = 0
                    
                x = (x - cx_offset) * scale
                y = (y - cy_offset) * scale
                centers.append([x, y, 0.0])
        return centers

    # 3x4 Topology
    centers_3x4 = generate_topology(3, 4, scale=1.1)
    
    # 2x6 Topology
    centers_2x6 = generate_topology(2, 6, scale=1.1)
    
    best_score = 1e10
    best_inner = None
    best_outer_angle = 0.0
    
    topologies = [("3x4", centers_3x4), ("2x6", centers_2x6)]
    
    for name, centers in topologies:
        if time.time() > deadline - 10:
            break
            
        # Initial check
        init_pen = overlap_penalty(centers)
        if init_pen > 1e-6:
            continue # Skip if initial is invalid
            
        # Optimize
        x0_local = []
        for x, y, _ in centers:
            x0_local.extend([x, y, 0.0])
        x0_local.append(0.0) # outer angle
        
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
            
            pen = overlap_penalty(inner)
            if pen > 1e-4:
                return 1e8 + pen # Strong penalty for invalid
                
            verts = []
            for x, y, ang in inner:
                verts.append(get_hex_vertices(x, y, ang))
            verts = np.vstack(verts)
            side = min_outer_side_for_vertices(verts, np.deg2rad(outer_angle))
            
            return side
            
        try:
            res_local = minimize(obj_local, x0_local, method='Nelder-Mead', 
                                  options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
            
            # Check validity of result
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
                    
        except Exception:
            pass
            
    # Fallback
    if best_inner is None:
        # Use 3x4 initial (scaled)
        best_inner = [[x, y, th] for x, y, th in centers_3x4]
        verts = []
        for x, y, ang in best_inner:
            verts.append(get_hex_vertices(x, y, ang))
        verts = np.vstack(verts)
        best_score = min_outer_side_for_vertices(verts, 0.0)
        best_outer_angle = 0.0

    return best_inner, (0.0, 0.0), best_score, best_outer_angle
