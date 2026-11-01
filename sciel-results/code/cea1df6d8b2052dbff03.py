"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    return np.array([(cx + side * np.cos(a + i * np.pi / 3), 
                      cy + side * np.sin(a + i * np.pi / 3)) for i in range(6)])

def _point_to_segment_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    return np.sqrt((px - (x1 + t * dx))**2 + (py - (y1 + t * dy))**2)

def _poly_min_dist(v1, v2):
    """Min distance between two convex hexagons (arrays of 6 points each)."""
    min_d = float('inf')
    for i in range(6):
        for j in range(6):
            d = _point_to_segment_dist(v1[i][0], v1[i][1], v2[j][0], v2[j][1], 
                                       v2[(j+1)%6][0], v2[(j+1)%6][1])
            if d < min_d:
                min_d = d
            d = _point_to_segment_dist(v2[i][0], v2[i][1], v1[j][0], v1[j][1], 
                                       v1[(j+1)%6][0], v1[(j+1)%6][1])
            if d < min_d:
                min_d = d
    return min_d

def _containment_margin(cx, cy, angle_deg, outer_side, outer_cx=0.0, outer_cy=0.0, outer_angle=0.0):
    """Min signed distance from inner unit hex to outer hex boundary (positive=inside)."""
    inner_v = _hex_vertices(cx, cy, 1.0, angle_deg)
    outer_v = _hex_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    min_m = float('inf')
    for i in range(6):
        p1, p2 = outer_v[i], outer_v[(i+1)%6]
        ex, ey = p2[0]-p1[0], p2[1]-p1[1]
        el = np.sqrt(ex*ex + ey*ey)
        if el < 1e-12:
            continue
        nx, ny = -ey/el, ex/el  # inward normal for CCW
        for v in inner_v:
            d = (v[0]-p1[0])*nx + (v[1]-p1[1])*ny
            if d < min_m:
                min_m = d
    return min_m

def _fast_overlap(cx1, cy1, a1, cx2, cy2, a2):
    """Fast overlap check using center distance heuristic."""
    dx, dy = cx1-cx2, cy1-cy2
    dist = np.sqrt(dx*dx + dy*dy)
    # For two unit hexagons, if dist > 2.0 they definitely don't overlap
    if dist > 2.0:
        return dist - 2.0
    # If dist < sqrt(3) ≈ 1.732, they definitely overlap (2*inradius)
    if dist < np.sqrt(3) - 1e-9:
        return dist - np.sqrt(3)
    # In between, compute exact
    v1 = _hex_vertices(cx1, cy1, 1.0, a1)
    v2 = _hex_vertices(cx2, cy2, 1.0, a2)
    md = _poly_min_dist(v1, v2)
    # Check if they actually overlap (vertex inside other)
    # Simple: if md < 1e-9, likely touching or overlapping
    return md

def _objective(params):
    side = max(params[0], 1.0)
    ip = params[1:]
    pen = 0.0
    # Containment
    for i in range(N):
        m = _containment_margin(ip[i*3], ip[i*3+1], ip[i*3+2]*180/np.pi, side)
        if m < 0:
            pen += m * m
    # Overlap
    for i in range(N):
        for j in range(i+1, N):
            m = _fast_overlap(ip[i*3], ip[i*3+1], ip[i*3+2]*180/np.pi,
                             ip[j*3], ip[j*3+1], ip[j*3+2]*180/np.pi)
            if m < 0:
                pen += m * m
    return side + 50.0 * pen

def construct():
    t0 = time.time()
    deadline = t0 + 130.0
    
    best_side = float('inf')
    best_inner = None
    
    sqrt3 = np.sqrt(3)
    
    # Generate multiple initial configurations
    configs = []
    
    # 1. Linear
    configs.append([(2.0*(k-5), 0.0, 0.0) for k in range(N)])
    
    # 2. Two rows 6+5
    r1 = [(2.0*(k-2.5), 0.0, 0.0) for k in range(6)]
    r2 = [(2.0*(k-2.0), 1.5, 0.0) for k in range(5)]
    configs.append(r1 + r2)
    
    # 3. Three rows 4+4+3
    r1 = [(2.0*(k-1.5), 0.0, 0.0) for k in range(4)]
    r2 = [(2.0*(k-1.5), 1.5, 0.0) for k in range(4)]
    r3 = [(2.0*(k-1.0), 3.0, 0.0) for k in range(3)]
    configs.append(r1 + r2 + r3)
    
    # 4. Honeycomb 1+6+4
    hc = [(0.0, 0.0, 0.0)]
    for i in range(6):
        a = i*np.pi/3
        hc.append((sqrt3*np.cos(a), sqrt3*np.sin(a), 0.0))
    for i in range(4):
        a = i*np.pi/3
        hc.append((2*sqrt3*np.cos(a), 2*sqrt3*np.sin(a), 0.0))
    configs.append(hc)
    
    # 5. Compact 3x4 grid-ish
    for r in range(3):
        for c in range(4):
            if r*4 + c < N:
                configs[-1].append((1.5*c - 2.25, sqrt3*r, 0.0))
    
    for cfg in configs:
        if time.time() > deadline:
            break
        if len(cfg) != N:
            continue
        
        xs = [p[0] for p in cfg]
        ys = [p[1] for p in cfg]
        cx0 = (min(xs)+max(xs))/2
        cy0 = (min(ys)+max(ys))/2
        spread = max(max(xs)-min(xs), max(ys)-min(ys))
        side0 = spread/2 + 1.5
        
        x0 = [side0]
        for p in cfg:
            x0.extend([p[0]-cx0, p[1]-cy0, p[2]*np.pi/180])
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-3, 'fatol': 1e-3, 'adaptive': True})
            
            side_opt = max(res.x[0], 1.0)
            inner_opt = [[res.x[1+i*3], res.x[2+i*3], res.x[3+i*3]*180/np.pi] for i in range(N)]
            
            # Verify
            valid = True
            for i in range(N):
                m = _containment_margin(inner_opt[i][0], inner_opt[i][1], inner_opt[i][2], side_opt)
                if m < -1e-5:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        m = _fast_overlap(inner_opt[i][0], inner_opt[i][1], inner_opt[i][2],
                                          inner_opt[j][0], inner_opt[j][1], inner_opt[j][2])
                        if m < -1e-5:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid and side_opt < best_side:
                best_side = side_opt
                best_inner = inner_opt
                
        except Exception:
            pass
    
    if best_inner is None:
        inner = [[2.0*k - 10.0, 0.0, 0.0] for k in range(N)]
        return inner, (0.0, 0.0), 22.0, 0.0
    
    return best_inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END
