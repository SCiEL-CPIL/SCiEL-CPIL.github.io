import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hex_verts(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    return np.array([[cx + s*np.cos(a + k*np.pi/3), cy + s*np.sin(a + k*np.pi/3)] for k in range(6)])

def min_side_for_hexagons(positions, angles, cx, cy, outer_a):
    """Min side length S of outer hex (center cx,cy, rot outer_a) containing all unit hexes."""
    max_proj = 0.0
    oa = np.radians(outer_a)
    for i in range(len(positions)):
        v = hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
        for vx, vy in v:
            dx, dy = vx - cx, vy - cy
            for k in range(6):
                th = oa + k * np.pi/3
                proj = dx * np.cos(th) + dy * np.sin(th)
                if proj > max_proj:
                    max_proj = proj
    return max(max_proj / (np.sqrt(3)/2), 1.0)

def overlap_penalty(positions, angles):
    """Penalty for overlapping hexagons."""
    pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            # Quick reject
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist >= 2.0:
                continue
            # Detailed check
            v1 = hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
            v2 = hex_verts(positions[j][0], positions[j][1], 1.0, angles[j])
            min_d = float('inf')
            for p in v1:
                for k in range(6):
                    p1 = v2[k]
                    p2 = v2[(k+1)%6]
                    d = p2 - p1
                    f = p - p1
                    t = np.dot(f, d) / (np.dot(d, d) + 1e-15)
                    t = max(0.0, min(1.0, t))
                    c = p1 + t * d
                    dd = np.linalg.norm(p - c)
                    if dd < min_d:
                        min_d = dd
            if min_d < 0.01:
                pen += (0.01 - min_d) ** 2
    return pen

def objective(params):
    pos = [(params[3*i], params[3*i+1]) for i in range(N)]
    ang = [params[3*i+2] for i in range(N)]
    cx, cy, oa = params[3*N], params[3*N+1], params[3*N+2]
    S = min_side_for_hexagons(pos, ang, cx, cy, oa)
    pen = overlap_penalty(pos, ang)
    return S + 50.0 * pen

def make_layouts():
    layouts = []
    # Layout 1: hex cluster
    pos = [(0,0)]
    for k in range(6):
        th = k*np.pi/3
        pos.append((2*np.cos(th), 2*np.sin(th)))
    for k in range(4):
        th = k*np.pi/2 + np.pi/6
        pos.append((3.6*np.cos(th), 3.6*np.sin(th)))
    layouts.append((pos, [0.0]*N))
    
    # Layout 2: 3-4-3
    pos = []
    d = 2.0
    for i in range(3):
        pos.append((i*d - d, 0))
    for i in range(4):
        pos.append((i*d - 1.5*d, d*np.sqrt(3)))
    for i in range(3):
        pos.append((i*d - d, 2*d*np.sqrt(3)))
    layouts.append((pos, [0.0]*N))
    
    # Layout 3: line
    pos = [(i*2 - 5, 0) for i in range(N)]
    layouts.append((pos, [0.0]*N))
    
    # Layout 4: compact with some rotation
    pos = [(0,0)]
    for k in range(6):
        th = k*np.pi/3
        pos.append((2*np.cos(th), 2*np.sin(th)))
    for k in range(4):
        th = k*np.pi/2
        pos.append((3.2*np.cos(th), 3.2*np.sin(th)))
    layouts.append((pos, [0.0]*N))
    
    return layouts

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 135.0
    
    layouts = make_layouts()
    best_S = float('inf')
    best = None
    
    for li, (pos0, ang0) in enumerate(layouts):
        if time.time() > deadline:
            break
        p0 = []
        for i in range(N):
            p0.extend([pos0[i][0], pos0[i][1], ang0[i]])
        p0.extend([0.0, 0.0, 0.0])
        
        try:
            res = minimize(objective, p0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7})
            if time.time() > deadline:
                break
            pf = res.x
            posf = [(pf[3*i], pf[3*i+1]) for i in range(N)]
            angf = [pf[3*i+2] for i in range(N)]
            cx, cy, oa = pf[3*N], pf[3*N+1], pf[3*N+2]
            
            # Verify no overlap
            ok = True
            for i in range(N):
                for j in range(i+1, N):
                    dx = posf[i][0]-posf[j][0]
                    dy = posf[i][1]-posf[j][1]
                    if np.sqrt(dx*dx+dy*dy) < 1.99:
                        v1 = hex_verts(posf[i][0], posf[i][1], 1.0, angf[i])
                        v2 = hex_verts(posf[j][0], posf[j][1], 1.0, angf[j])
                        md = float('inf')
                        for p in v1:
                            for k in range(6):
                                a = v2[k]; b = v2[(k+1)%6]
                                d = b-a; f = p-a
                                t = np.dot(f,d)/(np.dot(d,d)+1e-15)
                                t = max(0,min(1,t))
                                c = a+t*d
                                md = min(md, np.linalg.norm(p-c))
                        if md < 0.005:
                            ok = False
                            break
                if not ok:
                    break
            
            if ok:
                S = min_side_for_hexagons(posf, angf, cx, cy, oa)
                if S < best_S:
                    best_S = S
                    best = (posf, angf, cx, cy, oa)
        except:
            pass
    
    if best is None:
        posf, angf = layouts[0]
        best_S = min_side_for_hexagons(posf, angf, 0, 0, 0)
        best = (posf, angf, 0, 0, 0)
    
    posf, angf, cx, cy, oa = best
    inner = [[posf[i][0], posf[i][1], angf[i]] for i in range(N)]
    return inner, (cx, cy), best_S, oa
# EVOLVE-BLOCK-END

