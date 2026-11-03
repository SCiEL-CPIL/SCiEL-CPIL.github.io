import numpy as np
from scipy.optimize import minimize
import time

N = 11
START_TIME = time.time()
DEADLINE = 140.0

def time_remaining():
    return DEADLINE - time.time()

def hex_verts(cx, cy, s, ang_deg=0):
    a = np.radians(ang_deg)
    return np.array([[cx + s*np.cos(a + k*np.pi/3), cy + s*np.sin(a + k*np.pi/3)] for k in range(6)])

def point_in_hex(p, S):
    """Check if point p is inside regular hexagon of side S centered at origin, flat-top."""
    sqrt3 = np.sqrt(3)
    x, y = p
    return (sqrt3*x + y <= S*sqrt3 + 1e-9 and
            y <= S*sqrt3/2 + 1e-9 and
            -sqrt3*x + y <= S*sqrt3 + 1e-9 and
            -sqrt3*x - y <= S*sqrt3 + 1e-9 and
            -y <= S*sqrt3/2 + 1e-9 and
            sqrt3*x - y <= S*sqrt3 + 1e-9)

def min_dist_hexes(v1, v2):
    """Min distance between two hexagons (0 if overlapping)."""
    def pt_in_hex(p, hex_v):
        n = 6
        inside = False
        j = n - 1
        for i in range(n):
            if ((hex_v[i][1] > p[1]) != (hex_v[j][1] > p[1])) and \
               (p[0] < (hex_v[j][0]-hex_v[i][0])*(p[1]-hex_v[i][1])/(hex_v[j][1]-hex_v[i][1])+hex_v[i][0]):
                inside = not inside
            j = i
        return inside
    
    for v in v1:
        if pt_in_hex(v, v2):
            return 0.0
    for v in v2:
        if pt_in_hex(v, v1):
            return 0.0
    
    min_d = 1e9
    for i in range(6):
        a, b = v1[i], v1[(i+1)%6]
        for j in range(6):
            c, d = v2[j], v2[(j+1)%6]
            ab = b - a
            cd = d - c
            ac = c - a
            aab = np.dot(ab, ab)
            ccd = np.dot(cd, cd)
            abcd = np.dot(ab, cd)
            den = aab * ccd - abcd * abcd
            if abs(den) < 1e-14:
                t = 0.0
                s = 0.0
                if aab > 1e-14:
                    t = np.clip(np.dot(ac, ab)/aab, 0, 1)
                if ccd > 1e-14:
                    s = np.clip(np.dot(ac - t*ab, cd)/ccd, 0, 1)
                min_d = min(min_d, np.linalg.norm(ac + s*cd - t*ab))
            else:
                t = (ccd * np.dot(ac, ab) - abcd * np.dot(ac, cd)) / den
                s = (aab * np.dot(ac, cd) - abcd * np.dot(ac, ab)) / den
                t = np.clip(t, 0, 1)
                s = np.clip(s, 0, 1)
                min_d = min(min_d, np.linalg.norm(ac + s*cd - t*ab))
    return min_d

def construct():
    # Initial guess: compact hexagonal arrangement
    # Row-based: rows of 3, 4, 3, 1? Or use a known good starting point.
    # Let's start with a grid-like arrangement.
    
    # Use a honeycomb-like initial guess
    centers = np.zeros((N, 2))
    
    # Place in a compact cluster
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 3 hexagons
    # Row 3: 1 hexagon
    # Total: 11
    
    dy = 1.5 * 1.001  # vertical spacing
    dx = np.sqrt(3) * 1.001  # horizontal spacing
    
    # Row 0: y=0, x = -dx, 0, dx
    centers[0] = [-dx, 0]
    centers[1] = [0, 0]
    centers[2] = [dx, 0]
    
    # Row 1: y=dy, x = -1.5*dx, -0.5*dx, 0.5*dx, 1.5*dx
    centers[3] = [-1.5*dx, dy]
    centers[4] = [-0.5*dx, dy]
    centers[5] = [0.5*dx, dy]
    centers[6] = [1.5*dx, dy]
    
    # Row 2: y=2*dy, x = -dx, 0, dx
    centers[7] = [-dx, 2*dy]
    centers[8] = [0, 2*dy]
    centers[9] = [dx, 2*dy]
    
    # Row 3: y=3*dy, x = 0
    centers[10] = [0, 3*dy]
    
    S_init = 5.0
    
    def objective(params):
        if time_remaining() < 5.0:
            return 1e6
        S = params[2*N]
        if S < 2.5:
            return 1e6
        c = params[:2*N].reshape(N, 2)
        
        penalty = 0.0
        
        # Check containment
        for i in range(N):
            verts = hex_verts(c[i][0], c[i][1], 1.001)
            for v in verts:
                if not point_in_hex(v, S):
                    penalty += 10.0
        
        # Check disjointness (only check close pairs)
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(c[i] - c[j])
                if d < 2.5:
                    v1 = hex_verts(c[i][0], c[i][1], 1.001)
                    v2 = hex_verts(c[j][0], c[j][1], 1.001)
                    md = min_dist_hexes(v1, v2)
                    if md < 0.01:
                        penalty += 5.0 * (0.01 - md)
        
        return S + penalty
    
    x0 = np.concatenate([centers.flatten(), [S_init]])
    
    best_S = S_init
    best_c = centers.copy()
    
    try:
        res = minimize(objective, x0, method='SLSQP', options={'maxiter': 300, 'ftol': 1e-7})
        if res.success and res.fun < S_init:
            best_S = res.x[2*N]
            best_c = res.x[:2*N].reshape(N, 2)
    except:
        pass
    
    # Tighten S
    def tighten(c_arr, S_lo, S_hi, max_iter=50):
        for _ in range(max_iter):
            if time_remaining() < 5.0:
                break
            mid = (S_lo + S_hi) / 2
            ok = True
            for i in range(N):
                verts = hex_verts(c_arr[i][0], c_arr[i][1], 1.001)
                for v in verts:
                    if not point_in_hex(v, mid):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                S_hi = mid
            else:
                S_lo = mid
        return (S_lo + S_hi) / 2
    
    try:
        S_tight = tighten(best_c, 3.0, best_S + 1, 40)
        best_S = min(best_S, S_tight)
    except:
        pass
    
    inner = [[float(best_c[i][0]), float(best_c[i][1]), 0.0] for i in range(N)]
    return inner, (0.0, 0.0), float(best_S), 0.0

