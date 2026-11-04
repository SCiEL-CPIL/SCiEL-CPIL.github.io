"""Packing 12 unit hexagons in a regular hexagon - minimize outer side length."""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 12

def hex_vertices(cx, cy, s, angle_deg=0.0):
    a = np.radians(angle_deg)
    return np.array([[cx + s*np.cos(a + i*np.pi/3), cy + s*np.sin(a + i*np.pi/3)] for i in range(6)])

def point_in_hexagon(px, py, cx, cy, s, angle_deg=0.0):
    a = np.radians(angle_deg)
    dx, dy = px - cx, py - cy
    ca, sa = np.cos(-a), np.sin(-a)
    lx = dx*ca - dy*sa
    ly = dx*sa + dy*ca
    apothem = s * np.sqrt(3) / 2
    c30, s30 = np.cos(np.pi/6), np.sin(np.pi/6)
    if abs(ly) > apothem: return False
    if abs(lx*c30 + ly*s30) > apothem: return False
    if abs(lx*c30 - ly*s30) > apothem: return False
    return True

def check_overlap(i, j, positions, angles, s=1.0):
    vi = hex_vertices(positions[i,0], positions[i,1], s, angles[i])
    vj = hex_vertices(positions[j,0], positions[j,1], s, angles[j])
    for k in range(6):
        if point_in_hexagon(vi[k,0], vi[k,1], positions[j,0], positions[j,1], s, angles[j]):
            return True
        if point_in_hexagon(vj[k,0], vj[k,1], positions[i,0], positions[i,1], s, angles[i]):
            return True
    return False

def outer_hex_side(positions, angles, outer_angle_deg=0.0):
    s = 1.0
    all_pts = []
    for i in range(N):
        vi = hex_vertices(positions[i,0], positions[i,1], s, angles[i])
        all_pts.append(vi)
    all_pts = np.vstack(all_pts)
    
    a = np.radians(outer_angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    px, py = all_pts[:,0], all_pts[:,1]
    lx = px*ca - py*sa
    ly = px*sa + py*ca
    
    c30, s30 = np.cos(np.pi/6), np.sin(np.pi/6)
    n = len(px)
    
    A_ub, b_ub = [], []
    ap = np.sqrt(3)/2
    
    for i in range(n):
        A_ub.extend([[1,0,-ap],[-1,0,-ap]])
        b_ub.extend([lx[i], -lx[i]])
        v1 = lx[i]*c30 + ly[i]*s30
        A_ub.extend([[c30,s30,-ap],[-c30,-s30,-ap]])
        b_ub.extend([v1, -v1])
        v2 = lx[i]*c30 - ly[i]*s30
        A_ub.extend([[c30,-s30,-ap],[-c30,s30,-ap]])
        b_ub.extend([v2, -v2])
    
    c = [0,0,1]
    bounds = [(None,None),(None,None),(0,None)]
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
    if res.success:
        return res.fun, res.x[0], res.x[1]
    return float('inf'), 0, 0

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0
    
    spacing = np.sqrt(3)
    vspacing = 1.5  # vertical spacing for hex lattice
    
    # Generate candidate configurations
    candidates = []
    
    # Various row arrangements summing to 12
    arrangements = [
        [3,4,3,2], [2,4,3,3], [4,4,4], [2,3,4,3], [3,3,3,3],
        [1,5,5,1], [2,4,4,2], [4,3,3,2], [3,5,4], [5,3,4],
        [2,2,4,4], [4,2,4,2], [1,4,4,3], [3,4,5], [5,4,3]
    ]
    
    for arr in arrangements:
        if sum(arr) != N: continue
        pos = []
        for row, count in enumerate(arr):
            offset = spacing/2 if row % 2 == 1 else 0
            for col in range(count):
                x = col * spacing + offset
                y = row * vspacing
                pos.append([x, y])
        if len(pos) == N:
            candidates.append(np.array(pos))
    
    angles = np.zeros(N)
    best_S = float('inf')
    best_pos = None
    best_cx, best_cy = 0, 0
    
    for pos in candidates:
        if time.time() > deadline: break
        # Center the configuration
        pos = pos - pos.mean(axis=0)
        S, cx, cy = outer_hex_side(pos, angles, 0.0)
        if S < best_S:
            best_S = S
            best_pos = pos.copy()
            best_cx, best_cy = cx, cy
    
    if best_pos is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N-N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Local optimization
    x0 = best_pos.flatten()
    
    def obj(x):
        positions = x.reshape(N, 2)
        for i in range(N):
            for j in range(i+1, N):
                if check_overlap(i, j, positions, angles):
                    return 1e6
        S, _, _ = outer_hex_side(positions, angles, 0.0)
        return S
    
    try:
        res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < best_S:
            best_S = res.fun
            best_pos = res.x.reshape(N, 2)
            _, best_cx, best_cy = outer_hex_side(best_pos, angles, 0.0)
    except:
        pass
    
    inner = [[best_pos[i,0], best_pos[i,1], 0.0] for i in range(N)]
    return inner, (best_cx, best_cy), best_S, 0.0
# EVOLVE-BLOCK-END
