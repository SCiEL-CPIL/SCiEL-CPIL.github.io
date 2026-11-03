import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 12

def hex_support(direction, angle_deg, side=1.0):
    """Support function of a regular hexagon (side=1) in a given unit direction."""
    a = np.radians(angle_deg)
    v_angles = a + np.arange(6) * np.pi / 3
    dx, dy = direction
    dots = side * (dx * np.cos(v_angles) + dy * np.sin(v_angles))
    return np.max(dots)

def min_outer_side(positions, angles_deg, outer_angle_deg):
    """Find min outer hex side S, center (cx,cy) for given inner hexagons and outer rotation."""
    theta = np.radians(outer_angle_deg)
    normal_angles = theta + np.pi/6 + np.arange(6) * np.pi/3
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    h = np.zeros((N, 6))
    for i in range(N):
        for k in range(6):
            h[i, k] = hex_support(normals[k], angles_deg[i], 1.0)
    
    pos_n = positions @ normals  # (N, 6)
    maxA = np.max(pos_n + h, axis=0)
    minB = np.min(pos_n - h, axis=0)
    
    sqrt3 = np.sqrt(3.0)
    A_ub = np.zeros((12, 3))
    b_ub = np.zeros(12)
    for k in range(6):
        nx, ny = normals[k]
        A_ub[2*k, 0] = -nx; A_ub[2*k, 1] = -ny; A_ub[2*k, 2] = sqrt3/2
        b_ub[2*k] = -maxA[k]
        A_ub[2*k+1, 0] = nx; A_ub[2*k+1, 1] = ny; A_ub[2*k+1, 2] = sqrt3/2
        b_ub[2*k+1] = -minB[k]
    
    res = linprog(c=[0,0,1], A_ub=A_ub, b_ub=b_ub, 
                  bounds=[(None,None),(None,None),(0,None)], method='highs')
    if res.success:
        return res.fun, res.x[0], res.x[1]
    return np.inf, 0.0, 0.0

def overlap_penalty(positions, angles_deg):
    pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            di = positions[j] - positions[i]
            dirs = set()
            for base_ang in [angles_deg[i], angles_deg[j]]:
                a = np.radians(base_ang)
                for k in range(6):
                    ang = a + k*np.pi/3 + np.pi/6
                    dirs.add(round(ang % (2*np.pi), 6))
            for ang in dirs:
                n = np.array([np.cos(ang), np.sin(ang)])
                hi = hex_support(n, angles_deg[i], 1.0)
                hj = hex_support(n, angles_deg[j], 1.0)
                proj = di @ n
                ov = hi + hj - abs(proj)
                if ov > 0:
                    pen += ov**2
    return pen

def cost(x):
    pos = x[:24].reshape(N, 2)
    angs = x[24:36]
    outer_ang = x[36]
    S, _, _ = min_outer_side(pos, angs, outer_ang)
    pen = overlap_penalty(pos, angs)
    return S + 50.0 * pen

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    
    e1 = np.array([np.sqrt(3), 0.0])
    e2 = np.array([np.sqrt(3)/2, 1.5])
    
    configs = []
    
    # 3x4 grid
    p1 = np.zeros((N,2)); a1 = np.zeros(N)
    idx=0
    for r in range(3):
        for c in range(4):
            p1[idx] = r*e2 + c*e1; idx+=1
    p1 -= p1.mean(axis=0)
    configs.append((p1, a1, 0.0))
    
    # Hex cluster: center+6+5
    p2 = np.zeros((N,2)); a2 = np.zeros(N)
    p2[0] = [0,0]
    for k in range(6):
        ang = k*np.pi/3
        p2[k+1] = np.sqrt(3)*np.array([np.cos(ang), np.sin(ang)])
    for k in range(5):
        ang = k*2*np.pi/5 + np.pi/6
        p2[k+7] = 2*np.sqrt(3)*np.array([np.cos(ang), np.sin(ang)])
    p2 -= p2.mean(axis=0)
    configs.append((p2, a2, 0.0))
    
    # 4x3 grid
    p3 = np.zeros((N,2)); a3 = np.zeros(N)
    idx=0
    for r in range(4):
        for c in range(3):
            p3[idx] = r*e1 + c*e2; idx+=1
    p3 -= p3.mean(axis=0)
    configs.append((p3, a3, 30.0))
    
    # 3x4 with 30deg rotation
    p4 = p1.copy(); a4 = np.full(N, 30.0)
    configs.append((p4, a4, 0.0))
    
    best_x = None
    best_cost = np.inf
    
    for pos0, ang0, oa0 in configs:
        if time.time() > deadline:
            break
        x0 = np.concatenate([pos0.flatten(), ang0, [oa0]])
        try:
            res = minimize(cost, x0, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol':1e-5, 'fatol':1e-5})
            if res.fun < best_cost:
                best_cost = res.fun
                best_x = res.x
        except:
            pass
    
    if best_x is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N-N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    bp = best_x[:24].reshape(N,2)
    ba = best_x[24:36]
    bo = best_x[36]
    S, cx, cy = min_outer_side(bp, ba, bo)
    
    if not np.isfinite(S) or S > 20:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N-N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = [[float(bp[i,0]), float(bp[i,1]), float(ba[i])] for i in range(N)]
    return inner, (float(cx), float(cy)), float(S), float(bo)
# EVOLVE-BLOCK-END
