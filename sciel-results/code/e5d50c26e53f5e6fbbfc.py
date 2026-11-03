import numpy as np
from scipy.optimize import minimize
import time

N = 11

def construct():
    deadline = time.time() + 140.0
    
    hex_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    unit_hex_verts = np.stack([np.cos(hex_angles), np.sin(hex_angles)], axis=1)
    
    def container_side(pos, ang, oc, oa):
        theta = np.radians(oa)
        c, s_ = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s_], [s_, c]])
        fn_angles = (30 + np.arange(6) * 60) * np.pi / 180
        fn = np.stack([np.cos(fn_angles), np.sin(fn_angles)], axis=1)
        fn = fn @ R.T
        
        max_proj = 0.0
        for i in range(N):
            theta_i = np.radians(ang[i])
            ci, si = np.cos(theta_i), np.sin(theta_i)
            Ri = np.array([[ci, -si], [si, ci]])
            verts = pos[i] + unit_hex_verts @ Ri.T
            verts_rel = verts - oc
            projs = verts_rel @ fn.T
            mp = np.max(projs)
            if mp > max_proj:
                max_proj = mp
        return 2.0 * max_proj / np.sqrt(3.0)
    
    hex_dist = np.sqrt(3.0)
    base = [(0.0, 0.0)]
    for k in range(6):
        a = k * 60 * np.pi / 180
        base.append((hex_dist * np.cos(a), hex_dist * np.sin(a)))
    
    ring2 = []
    for k in range(12):
        a = (30 + k * 30) * np.pi / 180
        ring2.append((2 * hex_dist * np.cos(a), 2 * hex_dist * np.sin(a)))
    
    configs = [
        base + ring2[:4],
        base + ring2[3:7],
        base + ring2[6:10],
        base + ring2[9:13 % 12],
    ]
    # Fix the last one
    configs[-1] = base + ring2[9:12] + ring2[0:1]
    
    best_inner = None
    best_oc = np.array([0.0, 0.0])
    best_s = float('inf')
    best_oa = 0.0
    
    for cfg in configs:
        if time.time() > deadline - 20:
            break
        pos0 = np.array(cfg, dtype=float)
        ang0 = np.zeros(N)
        
        def obj(params):
            if time.time() > deadline:
                return 1e10
            p = params[:2*N].reshape(N, 2)
            a = params[2*N:3*N]
            oc = params[3*N:3*N+2]
            oa = params[3*N+2]
            s = container_side(p, a, oc, oa)
            pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(p[i] - p[j])
                    if d < hex_dist:
                        pen += (hex_dist - d) ** 2
            return s + 50.0 * pen
        
        init = np.concatenate([pos0.flatten(), ang0, [0.0, 0.0], [0.0]])
        try:
            res = minimize(obj, init, method='L-BFGS-B',
                          options={'maxiter': 60, 'ftol': 1e-10})
            p = res.x[:2*N].reshape(N, 2)
            a = res.x[2*N:3*N]
            oc = res.x[3*N:3*N+2]
            oa = res.x[3*N+2]
            s = container_side(p, a, oc, oa)
            if s < best_s:
                best_s = s
                best_inner = [[p[i,0], p[i,1], np.degrees(a[i])] for i in range(N)]
                best_oc = oc.copy()
                best_oa = oa
        except:
            pass
    
    if best_inner is None:
        pos0 = np.array(base + ring2[:4], dtype=float)
        best_inner = [[pos0[i,0], pos0[i,1], 0.0] for i in range(N)]
        best_s = container_side(pos0, np.zeros(N), np.array([0.0,0.0]), 0.0)
    
    return best_inner, (float(best_oc[0]), float(best_oc[1])), float(best_s), float(np.degrees(best_oa))
