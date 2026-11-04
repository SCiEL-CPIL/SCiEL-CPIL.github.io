import numpy as np
from scipy.optimize import minimize

def construct():
    N = 12
    
    # Initial positions: 3x4 grid
    centers = []
    for i in range(-1, 2): # -1, 0, 1
        for j in range(-1, 3): # -1, 0, 1, 2
            x = i * np.sqrt(3)
            y = 2 * j + (1 if i % 2 != 0 else 0)
            centers.append([x, y, 0.0]) # x, y, angle
            
    centers = np.array(centers)
    centers[:, 0] -= np.mean(centers[:, 0])
    centers[:, 1] -= np.mean(centers[:, 1])
    
    base_angles = np.deg2rad([90, 150, 210, 270, 330, 30])
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    def get_hex_vertices(cx, cy, angle_deg):
        rot = np.deg2rad(angle_deg)
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        return R @ base_verts.T + np.array([cx, cy])
        
    def min_dist_poly(poly1, poly2):
        min_d = float('inf')
        for i in range(6):
            p1 = poly1[i]
            for j in range(6):
                p2_start = poly2[j]
                p2_end = poly2[(j+1)%6]
                v = p2_end - p2_start
                w = p1 - p2_start
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2_start)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p2_end)
                else:
                    t = c1 / c2
                    proj = p2_start + t * v
                    d = np.linalg.norm(p1 - proj)
                if d < min_d:
                    min_d = d
        for i in range(6):
            p2 = poly2[i]
            for j in range(6):
                p1_start = poly1[j]
                p1_end = poly1[(j+1)%6]
                v = p1_end - p1_start
                w = p2 - p1_start
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p2 - p1_start)
                elif c1 >= c2:
                    d = np.linalg.norm(p2 - p1_end)
                else:
                    t = c1 / c2
                    proj = p1_start + t * v
                    d = np.linalg.norm(p2 - proj)
                if d < min_d:
                    min_d = d
        return min_d

    def objective(params):
        xs = params[0:12]
        ys = params[12:24]
        ths = params[24:36]
        Phi = params[36]
        
        all_verts = []
        hex_verts_list = []
        for i in range(12):
            verts = get_hex_vertices(xs[i], ys[i], ths[i])
            hex_verts_list.append(verts)
            all_verts.append(verts)
            
        all_verts = np.vstack(all_verts)
        
        max_proj = 0
        for k in range(6):
            alpha = np.deg2rad(Phi + k * 60)
            ca, sa = np.cos(alpha), np.sin(alpha)
            projs = all_verts[:, 0] * ca + all_verts[:, 1] * sa
            max_proj = max(max_proj, np.max(np.abs(projs)))
            
        L = (2 / np.sqrt(3)) * max_proj
        
        penalty = 0
        for i in range(12):
            for j in range(i+1, 12):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d_c = np.sqrt(dx*dx + dy*dy)
                if d_c < 2.0:
                    d_min = min_dist_poly(hex_verts_list[i], hex_verts_list[j])
                    if d_min < 0:
                        penalty += (-d_min) ** 2
                    elif d_min < 0.01:
                        penalty += (0.01 - d_min) ** 2 * 0.1
                        
        return L + 10.0 * penalty

    x0 = np.zeros(37)
    x0[0:12] = centers[:, 0]
    x0[12:24] = centers[:, 1]
    x0[24:36] = centers[:, 2]
    x0[36] = 0.0
    
    res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 1000, 'maxfev': 2000})
    
    best_params = res.x
    xs = best_params[0:12]
    ys = best_params[12:24]
    ths = best_params[24:36]
    Phi = best_params[36]
    
    all_verts = []
    for i in range(12):
        verts = get_hex_vertices(xs[i], ys[i], ths[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    max_proj = 0
    for k in range(6):
        alpha = np.deg2rad(Phi + k * 60)
        ca, sa = np.cos(alpha), np.sin(alpha)
        projs = all_verts[:, 0] * ca + all_verts[:, 1] * sa
        max_proj = max(max_proj, np.max(np.abs(projs)))
        
    final_L = (2 / np.sqrt(3)) * max_proj
    
    inner = []
    for i in range(12):
        inner.append([xs[i], ys[i], ths[i]])
        
    return inner, (0.0, 0.0), final_L, Phi
