import numpy as np
from scipy.optimize import minimize

def construct():
    N = 12
    
    # Generate initial centers from a hexagonal tiling
    # Pick the 12 closest hexes to the origin in a hex grid
    candidates = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            # Hex distance
            d = (abs(q) + abs(r) + abs(q + r)) / 2
            x = np.sqrt(3) * q + np.sqrt(3)/2 * r
            y = 1.5 * r
            candidates.append((d, x, y))
            
    candidates.sort(key=lambda x: x[0])
    centers = np.array([[c[1], c[2]] for c in candidates[:N]])
    centers -= np.mean(centers, axis=0)
    
    # Initial angles: 0
    angles = np.zeros(N)
    
    x0 = np.zeros(36)
    for i in range(N):
        x0[i*3] = centers[i, 0]
        x0[i*3+1] = centers[i, 1]
        x0[i*3+2] = 0.0
        
    base_angles = np.deg2rad(np.arange(0, 360, 60))
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    def get_inner_verts(params):
        verts_list = []
        for i in range(N):
            cx = params[i*3]
            cy = params[i*3+1]
            ang = params[i*3+2]
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s], [s, c]])
            v = (R @ base_verts.T).T + np.array([cx, cy])
            verts_list.append(v)
        return np.array(verts_list)
        
    def calc_L_from_verts(verts):
        x = verts[:, :, 0]
        y = verts[:, :, 1]
        term1 = x + y / np.sqrt(3)
        term2 = 2 * y / np.sqrt(3)
        term3 = -x + y / np.sqrt(3)
        term4 = -x - y / np.sqrt(3)
        term5 = -2 * y / np.sqrt(3)
        term6 = x - y / np.sqrt(3)
        L_points = np.max(np.stack([term1, term2, term3, term4, term5, term6]), axis=0)
        return np.max(L_points)

    def point_to_segment_dist(p, a, b):
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0.0, 1.0)
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    def is_point_in_poly(p, poly):
        signs = []
        for i in range(6):
            e1 = poly[i]
            e2 = poly[(i+1)%6]
            vec = e2 - e1
            vec_p = p - e1
            cross = vec[0]*vec_p[1] - vec[1]*vec_p[0]
            signs.append(np.sign(cross))
        if all(s > 0 for s in signs) or all(s < 0 for s in signs):
            return True
        return False

    def hex_min_dist(v1, v2):
        min_d = np.inf
        for i in range(6):
            for j in range(6):
                d = np.linalg.norm(v1[i] - v2[j])
                if d < min_d: min_d = d
        for i in range(6):
            for j in range(6):
                d = point_to_segment_dist(v1[i], v2[j], v2[(j+1)%6])
                if d < min_d: min_d = d
                d = point_to_segment_dist(v2[j], v1[i], v1[(i+1)%6])
                if d < min_d: min_d = d
        if min_d < 1e-9:
            if is_point_in_poly(v1[0], v2) or is_point_in_poly(v2[0], v1):
                return -min_d
            return 0.0
        return min_d

    def overlap_penalty(params):
        verts = get_inner_verts(params)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = hex_min_dist(verts[i], verts[j])
                if d < 0:
                    penalty += d**2
        return penalty

    def objective(params):
        verts = get_inner_verts(params)
        L = calc_L_from_verts(verts)
        pen = overlap_penalty(params)
        return L + 100.0 * pen
        
    res = minimize(objective, x0, method='Powell', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000})
    
    best_params = res.x
    best_L = calc_L_from_verts(get_inner_verts(best_params))
    
    inner = []
    for i in range(N):
        cx = best_params[i*3]
        cy = best_params[i*3+1]
        ang_deg = np.rad2deg(best_params[i*3+2])
        inner.append([cx, cy, ang_deg])
        
    return inner, (0.0, 0.0), best_L, 0.0
