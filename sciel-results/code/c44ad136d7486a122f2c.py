"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin before 120s limit
    
    n_opt = N - 2

    def objective_reduced(x):
        pts = np.zeros((N, D))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2:] = x.reshape(n_opt, D)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        sq_dists = sq_dists[np.triu_indices(N, k=1)]
        d_min_sq = np.min(sq_dists)
        d_max_sq = np.max(sq_dists)
        if d_min_sq < 1e-12:
            return 1e9
        return (d_max_sq / d_min_sq)

    def normalize_and_fix(pts):
        """Normalize points: fix first two points at [0,0] and [1,0]"""
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        
        # Find closest pair
        min_d = 1e9
        p1, p2 = 0, 1
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_d:
                    min_d = d
                    p1, p2 = i, j
                    
        pts = pts - pts[p1]
        dx, dy = pts[p2]
        d = np.sqrt(dx**2 + dy**2)
        pts = pts / d
        theta = np.arctan2(dy, dx)
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
        
        mask = np.ones(len(pts), dtype=bool)
        mask[p1] = False
        mask[p2] = False
        return pts[mask].flatten()

    def make_hex_init(rotation_angle=0.0):
        hex_pts = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i + 0.5 * j
                y = np.sqrt(3) / 2 * j
                hex_pts.append([x, y])
        hex_pts = np.array(hex_pts)
        
        if rotation_angle != 0:
            c, s = np.cos(rotation_angle), np.sin(rotation_angle)
            R = np.array([[c, -s], [s, c]])
            hex_pts = hex_pts @ R.T
        
        dists_to_origin = np.sum(hex_pts**2, axis=1)
        idx = np.argsort(dists_to_origin)[:N]
        pts = hex_pts[idx]
        
        return normalize_and_fix(pts)

    def make_concentric_circles_init():
        # 1 center + 5 inner ring + 10 outer ring
        pts = []
        pts.append([0, 0])
        for k in range(5):
            angle = 2 * np.pi * k / 5
            pts.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
        for k in range(10):
            angle = 2 * np.pi * k / 10 + np.pi/10
            pts.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
        
        pts = np.array(pts)
        return normalize_and_fix(pts)

    best_x = None
    best_score = 1e9
    
    # Generate diverse initializations
    inits = []
    
    # Hexagonal lattice with different rotations
    for angle in [0, np.pi/12, np.pi/6, np.pi/4, np.pi/3, np.pi/5, np.pi/7]:
        x0 = make_hex_init(rotation_angle=angle)
        inits.append(x0)
    
    # Concentric circles
    inits.append(make_concentric_circles_init())
    
    # Random inits
    for _ in range(8):
        x0_rand = np.random.uniform(-2, 2, (n_opt, D)).flatten()
        inits.append(x0_rand)
    
    # Evaluate and sort by initial score
    scored_inits = []
    for x_init in inits:
        try:
            score = objective_reduced(x_init)
            scored_inits.append((score, x_init))
        except:
            pass
    scored_inits.sort(key=lambda t: t[0])
    
    # Optimize each init with Nelder-Mead
    for score, x_init in scored_inits:
        if time.time() > deadline:
            break
        
        # Add small noise to break symmetry
        x_noisy = x_init + np.random.normal(0, 0.005, x_init.shape)
        
        try:
            res = scipy.optimize.minimize(objective_reduced, x_noisy, method='Nelder-Mead',
                                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8,
                                                   'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final polish with Powell if time permits
    if best_x is not None and time.time() < deadline - 10:
        try:
            res = scipy.optimize.minimize(objective_reduced, best_x, method='Powell',
                                          options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    if best_x is None:
        x0_hex = make_hex_init(rotation_angle=0.0)
        best_x = x0_hex
    
    pts = np.zeros((N, D))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2:] = best_x.reshape(n_opt, D)
    
    return pts
# EVOLVE-BLOCK-END
