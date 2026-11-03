import numpy as np
import time
import scipy.optimize

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Initial Configuration: Hexagonal Lattice
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    k = int(1.0 / s) + 2
    xs = np.arange(-k, k+1) * s
    points_list = []
    for j, y in enumerate(np.arange(-k, k+1) * s * np.sqrt(3)/2):
        offset = (s/2) if j % 2 else 0
        for x in xs:
            points_list.append([x + offset, y])
    points = np.array(points_list)
    
    radii = np.linalg.norm(points, axis=1)
    points = points[radii < 1.0]
    
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(p) < 1.0:
                points = np.vstack([points, p])
        
    points = points[:N]

    # 2. Simulated Annealing
    T = 0.005
    T_min = 1e-7
    cooling = 0.9999
    
    def get_score(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        radii = np.linalg.norm(pts, axis=1)
        min_bound = np.min(1.0 - radii)
        return min(min_pair, min_bound)

    current_score = get_score(points)
    best_score = current_score
    best_points = points.copy()
    
    # Pre-allocate for speed?
    
    while time.time() < deadline - 10.0: # Leave 10s for polish
        i = np.random.randint(N)
        
        # Adaptive step size
        # If score is improving, keep step small. If stuck, increase?
        # Simple: step proportional to T
        r_move = 0.05 * (T / 0.005)
        
        dx, dy = np.random.uniform(-r_move, r_move, 2)
        new_p = points[i].copy()
        new_p[0] += dx
        new_p[1] += dy
        
        if np.linalg.norm(new_p) >= 1.0:
            continue
            
        points[i] = new_p
        new_score = get_score(points)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = points.copy()
        else:
            delta = new_score - current_score
            prob = np.exp(delta / T)
            if np.random.rand() < prob:
                current_score = new_score
            else:
                points[i] = new_p - np.array([dx, dy])
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    # 3. Polish Phase
    if time.time() < deadline - 1.0:
        def soft_min_neg(X):
            X = X.reshape(N, 2)
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dists, np.inf)
            radii = np.linalg.norm(X, axis=1)
            bound_dists = 1.0 - radii
            
            all_dists = np.concatenate([dists.flatten(), bound_dists])
            all_dists = all_dists[np.isfinite(all_dists)]
            
            eps = 0.01
            # Minimize log(sum(exp(-d/eps)))
            # Equivalent to maximizing soft min
            val = np.log(np.sum(np.exp(-all_dists / eps)))
            return val
        
        x0 = best_points.flatten()
        
        try:
            res = scipy.optimize.minimize(
                soft_min_neg, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 50, 'ftol': 1e-10}
            )
            if res.success:
                new_pts = res.x.reshape(N, 2)
                # Verify validity
                if np.all(np.linalg.norm(new_pts, axis=1) < 1.0):
                    new_score = get_score(new_pts)
                    if new_score > best_score:
                        best_points = new_pts
                        best_score = new_score
        except Exception:
            pass

    return best_points
