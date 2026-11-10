import numpy as np
import time
import random

def fast_score(points):
    """
    points: numpy array of shape (N, 2)
    Returns: max multiplicity of a positive doubled area
    """
    n = points.shape[0]
    if n < 3:
        return 0
        
    # Vectorized calculation of determinants for all triples (i, j, k)
    p1 = points[:, None, None, :]
    p2 = points[None, :, None, :]
    p3 = points[None, None, :, :]
    
    det = (p2[..., 0] - p1[..., 0]) * (p3[..., 1] - p1[..., 1]) - \
          (p2[..., 1] - p1[..., 1]) * (p3[..., 0] - p1[..., 0])
          
    idx = np.indices((n, n, n))
    mask = (idx[0] < idx[1]) & (idx[1] < idx[2])
    
    valid_dets = np.abs(det[mask])
    valid_dets = valid_dets[valid_dets > 0]
    
    if len(valid_dets) == 0:
        return 0
        
    counts = np.bincount(valid_dets)
    return int(counts.max())

def construct():
    N = 30
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit (using 50s to be safe)
    
    candidates = []
    
    # Rectangular Grids W x H such that W*H = 30
    for W in [1, 2, 3, 5, 6, 10, 15, 30]:
        H = 30 // W
        if W * H != 30: continue
        pts = []
        for i in range(W):
            for j in range(H):
                x = i - W // 2
                y = j - H // 2
                pts.append((x, y))
        
        if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts):
            candidates.append(np.array(pts, dtype=int))
            
    # Add a "clustered" start: points close to each other
    for _ in range(5):
        pts = []
        base_x = random.randint(-20, 20)
        base_y = random.randint(-20, 20)
        while len(pts) < N:
            p = (base_x + random.randint(-8, 8), base_y + random.randint(-8, 8))
            if -30 <= p[0] <= 30 and -30 <= p[1] <= 30 and p not in pts:
                pts.append(p)
        candidates.append(np.array(pts, dtype=int))

    # Add random starts
    for _ in range(5):
        pts = []
        while len(pts) < N:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in pts:
                pts.append(p)
        candidates.append(np.array(pts, dtype=int))

    # Evaluate candidates and pick the best few for local search
    scored_candidates = []
    for cand in candidates:
        s = fast_score(cand)
        scored_candidates.append((s, cand))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_global_score = -1
    best_global_pts = None
    
    # Run local search on the top 3 initial candidates
    for initial_score, initial_pts in scored_candidates[:3]:
        current_pts = initial_pts.copy()
        current_score = initial_score
        
        if current_score > best_global_score:
            best_global_score = current_score
            best_global_pts = current_pts.copy()
            
        # Simulated Annealing Parameters
        T = 1.0
        T_min = 0.01
        alpha = 0.995
        no_improve_count = 0
        max_no_improve = 300
        
        while time.time() < deadline:
            if current_score > best_global_score:
                best_global_score = current_score
                best_global_pts = current_pts.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
            
            idx = np.random.randint(N)
            
            if random.random() < 0.7:
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                if dx == 0 and dy == 0:
                    continue
                new_p = current_pts[idx] + np.array([dx, dy])
            else:
                new_p = np.array([random.randint(-30, 30), random.randint(-30, 30)])
                
            if new_p[0] < -30 or new_p[0] > 30 or new_p[1] < -30 or new_p[1] > 30:
                continue
                
            is_distinct = True
            for i in range(N):
                if i != idx and np.array_equal(current_pts[i], new_p):
                    is_distinct = False
                    break
                    
            if not is_distinct:
                continue
                
            new_pts = current_pts.copy()
            new_pts[idx] = new_p
            
            new_score = fast_score(new_pts)
            
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
            else:
                delta = current_score - new_score
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_pts = new_pts
                        current_score = new_score
                elif random.random() < 0.5:
                    current_pts = new_pts
            
            T = max(T_min, T * alpha)

    # Final Polish Phase
    if best_global_pts is not None:
        current_pts = best_global_pts.copy()
        current_score = best_global_score
        
        T = 0.1
        T_min = 0.001
        alpha = 0.999
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline:
            if current_score > best_global_score:
                best_global_score = current_score
                best_global_pts = current_pts.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
            
            idx = np.random.randint(N)
            
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            if dx == 0 and dy == 0:
                continue
            new_p = current_pts[idx] + np.array([dx, dy])
                
            if new_p[0] < -30 or new_p[0] > 30 or new_p[1] < -30 or new_p[1] > 30:
                continue
                
            is_distinct = True
            for i in range(N):
                if i != idx and np.array_equal(current_pts[i], new_p):
                    is_distinct = False
                    break
                    
            if not is_distinct:
                continue
                
            new_pts = current_pts.copy()
            new_pts[idx] = new_p
            
            new_score = fast_score(new_pts)
            
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
            else:
                delta = current_score - new_score
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_pts = new_pts
                        current_score = new_score
                elif random.random() < 0.5:
                    current_pts = new_pts
            
            T = max(T_min, T * alpha)

    return [(int(p[0]), int(p[1])) for p in best_global_pts]
