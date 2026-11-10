import numpy as np
import time
import random
from collections import Counter

def fast_score(points):
    """
    points: numpy array of shape (N, 2)
    Returns: max multiplicity of a positive doubled area
    """
    n = points.shape[0]
    if n < 3:
        return 0
    
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

def incremental_score_update(points, moved_idx, old_pt, new_pt):
    """
    Fast incremental update: compute how the score changes when one point moves.
    Returns the new full score by recomputing only affected triples.
    """
    n = points.shape[0]
    # For a single point move, only triples involving that point change
    # There are C(n-1, 2) such triples
    others = [i for i in range(n) if i != moved_idx]
    
    # Compute all determinants for triples involving moved_idx
    new_det_list = []
    for i in range(len(others)):
        for j in range(i+1, len(others)):
            a = others[i]
            b = others[j]
            # det of (new_pt, points[a], points[b])
            x1, y1 = new_pt
            x2, y2 = points[a]
            x3, y3 = points[b]
            d = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
            if d > 0:
                new_det_list.append(d)
    
    # Also need to know the old determinants to subtract
    old_det_list = []
    for i in range(len(others)):
        for j in range(i+1, len(others)):
            a = others[i]
            b = others[j]
            x1, y1 = old_pt
            x2, y2 = points[a]
            x3, y3 = points[b]
            d = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
            if d > 0:
                old_det_list.append(d)
    
    # Full score: we need the global max. This is expensive to do incrementally
    # without maintaining a full histogram. For now, just do full recompute
    # but this function structure allows future optimization.
    return None  # Not used in final version, kept for clarity

def construct():
    N = 30
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    def generate_line_config(n_points, x_start, x_end, y, extra_pts):
        """Generate points on a horizontal line plus some extras."""
        pts = []
        if x_end - x_start + 1 >= n_points:
            for i in range(n_points):
                pts.append((x_start + i, y))
        else:
            # Spread out
            for i in range(n_points):
                x = x_start + int(i * (x_end - x_start) / max(1, n_points - 1))
                pts.append((x, y))
        
        # Add extra points off the line
        for p in extra_pts:
            if p not in pts:
                pts.append(p)
        
        # If still need more, add random
        while len(pts) < N:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in pts:
                pts.append(p)
        
        return pts[:N]
    
    def generate_grid_config(W, H, x_off, y_off):
        """Generate a W x H grid."""
        pts = []
        for i in range(W):
            for j in range(H):
                pts.append((x_off + i, y_off + j))
        # If W*H < N, add more
        while len(pts) < N:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in pts:
                pts.append(p)
        return pts[:N]
    
    def generate_2d_lattice_config(n1, n2, dx1, dy1, dx2, dy2, x0, y0):
        """Generate points on a 2D lattice."""
        pts = []
        for i in range(n1):
            for j in range(n2):
                x = x0 + i*dx1 + j*dx2
                y = y0 + i*dy1 + j*dy2
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        while len(pts) < N:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in pts:
                pts.append(p)
        return pts[:N]
    
    candidates = []
    
    # Strategy 1: Points on a line (1D arithmetic progression)
    # 25 points on a line + 5 off-line
    for y_line in range(-25, 26):
        for x_start in range(-30, 31 - 24):
            pts = []
            for i in range(25):
                pts.append((x_start + i, y_line))
            # Add 5 off-line points
            for _ in range(5):
                x = random.randint(-30, 30)
                y = y_line + random.choice([-2, -1, 1, 2, 3, 5])
                if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in pts:
                    pts.append((x, y))
            while len(pts) < N:
                p = (random.randint(-30, 30), random.randint(-30, 30))
                if p not in pts:
                    pts.append(p)
            if len(pts) == N:
                candidates.append(np.array(pts[:N], dtype=int))
    
    # Strategy 2: 20 points on a line + 10 off-line
    for y_line in range(-25, 26):
        for x_start in range(-30, 31 - 19):
            pts = []
            for i in range(20):
                pts.append((x_start + i, y_line))
            for _ in range(10):
                x = random.randint(-30, 30)
                y = y_line + random.choice([-2, -1, 1, 2, 3, 4, 5])
                if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in pts:
                    pts.append((x, y))
            while len(pts) < N:
                p = (random.randint(-30, 30), random.randint(-30, 30))
                if p not in pts:
                    pts.append(p)
            if len(pts) == N:
                candidates.append(np.array(pts[:N], dtype=int))
    
    # Strategy 3: Grid configurations
    for W in [2, 3, 5, 6, 10, 15]:
        H = N // W
        if W * H != N:
            continue
        for x_off in range(-30, 31 - W + 1):
            for y_off in range(-30, 31 - H + 1):
                if x_off + W - 1 > 30 or y_off + H - 1 > 30:
                    continue
                pts = generate_grid_config(W, H, x_off, y_off)
                if len(pts) == N:
                    candidates.append(np.array(pts, dtype=int))
    
    # Strategy 4: 2D lattice with small steps
    for dx1 in [1, 2, 3]:
        for dy1 in [0, 1, 2]:
            for dx2 in [0, 1, 2, 3]:
                for dy2 in [1, 2, 3]:
                    if dx1 == dx2 and dy1 == dy2:
                        continue
                    n1 = 5
                    n2 = 6
                    x0 = random.randint(-25, 25)
                    y0 = random.randint(-25, 25)
                    pts = generate_2d_lattice_config(n1, n2, dx1, dy1, dx2, dy2, x0, y0)
                    if len(pts) == N:
                        candidates.append(np.array(pts, dtype=int))
    
    # Strategy 5: Random starts
    for _ in range(10):
        pts = []
        while len(pts) < N:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in pts:
                pts.append(p)
        candidates.append(np.array(pts, dtype=int))
    
    # Strategy 6: Clustered starts
    for _ in range(5):
        base_x = random.randint(-15, 15)
        base_y = random.randint(-15, 15)
        pts = []
        while len(pts) < N:
            p = (base_x + random.randint(-5, 5), base_y + random.randint(-5, 5))
            if -30 <= p[0] <= 30 and -30 <= p[1] <= 30 and p not in pts:
                pts.append(p)
        candidates.append(np.array(pts, dtype=int))
    
    # Evaluate all candidates
    scored_candidates = []
    for cand in candidates:
        s = fast_score(cand)
        scored_candidates.append((s, cand))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_global_score = -1
    best_global_pts = None
    
    # Run local search on top candidates
    top_k = min(5, len(scored_candidates))
    for initial_score, initial_pts in scored_candidates[:top_k]:
        current_pts = initial_pts.copy()
        current_score = initial_score
        
        if current_score > best_global_score:
            best_global_score = current_score
            best_global_pts = current_pts.copy()
        
        # Simulated Annealing
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
            
            # 70% local, 30% global
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
    
    # Final polish phase
    if best_global_pts is not None:
        current_pts = best_global_pts.copy()
        current_score = best_global_score
        
        T = 0.05
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
