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

def compute_area_histogram(points):
    """Compute histogram of all positive doubled areas."""
    n = points.shape[0]
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
        return np.array([0])
        
    return np.bincount(valid_dets)

def incremental_score(points, moved_idx, old_pt, new_pt):
    """
    Fast incremental update: compute new score when one point moves.
    Returns new score.
    """
    n = points.shape[0]
    others = [i for i in range(n) if i != moved_idx]
    
    # Compute new determinants for triples involving moved_idx
    new_dets = []
    for i in range(len(others)):
        for j in range(i+1, len(others)):
            a = others[i]
            b = others[j]
            x1, y1 = new_pt
            x2, y2 = points[a]
            x3, y3 = points[b]
            d = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
            if d > 0:
                new_dets.append(d)
    
    # Compute old determinants
    old_dets = []
    for i in range(len(others)):
        for j in range(i+1, len(others)):
            a = others[i]
            b = others[j]
            x1, y1 = old_pt
            x2, y2 = points[a]
            x3, y3 = points[b]
            d = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
            if d > 0:
                old_dets.append(d)
    
    # We need to recompute the full histogram to get the max
    # This is still O(n^3) but we can use the fact that only O(n^2) entries change
    # For now, just do full recompute
    new_pts = points.copy()
    new_pts[moved_idx] = new_pt
    return fast_score(new_pts)

def construct():
    N = 30
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    def generate_arithmetic_line(n_points, x_start, y, step=1):
        """Generate points on an arithmetic progression line."""
        pts = []
        for i in range(n_points):
            x = x_start + i * step
            if -30 <= x <= 30:
                pts.append((x, y))
        return pts
    
    def generate_grid(W, H, x_off, y_off, sx=1, sy=1):
        """Generate a W x H grid with spacing."""
        pts = []
        for i in range(W):
            for j in range(H):
                x = x_off + i * sx
                y = y_off + j * sy
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        return pts
    
    def generate_lattice(n1, n2, v1, v2, origin):
        """Generate points on a 2D lattice."""
        pts = []
        x0, y0 = origin
        dx1, dy1 = v1
        dx2, dy2 = v2
        for i in range(n1):
            for j in range(n2):
                x = x0 + i*dx1 + j*dx2
                y = y0 + i*dy1 + j*dy2
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        return pts
    
    candidates = []
    
    # Strategy 1: Strong arithmetic progressions on a line
    # Many points on one line create many equal areas with off-line points
    for y_line in range(-28, 29):
        for step in [1, 2, 3]:
            for x_start in range(-30, 31):
                line_pts = generate_arithmetic_line(25, x_start, y_line, step)
                if len(line_pts) < 20:
                    continue
                # Add off-line points to create many equal areas
                for off_y in range(y_line - 3, y_line + 4):
                    if off_y == y_line:
                        continue
                    for off_x_start in range(-30, 31):
                        off_pts = []
                        for i in range(5):
                            x = off_x_start + i * step
                            if -30 <= x <= 30:
                                off_pts.append((x, off_y))
                        if len(off_pts) < 3:
                            continue
                        
                        all_pts = line_pts[:25] + off_pts[:5]
                        # Ensure distinctness
                        seen = set()
                        unique_pts = []
                        for p in all_pts:
                            if p not in seen:
                                seen.add(p)
                                unique_pts.append(p)
                        if len(unique_pts) == N:
                            candidates.append(np.array(unique_pts, dtype=int))
    
    # Strategy 2: Two parallel lines
    for y1 in range(-28, 29):
        for y2 in range(y1 + 1, 30):
            if abs(y2 - y1) > 10:
                continue
            for step in [1, 2]:
                for x_start in range(-30, 31):
                    line1 = generate_arithmetic_line(15, x_start, y1, step)
                    line2 = generate_arithmetic_line(15, x_start, y2, step)
                    if len(line1) < 15 or len(line2) < 15:
                        continue
                    all_pts = line1[:15] + line2[:15]
                    seen = set()
                    unique_pts = []
                    for p in all_pts:
                        if p not in seen:
                            seen.add(p)
                            unique_pts.append(p)
                    if len(unique_pts) == N:
                        candidates.append(np.array(unique_pts, dtype=int))
    
    # Strategy 3: Grid configurations
    for W in [2, 3, 5, 6, 10, 15]:
        H = N // W
        if W * H != N:
            continue
        for sx in [1, 2]:
            for sy in [1, 2]:
                for x_off in range(-30, 31):
                    for y_off in range(-30, 31):
                        pts = generate_grid(W, H, x_off, y_off, sx, sy)
                        if len(pts) == N:
                            candidates.append(np.array(pts, dtype=int))
    
    # Strategy 4: 2D lattice with various vectors
    for v1 in [(1,0), (2,0), (1,1), (2,1), (3,1)]:
        for v2 in [(0,1), (0,2), (1,1), (1,2), (1,3)]:
            if v1 == v2:
                continue
            for n1 in [5, 6]:
                for n2 in [5, 6]:
                    if n1 * n2 != N:
                        continue
                    for x0 in range(-25, 26):
                        for y0 in range(-25, 26):
                            pts = generate_lattice(n1, n2, v1, v2, (x0, y0))
                            if len(pts) == N:
                                candidates.append(np.array(pts, dtype=int))
    
    # Strategy 5: Random starts
    for _ in range(20):
        pts = []
        while len(pts) < N:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in pts:
                pts.append(p)
        candidates.append(np.array(pts, dtype=int))
    
    # Strategy 6: Clustered starts
    for _ in range(10):
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
    top_k = min(8, len(scored_candidates))
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
        max_no_improve = 400
        
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
            
            # 60% local, 40% global
            if random.random() < 0.6:
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
    
    # Final polish phase with tighter moves
    if best_global_pts is not None:
        current_pts = best_global_pts.copy()
        current_score = best_global_score
        
        T = 0.05
        T_min = 0.001
        alpha = 0.999
        no_improve_count = 0
        max_no_improve = 600
        
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
