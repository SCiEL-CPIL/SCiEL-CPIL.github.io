import numpy as np
import time
import random
from collections import Counter

def construct():
    N = 30
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    def score(points):
        """
        points: list of tuples (x, y)
        Returns: max multiplicity of a positive doubled area
        """
        n = len(points)
        if n < 3:
            return 0
        
        # Use numpy for fast computation
        P = np.array(points, dtype=np.int64)
        
        # Compute all determinants for triples (i, j, k) with i < j < k
        dets = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    det = abs((P[j][0] - P[i][0]) * (P[k][1] - P[i][1]) - 
                              (P[j][1] - P[i][1]) * (P[k][0] - P[i][0]))
                    if det > 0:
                        dets.append(det)
        
        if not dets:
            return 0
            
        counts = Counter(dets)
        return max(counts.values())
    
    def incremental_score(points):
        """
        Calculate score and return counts dict for incremental updates.
        """
        n = len(points)
        counts = {}
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    det = abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) - 
                              (points[j][1] - points[i][1]) * (points[k][0] - points[i][0]))
                    if det > 0:
                        counts[det] = counts.get(det, 0) + 1
        
        max_score = max(counts.values()) if counts else 0
        return max_score, counts
    
    def update_counts(points, idx, old_pt, new_pt, counts):
        """
        Update counts after moving point at idx from old_pt to new_pt.
        Returns new max score.
        """
        n = len(points)
        
        # For each pair (i, j) with i < j, i != idx, j != idx:
        # The triple (i, j, idx) changes
        for i in range(n):
            if i == idx:
                continue
            for j in range(i+1, n):
                if j == idx:
                    continue
                
                # Calculate area with old_pt
                p1 = points[i]
                p2 = points[j]
                det_old = abs((p2[0] - p1[0]) * (old_pt[1] - p1[1]) - 
                              (p2[1] - p1[1]) * (old_pt[0] - p1[0]))
                if det_old > 0:
                    counts[det_old] -= 1
                    if counts[det_old] == 0:
                        del counts[det_old]
                
                # Calculate area with new_pt
                det_new = abs((p2[0] - p1[0]) * (new_pt[1] - p1[1]) - 
                              (p2[1] - p1[1]) * (new_pt[0] - p1[0]))
                if det_new > 0:
                    counts[det_new] = counts.get(det_new, 0) + 1
                    
        return max(counts.values()) if counts else 0

    # 1. Generate Initial Candidates
    candidates = []
    
    # Rectangular Grids with various spacings
    factors = [(1, 30), (2, 15), (3, 10), (5, 6), (6, 5), (10, 3), (15, 2), (30, 1)]
    for W, H in factors:
        max_sx = 60 // (W - 1) if W > 1 else 60
        max_sy = 60 // (H - 1) if H > 1 else 60
        
        for sx in range(1, max_sx + 1):
            for sy in range(1, max_sy + 1):
                grid_width = (W - 1) * sx
                grid_height = (H - 1) * sy
                
                if grid_width <= 60 and grid_height <= 60:
                    # Center the grid
                    offset_x = -grid_width // 2
                    offset_y = -grid_height // 2
                    pts = []
                    for i in range(W):
                        for j in range(H):
                            x = i * sx + offset_x
                            y = j * sy + offset_y
                            if -30 <= x <= 30 and -30 <= y <= 30:
                                pts.append((x, y))
                    if len(pts) == N:
                        candidates.append(pts)
    
    # Add some random starts
    for _ in range(5):
        pts = []
        while len(pts) < N:
            p = (random.randint(-30, 30), random.randint(-30, 30))
            if p not in pts:
                pts.append(p)
        candidates.append(pts)
    
    # Add "hexagonal-like" patterns
    for r in [3, 5, 7, 10]:
        pts = [(0, 0)]
        for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
            x = int(round(r * np.cos(angle)))
            y = int(round(r * np.sin(angle)))
            if (x, y) not in pts and -30 <= x <= 30 and -30 <= y <= 30:
                pts.append((x, y))
        # Fill remaining
        while len(pts) < N:
            p = (random.randint(-r-2, r+2), random.randint(-r-2, r+2))
            if -30 <= p[0] <= 30 and -30 <= p[1] <= 30 and p not in pts:
                pts.append(p)
        candidates.append(pts)
    
    # Evaluate candidates and pick the best few for local search
    scored_candidates = []
    for cand in candidates:
        s = score(cand)
        scored_candidates.append((s, cand))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_global_score = -1
    best_global_pts = None
    
    # Run local search on the top 5 initial candidates
    for initial_score, initial_pts in scored_candidates[:5]:
        current_pts = initial_pts[:]
        current_score, counts = incremental_score(current_pts)
        
        if current_score > best_global_score:
            best_global_score = current_score
            best_global_pts = current_pts[:]
            
        # Simulated Annealing Parameters
        T = 1.0
        T_min = 0.01
        alpha = 0.9995  # Slower cooling
        no_improve_count = 0
        max_no_improve = 500
        
        point_set = set(current_pts)
        
        while time.time() < deadline:
            if current_score > best_global_score:
                best_global_score = current_score
                best_global_pts = current_pts[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
            
            # Pick a random point to move
            idx = random.randint(0, N-1)
            old_pt = current_pts[idx]
            
            # Decide move type: 70% local, 30% global
            if random.random() < 0.7:
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                if dx == 0 and dy == 0:
                    continue
                new_p = (old_pt[0] + dx, old_pt[1] + dy)
            else:
                new_p = (random.randint(-30, 30), random.randint(-30, 30))
                
            if new_p[0] < -30 or new_p[0] > 30 or new_p[1] < -30 or new_p[1] > 30:
                continue
                
            if new_p in point_set:
                continue
                
            # Calculate new score incrementally
            new_score = update_counts(current_pts, idx, old_pt, new_p, counts)
            
            if new_score > current_score:
                point_set.remove(old_pt)
                point_set.add(new_p)
                current_pts[idx] = new_p
                current_score = new_score
            else:
                delta = current_score - new_score
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        point_set.remove(old_pt)
                        point_set.add(new_p)
                        current_pts[idx] = new_p
                        current_score = new_score
                elif random.random() < 0.3:
                    point_set.remove(old_pt)
                    point_set.add(new_p)
                    current_pts[idx] = new_p
                    current_score = new_score
            
            T = max(T_min, T * alpha)
        
        # After SA, do a final hill-climbing pass
        for _ in range(3):
            improved = False
            for idx in range(N):
                if time.time() > deadline - 5:
                    break
                ox, oy = current_pts[idx]
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        nx = ox + dx
                        ny = oy + dy
                        if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                            continue
                        if (nx, ny) in point_set:
                            continue
                        
                        new_p = (nx, ny)
                        new_score = update_counts(current_pts, idx, (ox, oy), new_p, counts)
                        
                        if new_score > current_score:
                            point_set.remove((ox, oy))
                            point_set.add(new_p)
                            current_pts[idx] = new_p
                            current_score = new_score
                            improved = True
                            if current_score > best_global_score:
                                best_global_score = current_score
                                best_global_pts = current_pts[:]
                        else:
                            # Revert
                            pass # Already reverted in logic above by not applying
            if not improved:
                break

    return best_global_pts
