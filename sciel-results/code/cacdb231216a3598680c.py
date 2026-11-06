import numpy as np
import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = 80.0  # 10s safety margin
    n = 30
    
    def build_full_hist(pts):
        h = Counter()
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    pi = pts[i]
                    pj = pts[j]
                    pk = pts[k]
                    det = (pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0])
                    a = abs(det)
                    if a > 0:
                        h[a] += 1
        return h
    
    def get_areas_involving(idx, pts):
        areas = []
        pi = pts[idx]
        for j in range(n):
            if j == idx:
                continue
            for k in range(j+1, n):
                if k == idx:
                    continue
                pj = pts[j]
                pk = pts[k]
                det = (pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0])
                a = abs(det)
                if a > 0:
                    areas.append(a)
        return areas
    
    def try_move_sa(idx, new_pos, hist, pts, current_score, temperature):
        # Remove old areas
        old_areas = get_areas_involving(idx, pts)
        for a in old_areas:
            hist[a] -= 1
            if hist[a] == 0:
                del hist[a]
        
        old_pos = pts[idx]
        pts[idx] = new_pos
        
        # Add new areas
        new_areas = get_areas_involving(idx, pts)
        for a in new_areas:
            hist[a] += 1
        
        new_score = max(hist.values()) if hist else 0
        
        if new_score > current_score:
            return True, new_score
        elif temperature > 0 and random.random() < np.exp((new_score - current_score) / max(temperature, 1e-10)):
            return True, new_score
        else:
            # Revert
            for a in new_areas:
                hist[a] -= 1
                if hist[a] == 0:
                    del hist[a]
            pts[idx] = old_pos
            for a in old_areas:
                hist[a] += 1
            return False, current_score
    
    def hill_climb(pts, hist, time_limit, use_sa=True, initial_temp=10.0):
        best_score = max(hist.values()) if hist else 0
        best_pts = pts[:]
        best_hist = hist.copy()
        temperature = initial_temp if use_sa else 0
        cooling = 0.95
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            random.shuffle(range(n))  # Randomize point order
            for idx in range(n):
                if (time.time() - start_time) > time_limit:
                    break
                orig_x, orig_y = best_pts[idx]
                # Neighborhood: try moves in expanding radius
                for radius in [1, 2, 3]:
                    for dx in range(-radius, radius+1):
                        for dy in range(-radius, radius+1):
                            if dx == 0 and dy == 0:
                                continue
                            new_x = orig_x + dx
                            new_y = orig_y + dy
                            if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
                                continue
                            new_pos = (new_x, new_y)
                            if new_pos in best_pts:
                                continue
                            
                            accepted, new_score = try_move_sa(idx, new_pos, best_hist, best_pts, best_score, temperature)
                            if accepted:
                                best_score = new_score
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                
                if use_sa and temperature > 0.01:
                    temperature *= cooling
        
        return best_pts, best_hist, best_score
    
    def generate_starting_config(seed_type, seed_val=0):
        random.seed(seed_val)
        if seed_type == 0:
            # 6x5 grid
            return [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
        elif seed_type == 1:
            # Sheared grid
            pts = []
            for i in range(6):
                for j in range(5):
                    x = -2 + i + j
                    y = -2 + j
                    if -30 <= x <= 30 and -30 <= y <= 30:
                        pts.append((x, y))
            return pts[:30]
        elif seed_type == 2:
            # Clustered random
            centers = [(0, 0), (5, 5), (-5, 5), (5, -5)]
            pts = []
            for cx, cy in centers:
                for _ in range(8):
                    x = cx + random.randint(-2, 2)
                    y = cy + random.randint(-2, 2)
                    if -30 <= x <= 30 and -30 <= y <= 30 and (x, y) not in pts:
                        pts.append((x, y))
            while len(pts) < 30:
                x = random.randint(-10, 10)
                y = random.randint(-10, 10)
                if (x, y) not in pts:
                    pts.append((x, y))
            return pts
        elif seed_type == 3:
            # Dense small grid
            pts = []
            for x in range(-3, 4):
                for y in range(-3, 4):
                    if -30 <= x <= 30 and -30 <= y <= 30:
                        pts.append((x, y))
            random.shuffle(pts)
            return pts[:30]
        else:
            # Random
            pts = []
            while len(pts) < 30:
                x = random.randint(-15, 15)
                y = random.randint(-15, 15)
                if (x, y) not in pts:
                    pts.append((x, y))
            return pts
    
    # Multi-start with different configurations
    best_overall_pts = None
    best_overall_score = 0
    best_overall_hist = None
    
    seed_types = [0, 1, 2, 3, 4]
    num_starts = 5
    
    for s in range(num_starts):
        if (time.time() - start_time) > deadline:
            break
        
        seed_type = seed_types[s % len(seed_types)]
        pts = generate_starting_config(seed_type, seed_val=s*100+42)
        
        # Ensure we have exactly 30 distinct points
        if len(pts) != 30 or len(set(pts)) != 30:
            continue
        
        hist = build_full_hist(pts)
        score = max(hist.values()) if hist else 0
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_pts = pts[:]
            best_overall_hist = hist.copy()
        
        # Hill climb with SA
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 2.0:
            break
        
        # Allocate time proportionally
        time_alloc = remaining_time / (num_starts - s)
        result_pts, result_hist, result_score = hill_climb(pts, hist, start_time + time_alloc, use_sa=True, initial_temp=5.0)
        
        if result_score > best_overall_score:
            best_overall_score = result_score
            best_overall_pts = result_pts[:]
            best_overall_hist = result_hist.copy()
    
    # Final polish pass with wider neighborhood
    if best_overall_pts is not None and (time.time() - start_time) < deadline - 1.0:
        polish_time = deadline - (time.time() - start_time) - 1.0
        if polish_time > 2.0:
            result_pts, result_hist, result_score = hill_climb(
                best_overall_pts, 
                best_overall_hist.copy(), 
                start_time + polish_time,
                use_sa=False  # Pure greedy for final polish
            )
            if result_score > best_overall_score:
                best_overall_pts = result_pts
                best_overall_score = result_score
    
    return best_overall_pts
