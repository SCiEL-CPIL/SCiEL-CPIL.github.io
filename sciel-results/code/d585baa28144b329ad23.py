import time
import random
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit

    # Strategy:
    # 1. Generate diverse candidate configurations (grids, sheared, staggered, random)
    # 2. Use vectorized area counting to quickly evaluate candidates
    # 3. Local search with efficient delta evaluation
    
    def evaluate_fast(points):
        """Vectorized evaluation of max area multiplicity."""
        n = len(points)
        if n < 3:
            return 0, 0
        pts = np.array(points, dtype=np.int32)
        
        # Compute all triple determinants efficiently
        areas = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    det = abs((pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) - 
                             (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0]))
                    if det > 0:
                        areas.append(det)
        
        if not areas:
            return 0, 0
        
        counter = Counter(areas)
        max_mult = max(counter.values())
        max_area = max(counter, key=counter.get)
        return max_mult, max_area
    
    # Generate candidates
    candidates = []
    
    # Standard grids
    for w, h in [(6, 5), (5, 6), (3, 10), (10, 3), (2, 15), (15, 2)]:
        pts = [(x, y) for x in range(w) for y in range(h)]
        if len(pts) == 30 and len(set(pts)) == 30:
            cx, cy = w // 2, h // 2
            pts = [(x - cx, y - cy) for x, y in pts]
            candidates.append(pts)
    
    # Sheared grids
    for w, h in [(6, 5), (5, 6), (3, 10)]:
        for k in [1, -1, 2, -2, 3, -3]:
            pts = []
            valid = True
            for x in range(w):
                for y in range(h):
                    px = x + k * y
                    py = y
                    if not (-30 <= px <= 30 and -30 <= py <= 30):
                        valid = False
                        break
                    pts.append((px, py))
                if not valid:
                    break
            if valid and len(pts) == 30 and len(set(pts)) == 30:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx = (min(xs) + max(xs)) // 2
                cy = (min(ys) + max(ys)) // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                candidates.append(pts)
    
    # Staggered patterns
    for w, h in [(6, 5), (5, 6)]:
        for offset in [0, 1, 2]:
            pts = []
            valid = True
            for y in range(h):
                off = (offset + y) % 2
                for x in range(w):
                    px = x + off
                    py = y
                    if not (-30 <= px <= 30 and -30 <= py <= 30):
                        valid = False
                        break
                    pts.append((px, py))
                if not valid:
                    break
            if valid and len(pts) == 30 and len(set(pts)) == 30:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx = (min(xs) + max(xs)) // 2
                cy = (min(ys) + max(ys)) // 2
                pts = [(x - cx, y - cy) for x, y in pts]
                candidates.append(pts)
    
    # Random configurations
    for _ in range(80):
        if time.time() > deadline - 15:
            break
        pts = set()
        attempts = 0
        while len(pts) < 30 and attempts < 200:
            pts.add((random.randint(-25, 25), random.randint(-25, 25)))
            attempts += 1
        if len(pts) == 30:
            candidates.append(list(pts))
    
    # Evaluate all candidates
    best_score = 0
    best_points = candidates[0] if candidates else [(0, 0)] * 30
    best_area = 1
    
    for pts in candidates:
        if time.time() > deadline - 10:
            break
        score, area = evaluate_fast(pts)
        if score > best_score:
            best_score = score
            best_points = list(pts)
            best_area = area
    
    # Local search with incremental area tracking
    current_points = list(best_points)
    current_score = best_score
    current_area = best_area
    
    # Build area counter
    def build_area_counter(points):
        n = len(points)
        counter = Counter()
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if det > 0:
                        counter[det] += 1
        return counter
    
    area_counter = build_area_counter(current_points)
    
    # Delta calculation for moving point idx from old to new
    def calc_delta(points, idx, old_pos, new_pos, target_area):
        delta = 0
        x1, y1 = new_pos
        ox, oy = old_pos
        n = len(points)
        
        for j in range(n):
            if j == idx:
                continue
            x2, y2 = points[j]
            for k in range(j + 1, n):
                if k == idx:
                    continue
                x3, y3 = points[k]
                det_new = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if det_new == target_area:
                    delta += 1
                det_old = abs((x2 - ox) * (y3 - oy) - (y2 - oy) * (x3 - ox))
                if det_old == target_area:
                    delta -= 1
        return delta
    
    # Local search loop
    steps = 0
    no_improve = 0
    max_no_improve = 300
    
    while time.time() < deadline - 2:
        steps += 1
        idx = random.randint(0, 29)
        old_x, old_y = current_points[idx]
        
        best_move = None
        best_delta = 0
        
        # Try moves in neighborhood
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = old_x + dx, old_y + dy
                if not (-30 <= nx <= 30 and -30 <= ny <= 30):
                    continue
                new_pos = (nx, ny)
                if new_pos in current_points:
                    continue
                
                delta = calc_delta(current_points, idx, (old_x, old_y), new_pos, current_area)
                if delta > best_delta:
                    best_delta = delta
                    best_move = new_pos
        
        if best_move:
            current_points[idx] = best_move
            # Update area counter
            area_counter = build_area_counter(current_points)
            current_score, current_area = max(area_counter.values()), max(area_counter, key=area_counter.get)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                # Reset and try bigger jumps
                current_points = list(best_points)
                current_score = best_score
                current_area = best_area
                area_counter = build_area_counter(current_points)
                no_improve = 0
                
                # Try random big jumps
                for _ in range(50):
                    if time.time() > deadline - 2:
                        break
                    idx2 = random.randint(0, 29)
                    ox2, oy2 = current_points[idx2]
                    for _ in range(10):
                        nx2, ny2 = random.randint(-30, 30), random.randint(-30, 30)
                        if (nx2, ny2) in current_points:
                            continue
                        delta = calc_delta(current_points, idx2, (ox2, oy2), (nx2, ny2), current_area)
                        if delta > 0:
                            current_points[idx2] = (nx2, ny2)
                            area_counter = build_area_counter(current_points)
                            new_score, new_area = max(area_counter.values()), max(area_counter, key=area_counter.get)
                            if new_score > current_score:
                                current_score, current_area = new_score, new_area
                            break
    
    # Final evaluation
    final_score, final_area = evaluate_fast(current_points)
    if final_score > best_score:
        best_score = final_score
        best_points = current_points
    
    return best_points

def _calculate_max_multiplicity(points):
    """Calculate the maximum multiplicity of any positive doubled area."""
    n = len(points)
    if n < 3:
        return 0, 0
    areas = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            for k in range(j + 1, n):
                x3, y3 = points[k]
                det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if det > 0:
                    areas[det] = areas.get(det, 0) + 1
    if not areas:
        return 0, 0
    max_mult = max(areas.values())
    max_area = [a for a, m in areas.items() if m == max_mult][0]
    return max_mult, max_area
# EVOLVE-BLOCK-END
