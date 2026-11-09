import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # Precompute all triples (i, j, k) with i < j < k for 30 points
    triples = list(combinations(range(30), 3))
    # Precompute indices for vectorization
    i_idx = np.array([t[0] for t in triples], dtype=np.int64)
    j_idx = np.array([t[1] for t in triples], dtype=np.int64)
    k_idx = np.array([t[2] for t in triples], dtype=np.int64)
    
    def score(points):
        """Compute the max multiplicity of any positive doubled area."""
        n = len(points)
        xs = np.array([p[0] for p in points], dtype=np.int64)
        ys = np.array([p[1] for p in points], dtype=np.int64)
        
        xi = xs[i_idx]
        yi = ys[i_idx]
        xj = xs[j_idx]
        yj = ys[j_idx]
        xk = xs[k_idx]
        yk = ys[k_idx]
        
        area2 = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
        
        # Exclude zero areas
        mask = area2 > 0
        if not mask.any():
            return 0, 0, {}
        
        # Count frequencies
        unique, counts = np.unique(area2[mask], return_counts=True)
        max_count = int(counts.max())
        # Find the area corresponding to the max count
        # If there are ties, pick the first one or random? Let's pick the one with max count.
        # If multiple areas have the same max count, we might want to know which one.
        # For the search, we need the specific area.
        target_area = int(unique[np.argmax(counts)])
        
        # Build a map of area to count for faster lookup in local search?
        # Or just return the target area and count.
        return max_count, target_area, {}

    def generate_candidates():
        candidates = []
        
        def make_grid(w, h, x0, y0, dx, dy):
            points = []
            for i in range(w):
                for j in range(h):
                    x = x0 + i * dx
                    y = y0 + j * dy
                    if -30 <= x <= 30 and -30 <= y <= 30:
                        points.append((x, y))
            return points[:30] if len(points) >= 30 else None
        
        # Standard grids
        for w in range(3, 11):
            for h in range(3, 11):
                if w * h >= 30:
                    for x0 in range(-30, 10, 3):
                        for y0 in range(-30, 10, 3):
                            for dx in [2, 3, 4, 5, 6]:
                                for dy in [2, 3, 4, 5, 6]:
                                    pts = make_grid(w, h, x0, y0, dx, dy)
                                    if pts and len(set(pts)) == 30:
                                        candidates.append(pts)
        
        # Limit candidates
        random.shuffle(candidates)
        return candidates[:500]

    def spectral_local_search(points, time_limit):
        """
        Local search guided by 'spectral' centrality.
        Focuses on increasing the count of the current dominant area.
        """
        current = points[:]
        current_score, current_area, _ = score(current)
        
        if current_score == 0:
            return current, 0

        # Precompute triple indices relative to points for delta calculation
        # We need to know which triples involve a specific point index.
        # Since indices are fixed 0..29, we can precompute lists of triples for each point index.
        triples_by_point = [[] for _ in range(30)]
        for t_idx, (i, j, k) in enumerate(triples):
            triples_by_point[i].append(t_idx)
            triples_by_point[j].append(t_idx)
            triples_by_point[k].append(t_idx)

        # Convert current points to arrays
        xs = np.array([p[0] for p in current], dtype=np.int64)
        ys = np.array([p[1] for p in current], dtype=np.int64)

        def calc_area2(i, j, k):
            return abs((xs[j] - xs[i]) * (ys[k] - ys[i]) - (ys[j] - ys[i]) * (xs[k] - xs[i]))

        # Initial count of current_area
        # We already have current_score which is the count of current_area
        
        # We need to be able to calculate the delta in count of current_area
        # when we move point idx to new_pos.
        
        # To speed up, we can precompute the "contribution" of each point to current_area.
        # contrib[p] = number of triples involving p that have area == current_area
        
        contrib = np.zeros(30, dtype=np.int64)
        # This is O(N^3) once, which is fine.
        for t_idx, (i, j, k) in enumerate(triples):
            a = calc_area2(i, j, k)
            if a == current_area:
                contrib[i] += 1
                contrib[j] += 1
                contrib[k] += 1
        
        # Now, when we move point `idx` to `new_pos`:
        # 1. Remove contributions of all triples involving `idx` that currently have area == current_area.
        # 2. Calculate new areas for those triples with `new_pos`.
        # 3. Add back contributions for new triples that have area == current_area.
        # Delta = new_contrib - old_contrib.
        
        # Note: This only tracks the count for `current_area`.
        # If the count decreases, we might be losing the global max.
        # But if we find a move that increases the count for `current_area`, we accept it.
        # If we get stuck, we might need to restart or accept equal moves.
        
        # To handle "switching" to a new area, we can periodically do a full score.
        
        for it in range(10000):
            if time.time() > time_limit:
                break
            
            # Pick a point to move.
            # Prefer points with low centrality (low contrib) as they are less critical to current_area.
            # If all have high contrib, pick random.
            # Actually, moving a high contrib point is risky. Moving a low contrib point is safer.
            # Let's pick the point with the minimum contrib.
            idx = np.argmin(contrib)
            
            best_delta = 0
            best_new_pos = None
            best_new_contrib = None # array of new contribs
            
            # Try nearby moves
            for _ in range(20):
                ox = random.randint(-5, 5)
                oy = random.randint(-5, 5)
                new_x = current[idx][0] + ox
                new_y = current[idx][1] + oy
                
                if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                    continue
                
                new_point = (new_x, new_y)
                
                # Check distinctness
                if new_point in current[:idx] + current[idx+1:]:
                    continue
                
                # Calculate delta in contrib for current_area
                # Old contrib for idx is contrib[idx]
                # We need to check each triple involving idx.
                
                delta = 0
                new_contrib_arr = contrib.copy() # We need to update contrib for all points in the triples
                
                # Actually, we only need the total count change, not the full array, 
                # unless we want to update the contrib array for future steps.
                # Let's just calculate the new total count for current_area.
                # Old total = current_score.
                # New total = current_score - (old_count_involving_idx) + (new_count_involving_idx)
                
                old_count_involving_idx = 0
                new_count_involving_idx = 0
                
                for t_idx in triples_by_point[idx]:
                    i, j, k = triples[t_idx]
                    # Replace idx in the triple with new point for calculation
                    # We need to calculate the area of the triple with the new point.
                    # The other two points remain fixed.
                    
                    # Identify the other two indices
                    others = [x for x in (i, j, k) if x != idx]
                    o1, o2 = others[0], others[1]
                    
                    # Old area (should be current_area if it was counted in contrib)
                    # We don't need to recalc old area, we just subtract 1 if it was current_area.
                    old_a = calc_area2(i, j, k)
                    if old_a == current_area:
                        old_count_involving_idx += 1
                    
                    # New area
                    # Points: new_point, current[o1], current[o2]
                    # Order doesn't matter for abs determinant
                    x1, y1 = new_point
                    x2, y2 = current[o1]
                    x3, y3 = current[o2]
                    new_a = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    
                    if new_a == current_area:
                        new_count_involving_idx += 1
                
                delta = new_count_involving_idx - old_count_involving_idx
                
                if delta > best_delta:
                    best_delta = delta
                    best_new_pos = new_point
                    # We don't strictly need to update the full contrib array here for the decision,
                    # but we should if we accept.
                    # For now, just store the pos.
                
                # Plateau escape: accept equal moves with small prob
                elif delta == 0 and best_delta == 0 and random.random() < 0.1:
                    best_delta = 0
                    best_new_pos = new_point

            if best_new_pos is not None:
                # Apply move
                current[idx] = best_new_pos
                xs[idx] = best_new_pos[0]
                ys[idx] = best_new_pos[1]
                current_score += best_delta
                
                # Update contrib array
                # We need to recalculate contrib for all points involved in triples with idx?
                # Or just recalculate contrib for idx and the others?
                # To keep it simple and correct, let's just recalculate the full contrib array
                # every 10 accepted moves or if we suspect drift.
                # Actually, recalculating full contrib is O(N^3) ~ 4000 ops.
                # If we do it every move, it's slow.
                # Let's do it every 5 moves.
                if it % 5 == 0:
                    contrib = np.zeros(30, dtype=np.int64)
                    for t_idx, (i, j, k) in enumerate(triples):
                        a = calc_area2(i, j, k)
                        if a == current_area:
                            contrib[i] += 1
                            contrib[j] += 1
                            contrib[k] += 1
                            
                    # Check if current_area is still the max?
                    # This is the expensive part.
                    # Let's do a full score check every 20 moves.
                    if it % 20 == 0:
                        new_score, new_area, _ = score(current)
                        if new_score > current_score:
                            current_score = new_score
                            current_area = new_area
                            # Reset contrib
                            contrib = np.zeros(30, dtype=np.int64)
                            for t_idx, (i, j, k) in enumerate(triples):
                                a = calc_area2(i, j, k)
                                if a == current_area:
                                    contrib[i] += 1
                                    contrib[j] += 1
                                    contrib[k] += 1
                        elif new_score < current_score:
                            # This shouldn't happen if delta logic is correct for current_area,
                            # unless current_area stopped being the max and dropped.
                            # If current_area count dropped below the new max of another area,
                            # our local search is blind to that.
                            # If new_score == current_score, but new_area != current_area,
                            # we might want to switch target area if it allows further growth.
                            pass

        return current, current_score

    # Generate candidates
    candidates = generate_candidates()
    
    best_points = None
    best_score = 0
    best_area = 0
    
    # Evaluate candidates
    for pts in candidates:
        if time.time() > deadline - 20:
            break
        s, a, _ = score(pts)
        if s > best_score:
            best_score = s
            best_area = a
            best_points = pts[:]
            
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score, best_area, _ = score(best_points)
        
    # Run spectral local search
    if time.time() < deadline - 5:
        best_points, best_score = spectral_local_search(best_points, deadline - 2)
        
    # Try a few restarts
    for restart in range(3):
        if time.time() > deadline - 5:
            break
        start_pts = candidates[random.randint(0, len(candidates)-1)]
        result_pts, result_score = spectral_local_search(start_pts, deadline - 2)
        if result_score > best_score:
            best_score = result_score
            best_points = result_pts
            
    # Final validation
    assert len(best_points) == 30
    assert len(set(best_points)) == 30
    for x, y in best_points:
        assert -30 <= x <= 30 and -30 <= y <= 30
        
    return best_points
