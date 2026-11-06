import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 138  # Safety margin
    
    def line_sig(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0:
            return (0, 0, p1[0])
        if dy == 0:
            return (0, 1, p1[1])
        g = math.gcd(abs(dx), abs(dy))
        dx_n = dx // g
        dy_n = dy // g
        if dx_n < 0 or (dx_n == 0 and dy_n < 0):
            dx_n = -dx_n
            dy_n = -dy_n
        c = dy_n * p1[0] - dx_n * p1[1]
        return (dx_n, dy_n, c)

    def build_line_map(points):
        line_map = defaultdict(list)
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                sig = line_sig(points[i], points[j])
                line_map[sig].append(i)
                line_map[sig].append(j)
        for sig in line_map:
            line_map[sig] = list(set(line_map[sig]))
        return line_map

    def count_triples(line_map):
        count = 0
        for pts in line_map.values():
            if len(pts) == 3:
                count += 1
        return count

    def check_no_four(line_map):
        for pts in line_map.values():
            if len(pts) >= 4:
                return False
        return True

    def max_no_three_collinear(points):
        n = len(points)
        # Build set of collinear triples
        triple_set = set()
        line_map = build_line_map(points)
        for pts in line_map.values():
            if len(pts) == 3:
                triple_set.add(tuple(sorted(pts)))
        
        # Branch and bound to find max subset with no 3 collinear
        # We want to find the largest S such that no triple is fully in S.
        # This is equivalent to finding the minimum number of points to remove
        # to hit all triples (hitting set), then score = n - min_hitting_set_size.
        # Or directly: max independent set in hypergraph.
        
        # Let's use a simple branch and bound on the points.
        # Order points by degree (number of triples they are in) to prune faster.
        point_triples = defaultdict(list)
        for t in triple_set:
            for p in t:
                point_triples[p].append(t)
        
        # Sort points by descending number of triples
        sorted_points = sorted(range(n), key=lambda x: len(point_triples[x]), reverse=True)
        
        # Map sorted index to original index
        # We will iterate through sorted_points and decide to include or exclude.
        # If we include a point, we must exclude at least one point from each triple it belongs to.
        # This is complex. Let's stick to the standard B&B:
        # State: (idx, current_set)
        # If idx == n, update best.
        # If current_size + (n - idx) <= best, prune.
        # Try excluding sorted_points[idx].
        # Try including sorted_points[idx] if it doesn't complete a triple with current_set.
        
        best = [0]
        
        def bnb(idx, current_size, chosen):
            if idx >= n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            if current_size + (n - idx) <= best[0]:
                return
            
            p = sorted_points[idx]
            
            # Option 1: Exclude p
            bnb(idx + 1, current_size, chosen)
            
            # Option 2: Include p
            # Check if adding p creates a collinear triple with any two points in chosen
            valid = True
            chosen_list = list(chosen)
            for i in range(len(chosen_list)):
                for j in range(i + 1, len(chosen_list)):
                    a = chosen_list[i]
                    b = chosen_list[j]
                    # Check if (a, b, p) is a triple
                    # We can check if the line through a and b contains p
                    # But we have triple_set. We need to check if (min, mid, max) is in triple_set.
                    t = tuple(sorted((a, b, p)))
                    if t in triple_set:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                chosen.add(p)
                bnb(idx + 1, current_size + 1, chosen)
                chosen.remove(p)
        
        bnb(0, 0, set())
        return best[0]

    # Initial points: Random or Grid
    # Let's try to generate points that are likely to form triples.
    # A regular grid has many 4-collinear, so we must avoid that.
    # A perturbed grid or a set of points on a few lines.
    
    # Strategy: Start with points on a circle or random, then local search to maximize triples.
    
    candidates = []
    for x in range(-20, 21):
        for y in range(-20, 21):
            candidates.append([x, y])
    
    random.seed(42)
    random.shuffle(candidates)
    
    # Start with a random set
    points = [candidates[i] for i in range(20)]
    
    # Ensure distinct
    while len(set(map(tuple, points))) < 20:
        points = [candidates[i] for i in range(20)]
        if len(set(map(tuple, points))) == 20:
            break
        # If duplicates, pick next ones
        points = [candidates[i] for i in range(20, 40)]
        break # Just pick another slice
    
    # Local Search to Maximize Triples
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        line_map = build_line_map(points)
        if not check_no_four(line_map):
            # If invalid, try to fix by moving a point involved in a 4-collinear
            # For now, just restart or skip
            # Let's just move a random point
            idx = random.randint(0, 19)
            old = points[idx]
            for cand in candidates:
                if cand == old or cand in points:
                    continue
                points[idx] = cand
                new_line_map = build_line_map(points)
                if check_no_four(new_line_map):
                    break
            continue
            
        cur_triples = count_triples(line_map)
        
        for i in range(20):
            if time.time() > deadline - 20:
                break
            old_point = points[i]
            for cand in candidates:
                if time.time() > deadline - 20:
                    break
                if cand == old_point or cand in points:
                    continue
                
                new_points = points[:]
                new_points[i] = cand
                new_line_map = build_line_map(new_points)
                if not check_no_four(new_line_map):
                    continue
                
                new_triples = count_triples(new_line_map)
                if new_triples > cur_triples:
                    points = new_points
                    cur_triples = new_triples
                    improved = True
                    break
            # If we moved point i, the indices of subsequent points in the loop are still valid
            # because we only swapped values, not the list structure.
            # But we should break the inner loop if we improved, which we do.
            # We should also restart the outer loop? No, just continue to next i.
            # However, the line_map is stale for the next i if we updated points.
            # So we should break the outer loop and restart?
            # Or just accept that the line_map is stale and it might be less efficient.
            # Given the small size, let's just break and restart the while loop to be safe and accurate.
            if improved:
                break

    # Final Score Calculation
    # We want to minimize the score.
    # The score is the size of the largest subset with no 3 collinear.
    # We have maximized the number of triples, which should minimize the score.
    
    if time.time() < deadline - 5:
        # Verify no 4 collinear
        final_line_map = build_line_map(points)
        if not check_no_four(final_line_map):
             # Fallback to a known valid config if this fails
             # e.g. parabola
             points = [[x, x*x] for x in range(-10, 10)]
             
        score = max_no_three_collinear(points)
        
    return points
