import time
import random
import itertools
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin for 150s limit

    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def get_line_id(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A //= g
            B //= g
            C //= g
            
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def has_4_collinear(points):
        lines = defaultdict(set)
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                lid = get_line_id(points[i], points[j])
                lines[lid].add(i)
                lines[lid].add(j)
        for lid, pts_in_line in lines.items():
            if len(pts_in_line) >= 4:
                return True
        return False

    def max_no_3_collinear(points):
        """
        Exact branch and bound using bitmasks for speed.
        Optimized for n=20.
        """
        n = len(points)
        if n <= 2:
            return n
            
        # Precompute all lines
        all_lines = defaultdict(int) # lid -> bitmask of points
        for i in range(n):
            for j in range(i + 1, n):
                lid = get_line_id(points[i], points[j])
                all_lines[lid] |= (1 << i)
                all_lines[lid] |= (1 << j)
                
        # Precompute line_id for each pair
        line_id_pair = [[None]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    line_id_pair[i][j] = get_line_id(points[i], points[j])
                    
        best = 0
        full_mask = (1 << n) - 1
        
        # Order points by "difficulty": points involved in more collinear triples first?
        # Or just index order. Index order is fine for B&B if we have good pruning.
        # Let's try sorting by number of lines passing through them.
        point_line_count = [0]*n
        for i in range(n):
            lines_seen = set()
            for j in range(n):
                if i != j:
                    lines_seen.add(line_id_pair[i][j])
            point_line_count[i] = len(lines_seen)
            
        # Sort indices by descending line count (hardest to place first)
        indices = list(range(n))
        indices.sort(key=lambda x: -point_line_count[x])
        
        # Remap points to sorted order for better B&B performance?
        # Actually, B&B with start_idx works best if we process in a fixed order.
        # If we permute points, we must permute the masks too.
        # Let's stick to original indices but use the line_id_pair directly.
        
        def backtrack(S_mask, start_idx):
            nonlocal best
            current_len = S_mask.bit_count()
            
            # Pruning
            if current_len + (n - start_idx) <= best:
                return
            
            if start_idx == n:
                if current_len > best:
                    best = current_len
                return
            
            # Option 1: Skip start_idx
            backtrack(S_mask, start_idx + 1)
            
            # Option 2: Include start_idx
            i = start_idx
            valid = True
            # Check conflicts with points already in S
            temp_mask = S_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                j = lsb.bit_length() - 1
                lid = line_id_pair[i][j]
                M = all_lines[lid]
                # If any other point k in S is on this line (besides j), invalid.
                # (M & S_mask) contains j and any k.
                # If popcount > 1, invalid.
                if (M & S_mask).bit_count() > 1:
                    valid = False
                    break
                temp_mask ^= lsb
                
            if valid:
                backtrack(S_mask | (1 << i), start_idx + 1)
                
        backtrack(0, 0)
        return best

    def greedy_construct():
        """
        Construct points to minimize the max no-3-collinear subset.
        Strategy: 
        1. Start with an empty set.
        2. At each step, find the point (from a candidate pool) that, when added,
           creates the most "constraints" or forces the most triples.
           Specifically, we want to maximize the number of lines that have exactly 3 points,
           or lines that are "full" (3 points), because those points cannot be in a no-3-collinear set together.
           
        Alternative simpler strategy:
        Use a known good configuration type: points on a convex curve (parabola) 
        tend to have small no-3-collinear subsets? No, points on a parabola y=x^2 
        have no 3 collinear! That would give score 20. We want to MINIMIZE score.
        So we want to MANY collinearities.
        
        We want a configuration where it is hard to pick many points without 3 collinear.
        This is equivalent to maximizing the "collinearity density".
        
        Let's try to pack points into lines of 3.
        If we can partition 20 points into lines of 3 (with overlaps), we can force the score down.
        
        Greedy:
        1. Pick a random point P1.
        2. Pick a random point P2.
        3. Pick P3 such that P1, P2, P3 are collinear.
        4. Now we have a line with 3 points.
        5. Pick P4.
        6. Pick P5.
        7. Pick P6 such that P4, P5, P6 are collinear? Or maybe P6 collinear with P1, P4?
        
        Let's try a structured approach:
        Divide the plane into regions?
        
        Actually, a very effective way to minimize the max no-3-collinear subset is to use a "grid" 
        but perturb it slightly to break 4-collinear while keeping many 3-collinear.
        
        Let's try a 5x5 grid subset again, but choose the 20 points specifically to maximize 3-collinear triples.
        In a 5x5 grid, rows and cols have 5 points. If we take 4 from a row, we have 4 collinear (invalid).
        So we can take at most 3 from any row/col/diagonal.
        
        Let's try to construct a set of 20 points from a 5x5 grid (0..4, 0..4) 
        such that no line has > 3 points, and we maximize the number of lines with exactly 3 points.
        
        We will use a backtracking search to find such a subset, or a greedy heuristic.
        """
        grid_points = [(x, y) for x in range(5) for y in range(5)]
        
        # Precompute lines for the grid
        # We need to check if adding a point creates a 4-collinear.
        # And we want to prefer points that create 3-collinear.
        
        best_subset = None
        best_triples = -1
        
        # Random restarts of greedy
        for _ in range(50):
            if time.time() > deadline - 10:
                break
                
            random.shuffle(grid_points)
            pts = []
            # We want to pick 20 points.
            # Greedy: pick point that maximizes new triples formed, subject to no 4-collinear.
            
            current_lines = defaultdict(int) # lid -> count
            current_line_points = defaultdict(set) # lid -> set of indices
            
            for p in grid_points:
                if len(pts) == 20:
                    break
                    
                # Calculate score for adding p
                new_triples = 0
                valid = True
                temp_lines = defaultdict(int)
                temp_line_points = defaultdict(set)
                
                # Copy current state
                for lid, cnt in current_lines.items():
                    temp_lines[lid] = cnt
                for lid, s in current_line_points.items():
                    temp_line_points[lid] = set(s)
                    
                # Check p against existing points
                for i, existing_p in enumerate(pts):
                    lid = get_line_id(p, existing_p)
                    temp_lines[lid] += 1
                    temp_line_points[lid].add(i)
                    temp_line_points[lid].add(len(pts)) # index of p in new list
                    
                    if temp_lines[lid] >= 3:
                        # Check if this creates a 4-collinear
                        # If count becomes 3, we have 3 points.
                        # If count becomes 4, we have 4 points -> INVALID.
                        # Wait, line count is number of pairs? No, I'm counting points on line.
                        # Let's use a simpler check.
                        pass
                        
                # Simpler validity check:
                new_pts = pts + [p]
                if has_4_collinear(new_pts):
                    continue
                    
                # Count triples
                # A triple is a line with exactly 3 points.
                # We can count how many lines have exactly 3 points in new_pts.
                lines_in_new = defaultdict(set)
                for i in range(len(new_pts)):
                    for j in range(i+1, len(new_pts)):
                        lid = get_line_id(new_pts[i], new_pts[j])
                        lines_in_new[lid].add(i)
                        lines_in_new[lid].add(j)
                        
                triples = 0
                for lid, s in lines_in_new.items():
                    if len(s) == 3:
                        triples += 1
                        
                if triples > best_triples:
                    best_triples = triples
                    best_subset = new_pts.copy()
                    
                # Greedy step: just add it? Or only if it increases triples?
                # Let's just add it if valid, to ensure we get 20 points.
                # Then we will do a local search to improve triples.
                pts.append(p)
                
            if len(pts) == 20 and best_subset:
                # Refine best_subset
                return best_subset
                
        return None

    def local_search_triples(points):
        """
        Improve the configuration by maximizing the number of 3-collinear triples.
        More triples generally means a smaller max no-3-collinear subset.
        """
        current_triples = -1
        lines_in_cur = defaultdict(set)
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                lid = get_line_id(points[i], points[j])
                lines_in_cur[lid].add(i)
                lines_in_cur[lid].add(j)
        for lid, s in lines_in_cur.items():
            if len(s) == 3:
                current_triples += 1
                
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(points)):
                orig = points[i]
                # Try moving point i to a random location
                for _ in range(10):
                    new_x = random.randint(-100, 100)
                    new_y = random.randint(-100, 100)
                    new_p = (new_x, new_y)
                    if new_p in points:
                        continue
                        
                    points[i] = new_p
                    if has_4_collinear(points):
                        points[i] = orig
                        continue
                        
                    # Count triples
                    lines_in_new = defaultdict(set)
                    for a in range(len(points)):
                        for b in range(a+1, len(points)):
                            lid = get_line_id(points[a], points[b])
                            lines_in_new[lid].add(a)
                            lines_in_new[lid].add(b)
                            
                    new_triples = 0
                    for lid, s in lines_in_new.items():
                        if len(s) == 3:
                            new_triples += 1
                            
                    if new_triples > current_triples:
                        current_triples = new_triples
                        improved = True
                    else:
                        points[i] = orig
                        
        return points

    best_config = None
    best_score = 21
    
    # Strategy 1: Grid-based greedy
    for i in range(10):
        if time.time() > deadline - 20:
            break
            
        pts = greedy_construct()
        if pts is None:
            continue
            
        # Scale to fit |x|,|y| <= 100? Grid is 0..4, fits.
        # But maybe we can stretch it? No, integer coords.
        
        if not has_4_collinear(pts):
            # Refine to maximize triples
            refined_pts = local_search_triples(pts)
            score = max_no_3_collinear(refined_pts)
            
            if score < best_score:
                best_score = score
                best_config = [list(p) for p in refined_pts]
                
        if best_score <= 10:
            break

    # Strategy 2: Random with heavy bias towards collinearities
    for i in range(10):
        if time.time() > deadline - 10:
            break
            
        pts = []
        # Pick 6 random lines
        lines = []
        for _ in range(6):
            # Pick 2 random points to define a line
            p1 = (random.randint(-10, 10), random.randint(-10, 10))
            p2 = (random.randint(-10, 10), random.randint(-10, 10))
            if p1 == p2:
                continue
            lines.append((p1, p2))
            
        candidates = set()
        for p1, p2 in lines:
            # Add points on this line
            x1, y1 = p1
            x2, y2 = p2
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            step_x = dx // g
            step_y = dy // g
            
            # Extend line
            for k in range(-20, 21):
                x = x1 + k * step_x
                y = y1 + k * step_y
                if abs(x) <= 100 and abs(y) <= 100:
                    candidates.add((x, y))
                    
        random.shuffle(list(candidates))
        for p in candidates:
            if len(pts) == 20:
                break
            if p in pts:
                continue
            new_pts = pts + [p]
            if not has_4_collinear(new_pts):
                pts.append(p)
                
        # Fill rest
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = (x, y)
            if p in pts:
                continue
            new_pts = pts + [p]
            if not has_4_collinear(new_pts):
                pts.append(p)
                
        if len(pts) == 20:
            refined_pts = local_search_triples(pts)
            score = max_no_3_collinear(refined_pts)
            if score < best_score:
                best_score = score
                best_config = [list(p) for p in refined_pts]

    if best_config is None:
        # Fallback: Random valid config
        while True:
            pts = []
            for _ in range(20):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                p = (x, y)
                if p in pts:
                    continue
                new_pts = pts + [p]
                if not has_4_collinear(new_pts):
                    pts.append(p)
            if len(pts) == 20:
                return [list(p) for p in pts]
                
    return best_config
