import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 130  # 20s margin for safety
    
    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def get_line_id(p1, p2):
        """
        Returns a canonical representation of the line passing through p1 and p2.
        For integer points, we can use the normalized (dx, dy) and a point on the line.
        Or simpler: (A, B, C) from Ax + By + C = 0, normalized.
        """
        x1, y1 = p1
        x2, y2 = p2
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        # Normalize (A, B, C) so that gcd(A, B, C) = 1 and first non-zero is positive
        from math import gcd
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A //= g
            B //= g
            C //= g
            
        # Ensure canonical sign
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def max_no_3_collinear(points):
        """
        Exact branch and bound for largest subset with no 3 collinear.
        Optimized for n=20.
        """
        n = len(points)
        
        # Precompute collinearity: collinear[i] is a set of j such that i, j, k are collinear for some k?
        # Actually, for the "no 3 collinear" constraint, a set S is invalid if it contains any triple (i, j, k) that is collinear.
        # We can precompute for each pair (i, j), the set of k that are collinear with i and j.
        # But a faster check for adding point i to set S:
        # i is invalid if there exist j, k in S such that i, j, k are collinear.
        
        # Let's precompute a list of "bad triples" or a structure to quickly check conflicts.
        # For each point i, and each other point j, let line(i, j) be the line.
        # If we add i to S, we conflict if there are two points j, k in S such that i, j, k are collinear.
        # This is equivalent to: for any j in S, if i and j define a line L, are there any other points in S on L?
        
        # Precompute for each pair (i, j) the line ID.
        # And for each line, the list of points on it.
        
        lines = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                lid = get_line_id(points[i], points[j])
                lines[lid].append(i)
                lines[lid].append(j)
        
        # Clean up lines: unique points
        for lid in lines:
            lines[lid] = list(set(lines[lid]))
            
        # For the B&B, we can use a simple recursive approach with bitmasks or lists.
        # Given n=20, a list of indices is fine.
        
        best = 0
        
        # Sort points? Maybe by "degree" (number of collinear pairs) to prune earlier?
        # Points on lines with 3 points are "hotter".
        
        def backtrack(S, start_idx):
            nonlocal best
            current_len = len(S)
            # Pruning: if current_len + (n - start_idx) <= best, return
            if current_len + (n - start_idx) <= best:
                return
                
            if start_idx == n:
                if current_len > best:
                    best = current_len
                return
            
            # Option 1: Skip start_idx
            backtrack(S, start_idx + 1)
            
            # Option 2: Include start_idx
            # Check validity
            valid = True
            p_idx = start_idx
            for j in S:
                # Check if p_idx and j are collinear with any k in S
                # We can precompute for each pair (p_idx, j) the set of points collinear with them.
                # Or just check all k in S.
                # Since |S| <= 20, O(|S|^2) is acceptable.
                for k in S:
                    if k == j: continue
                    if are_collinear(points[p_idx], points[j], points[k]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                S.append(start_idx)
                backtrack(S, start_idx + 1)
                S.pop()
                
        backtrack([], 0)
        return best

    def greedy_construct():
        """
        Construct 20 points by greedily filling lines with 3 points each.
        We want to form 6 lines of 3 points.
        Points can be shared between lines (intersections).
        """
        points = []
        used_lines = set()
        
        # Strategy:
        # 1. Pick a random line L1. Place 3 points on it.
        # 2. Pick a random line L2 intersecting L1 at one of the existing points (if possible) or a new point.
        #    Actually, to minimize the independence number, we want to maximize the "clique" of constraints.
        #    A configuration of 6 lines, each with 3 points, where points are shared as much as possible.
        #    Max sharing: Each point can be on at most 2 lines? If a point is on 3 lines, it's a "triple intersection".
        #    Let's try to create a grid-like or triangular arrangement.
        
        # Simpler Greedy:
        # Maintain a set of candidate points.
        # At each step, pick the point that lies on the most "open" lines (lines with < 3 points).
        # Or, pick a line with the most points (<3), and add a point on it.
        
        # Let's try a different angle: Random restarts of a specific structured pattern.
        # Pattern: 6 lines.
        # Line 1: y=0, x in {-2, 0, 2}
        # Line 2: y=1, x in {-2, 0, 2}
        # Line 3: y=2, x in {-2, 0, 2}
        # This is a 3x3 grid (9 points) + 11 more?
        # In a 3x3 grid, rows and columns have 3 points.
        # Score of 3x3 grid: Max subset with no 3 collinear.
        # It's known that for 3x3, the max is 6? (e.g. 2 from each row, avoiding columns).
        # Actually, let's just run the B&B on a 3x3 grid to see.
        
        # Better: Use a "blow-up" of a small configuration.
        # Or, just use the random generation but with a strong bias towards collinearity.
        
        # Let's implement a "Line-Packing" heuristic.
        # 1. Generate a random set of 6 lines.
        # 2. Place 3 points on each line.
        # 3. Resolve conflicts (4 collinear) by moving points.
        
        # Actually, the most effective strategy for "minimize max no-3-collinear subset" is to make the points as "clustered" as possible into triples.
        # If we can form 6 disjoint triples (18 points) + 2 points, the score is likely 2 + (something from the triples).
        # But triples share points.
        
        # Let's try: Generate 20 points such that they lie on 6 specific lines.
        # We will randomly select 6 lines from a pool of "nice" lines (low slope, integer intercepts).
        # Then we pick 3 points on each line, ensuring distinctness and no 4 collinear.
        
        # Pool of lines:
        # y = m x + c
        # m in {-2, -1, 0, 1, 2}
        # c in {-10, -5, 0, 5, 10}
        # Total 5 * 5 = 25 lines.
        
        lines_pool = []
        for m in [-2, -1, 0, 1, 2]:
            for c in [-10, -5, 0, 5, 10]:
                lines_pool.append((m, c))
        
        random.shuffle(lines_pool)
        
        # Pick 6 lines
        selected_lines = lines_pool[:6]
        
        # For each selected line, generate candidate points
        # We need 3 points per line.
        # Points are integer (x, y) with |x|, |y| <= 100.
        
        all_points = set()
        line_points = {i: [] for i in range(6)}
        
        # We need to assign points to lines.
        # A point can belong to multiple lines (intersection).
        # If a point is an intersection of two lines, it "satisfies" both.
        
        # Step 1: Find intersections of selected lines.
        # If line i and line j intersect at an integer point, that's a good candidate.
        
        candidates = set()
        for i in range(6):
            for j in range(i + 1, 6):
                m1, c1 = selected_lines[i]
                m2, c2 = selected_lines[j]
                if m1 == m2:
                    continue # Parallel
                # m1 x + c1 = m2 x + c2
                # x = (c2 - c1) / (m1 - m2)
                num = c2 - c1
                den = m1 - m2
                if num % den == 0:
                    x = num // den
                    y = m1 * x + c1
                    if abs(x) <= 100 and abs(y) <= 100:
                        candidates.add((x, y))
        
        # Also add some random points on the lines to have more options
        for i in range(6):
            m, c = selected_lines[i]
            for _ in range(10):
                x = random.randint(-20, 20)
                y = m * x + c
                if abs(y) <= 100:
                    candidates.add((x, y))
        
        # Now, we need to select 20 points from candidates (or generated on the fly) 
        # such that each of the 6 lines has at least 3 points from our selection.
        # And no 4 points are collinear.
        
        # This is a constraint satisfaction problem.
        # Let's use a simple greedy/backtracking on the lines.
        
        # Actually, let's just generate a random set of 20 points from the candidates 
        # and check if it satisfies the "3 per line" constraint? 
        # No, we WANT the 3 per line constraint to force the score down.
        
        # Let's try a different approach:
        # Start with empty set.
        # While len(points) < 20:
        #   Find a line from our 6 lines that has the fewest points in the current set.
        #   Add a random point on that line (not already in set).
        #   Check if adding it creates a 4-collinear set. If so, try another point or line.
        
        points = []
        line_counts = defaultdict(int)
        
        # Precompute which candidate points lie on which selected lines
        point_lines = defaultdict(list)
        for p in candidates:
            for i in range(6):
                m, c = selected_lines[i]
                if p[1] == m * p[0] + c:
                    point_lines[p].append(i)
        
        # If a point is on multiple lines, it's valuable.
        # Sort candidates by number of lines they satisfy (descending)
        sorted_candidates = sorted(candidates, key=lambda p: len(point_lines[p]), reverse=True)
        
        for p in sorted_candidates:
            if len(points) >= 20:
                break
            # Check if adding p creates 4 collinear
            new_pts = points + [p]
            valid = True
            # Check 4 collinear
            n = len(new_pts)
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if are_collinear(new_pts[i], new_pts[j], new_pts[k]):
                            for m_idx in range(n):
                                if m_idx != i and m_idx != j and m_idx != k:
                                    if are_collinear(new_pts[i], new_pts[j], new_pts[m_idx]):
                                        valid = False
                                        break
                    if not valid: break
                if not valid: break
            if not valid:
                continue
                
            if p not in points:
                points.append(p)
                for l in point_lines[p]:
                    line_counts[l] += 1
        
        # If we didn't get 20, fill with random points
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            p = (x, y)
            if p in points:
                continue
            new_pts = points + [p]
            # Quick check for 4 collinear
            valid = True
            n = len(new_pts)
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if are_collinear(new_pts[i], new_pts[j], new_pts[k]):
                            for m_idx in range(n):
                                if m_idx != i and m_idx != j and m_idx != k:
                                    if are_collinear(new_pts[i], new_pts[j], new_pts[m_idx]):
                                        valid = False
                                        break
                    if not valid: break
                if not valid: break
            if valid:
                points.append(p)
        
        return [list(p) for p in points]

    best_config = None
    best_score = 21
    
    # Run multiple restarts of the greedy construction
    for _ in range(50):
        if time.time() > deadline - 5:
            break
            
        pts = greedy_construct()
        
        # Validate
        if len(set(map(tuple, pts))) != 20:
            continue
        if any(abs(p[0]) > 100 or abs(p[1]) > 100 for p in pts):
            continue
            
        # Check 4 collinear
        has_4 = False
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(pts[i], pts[j], pts[k]):
                        for m in range(n):
                            if m != i and m != j and m != k:
                                if are_collinear(pts[i], pts[j], pts[m]):
                                    has_4 = True
                                    break
                    if has_4: break
                if has_4: break
            if has_4: break
            
        if has_4:
            continue
            
        score = max_no_3_collinear(pts)
        
        if score < best_score:
            best_score = score
            best_config = [list(p) for p in pts]
            
        if best_score <= 10:
            break

    if best_config is None:
        # Fallback
        return [[x, x * x] for x in range(-10, 10)]
        
    return best_config

# EVOLVE-BLOCK-END
