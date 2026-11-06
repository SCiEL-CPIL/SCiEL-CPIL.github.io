import random
import math
from itertools import combinations

def construct():
    import time
    deadline = time.time() + 120
    
    def is_valid_no4(pts):
        """Check that no 4 points are collinear."""
        n = len(pts)
        # For each pair, count how many points are on the same line
        # But we need to be careful not to double count
        # Better: for each pair, find all points on that line, if any line has >= 4, invalid
        # Use a dictionary to avoid recomputation
        from collections import defaultdict
        line_counts = defaultdict(set)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                # Normalize direction
                if dx == 0 and dy == 0:
                    continue
                g = math.gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                    dx_n = -dx_n
                    dy_n = -dy_n
                # Line is identified by (dx_n, dy_n, offset) where offset = dy_n*x1 - dx_n*y1
                # Actually, line equation: dy_n*(x - x1) - dx_n*(y - y1) = 0
                # => dy_n*x - dx_n*y = dy_n*x1 - dx_n*y1
                offset = dy_n * x1 - dx_n * y1
                key = (dx_n, dy_n, offset)
                line_counts[key].add(i)
                line_counts[key].add(j)
        for key, idx_set in line_counts.items():
            if len(idx_set) >= 4:
                return False
        return True
    
    def no3_collinear_subset_size(pts):
        """Find the largest subset with no 3 collinear using branch and bound."""
        n = len(pts)
        # Precompute collinearity: collinear[i][j][k] = True if i,j,k are collinear
        collinear = [[[False]*n for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j+1, n):
                    x3, y3 = pts[k]
                    # Check if (i,j,k) are collinear
                    # Cross product: (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
                    cross = dx * (y3 - y1) - dy * (x3 - x1)
                    if cross == 0:
                        collinear[i][j][k] = True
                        collinear[i][k][j] = True
                        collinear[j][i][k] = True
                        collinear[j][k][i] = True
                        collinear[k][i][j] = True
                        collinear[k][j][i] = True
        
        best = [0]
        
        def backtrack(start, chosen, count):
            if count + (n - start) <= best[0]:
                return
            if start == n:
                best[0] = max(best[0], count)
                return
            # Option 1: skip point start
            backtrack(start + 1, chosen, count)
            # Option 2: include point start, if no 3 collinear with already chosen
            ok = True
            for a in range(len(chosen)):
                for b in range(a+1, len(chosen)):
                    if collinear[chosen[a]][chosen[b]][start]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                backtrack(start + 1, chosen + [start], count + 1)
        
        backtrack(0, [], 0)
        return best[0]
    
    def random_valid_config():
        """Generate a random valid configuration (no 4 collinear)."""
        while True:
            pts = []
            for _ in range(20):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                pts.append([x, y])
            if is_valid_no4(pts):
                return pts
    
    def try_improve(pts, time_limit):
        """Local search to minimize the score."""
        current_pts = [p[:] for p in pts]
        current_score = no3_collinear_subset_size(current_pts)
        best_pts = [p[:] for p in current_pts]
        best_score = current_score
        
        # Simple hill climbing: try moving one point at a time
        n = len(current_pts)
        for _ in range(10000):  # Max iterations
            if time.time() > time_limit:
                break
            # Pick a random point to move
            idx = random.randint(0, n-1)
            old_x, old_y = current_pts[idx]
            
            # Try several random new positions
            found_better = False
            for _ in range(100):
                if time.time() > time_limit:
                    break
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                if new_x == old_x and new_y == old_y:
                    continue
                # Check if the new point creates 4 collinear
                # Remove the old point, add the new one
                test_pts = [p[:] for p in current_pts]
                test_pts[idx] = [new_x, new_y]
                
                # Quick check: is this valid (no 4 collinear)?
                # For the new point, check against all pairs of other points
                valid = True
                for i in range(n):
                    if i == idx:
                        continue
                    for j in range(i+1, n):
                        if j == idx:
                            continue
                        x1, y1 = test_pts[i]
                        x2, y2 = test_pts[j]
                        dx = x2 - x1
                        dy = y2 - y1
                        # Check if new point is on line through i and j
                        # Cross product: dx*(new_y - y1) - dy*(new_x - x1)
                        cross = dx * (new_y - y1) - dy * (new_x - x1)
                        if cross == 0:
                            # New point is on line ij, need to check if there's a 3rd point on this line
                            # Count points on this line
                            count = 0
                            for k in range(n):
                                x3, y3 = test_pts[k]
                                cross2 = dx * (y3 - y1) - dy * (x3 - x1)
                                if cross2 == 0:
                                    count += 1
                            if count >= 4:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Valid, compute score
                new_score = no3_collinear_subset_size(test_pts)
                if new_score < current_score:
                    current_pts = test_pts
                    current_score = new_score
                    found_better = True
                    if current_score < best_score:
                        best_score = current_score
                        best_pts = [p[:] for p in current_pts]
                    break
                elif new_score == current_score:
                    # Accept equal score to explore
                    current_pts = test_pts
                    # Don't update best yet
            
        return best_pts, best_score
    
    # Generate initial random valid config
    initial_pts = random_valid_config()
    
    # Try multiple restarts
    best_pts = [p[:] for p in initial_pts]
    best_score = no3_collinear_subset_size(best_pts)
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        pts, score = try_improve(initial_pts, deadline - 10)
        if score < best_score:
            best_score = score
            best_pts = pts
    
    return best_pts
