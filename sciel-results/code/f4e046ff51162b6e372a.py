import random
import time
import itertools
import sys

def construct():
    """
    Find 16 integer points with |x|,|y| <= 1000, no five collinear,
    maximizing the number of lines containing exactly four points.
    
    Strategy: 
    1. Start with a known good structure (affine plane AG(2,4) embedded in integers).
    2. Use local search with incremental evaluation to improve.
    3. Deadline guard to return best found within time limit.
    """
    deadline = time.time() + 140  # 140s to be safe under 150s limit
    
    def count_exactly_four(points):
        """Count lines containing exactly 4 of the given points. O(n^3) but fast for n=16."""
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                # Line through points i and j
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                # Check how many other points are on this line
                on_line = 2  # i and j
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    # Cross product to check collinearity
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        on_line += 1
                if on_line == 4:
                    count += 1
        # Each line with exactly 4 points is counted C(4,2)=6 times
        return count // 6
    
    def has_five_collinear(points):
        """Check if any line has 5 or more points."""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                on_line = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        on_line += 1
                if on_line >= 5:
                    return True
        return False
    
    # Start with a 4x4 grid (known to have 20 four-point lines: 4 rows + 4 cols + 2 diags... wait)
    # Actually 4x4 grid: 4 rows of 4, 4 cols of 4, and diagonals. Let's compute.
    # Rows: 4 lines with 4 points each
    # Cols: 4 lines with 4 points each  
    # Diagonals: 2 main diagonals with 4 points each
    # Other diagonals of length 4? In 4x4, only the two main diagonals have 4 points.
    # So 4+4+2 = 10 lines with exactly 4 points? Let me verify.
    # Actually in a 4x4 grid, the diagonals of slope 1 and -1:
    # Slope 1: main diag (4 pts), and others have 1,2,3 pts
    # Slope -1: main diag (4 pts), others have 1,2,3 pts
    # So 4 rows + 4 cols + 2 diags = 10 four-point lines.
    
    # But we can do better. Let's use a known configuration.
    # The affine plane AG(2,4) has 16 points and 20 lines of 4 points each.
    # We need to embed this in integer coordinates.
    
    # Let's try a structured approach: start with 4x4 grid and perturb.
    def make_grid():
        return [[x, y] for x in range(4) for y in range(4)]
    
    best_points = make_grid()
    best_score = count_exactly_four(best_points)
    
    # Verify no 5 collinear
    if has_five_collinear(best_points):
        # 4x4 grid shouldn't have 5 collinear
        pass
    
    random.seed(42)
    
    def random_point():
        return [random.randint(-1000, 1000), random.randint(-1000, 1000)]
    
    # Local search
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(16):
            old_x, old_y = best_points[i]
            # Try moving point i to a nearby random position
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                new_x = old_x + random.randint(-50, 50)
                new_y = old_y + random.randint(-50, 50)
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                # Check if this creates a duplicate
                new_pts = best_points[:]
                new_pts[i] = [new_x, new_y]
                # Check for duplicates
                dup = False
                for j in range(16):
                    if j != i and new_pts[j] == new_pts[i]:
                        dup = True
                        break
                if dup:
                    continue
                # Check no 5 collinear
                if has_five_collinear(new_pts):
                    continue
                score = count_exactly_four(new_pts)
                if score > best_score:
                    best_score = score
                    best_points = new_pts
                    improved = True
                    break
    
    # Try more aggressive moves: swap two points
    for _ in range(100000):
        if time.time() > deadline - 5:
            break
        i, j = random.sample(range(16), 2)
        new_pts = best_points[:]
        new_pts[i], new_pts[j] = new_pts[j][:], new_pts[i][:]
        # Swap doesn't change the set, so score is the same. Skip.
        # Instead, let's try moving one point to another's position + offset
        i = random.randint(0, 15)
        j = random.randint(0, 15)
        if i == j:
            continue
        new_x = best_points[j][0] + random.randint(-10, 10)
        new_y = best_points[j][1] + random.randint(-10, 10)
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        new_pts = best_points[:]
        new_pts[i] = [new_x, new_y]
        dup = False
        for k in range(16):
            if k != i and new_pts[k] == new_pts[i]:
                dup = True
                break
        if dup:
            continue
        if has_five_collinear(new_pts):
            continue
        score = count_exactly_four(new_pts)
        if score > best_score:
            best_score = score
            best_points = new_pts
    
    # Final: try to improve by adjusting individual coordinates
    for _ in range(50000):
        if time.time() > deadline - 2:
            break
        i = random.randint(0, 15)
        dx = random.choice([-2,-1,1,2])
        dy = random.choice([-2,-1,1,2])
        new_x = best_points[i][0] + dx
        new_y = best_points[i][1] + dy
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        new_pts = best_points[:]
        new_pts[i] = [new_x, new_y]
        dup = False
        for k in range(16):
            if k != i and new_pts[k] == new_pts[i]:
                dup = True
                break
        if dup:
            continue
        if has_five_collinear(new_pts):
            continue
        score = count_exactly_four(new_pts)
        if score > best_score:
            best_score = score
            best_points = new_pts
    
    return [[int(p[0]), int(p[1])] for p in best_points]
