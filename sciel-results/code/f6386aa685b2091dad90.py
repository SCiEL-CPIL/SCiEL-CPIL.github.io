import time
import random
import math

def _normalize_line(A, B, C):
    """Normalize line coefficients A*x + B*y + C = 0."""
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    # Make first non-zero positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B == 0 and C < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin below 120s
    
    # Initial configuration: 4x4 grid
    # This has many 4-point lines (rows, cols, diagonals)
    # Points: (0,0) to (3,3)
    points = [[x, y] for x in range(4) for y in range(4)]
    n = 16
    
    # Data structures for incremental update
    # line_counts: dict mapping line_key -> count of points on that line
    # point_lines: list of lists, point_lines[i] is list of line_keys passing through point i
    # We also need to be able to revert, so we store the state.
    
    # Helper to compute line key between two points
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        return _normalize_line(A, B, C)
    
    # Initialize line_counts and point_lines
    line_counts = {}
    point_lines = [[] for _ in range(n)]
    
    # Build initial state
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key not in line_counts:
                line_counts[key] = 0
            line_counts[key] += 1
            point_lines[i].append(key)
            point_lines[j].append(key)
            
    # Calculate initial score
    def calc_score():
        score = 0
        for cnt in line_counts.values():
            if cnt == 4:
                score += 1
        return score

    current_score = calc_score()
    best_score = current_score
    best_points = [p[:] for p in points]
    
    random.seed(42)
    
    # Simulated Annealing / Hill Climbing with Incremental Updates
    T = 1.0
    T_min = 0.01
    cooling = 0.9999
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, n - 1)
        old_x, old_y = points[idx]
        old_point = [old_x, old_y]
        
        # Generate new point
        # Strategy: Random within range, biased towards small moves or specific patterns
        # To find 4-lines, we might want to move a point to lie on a line with 3 existing points.
        # But for simplicity and speed, random move is okay if we have many iterations.
        
        # Let's try a mix: 50% small perturb, 50% random
        if random.random() < 0.5:
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
        else:
            new_x = random.randint(-50, 50)
            new_y = random.randint(-50, 50)
            
        new_x = max(-1000, min(1000, new_x))
        new_y = max(-1000, min(1000, new_y))
        
        new_point = [new_x, new_y]
        
        # Check if new point is same as old
        if new_x == old_x and new_y == old_y:
            continue
            
        # Check if new point collides with another point
        # This check is O(N). We can skip if we accept duplicates? No, "distinct integer points".
        # We must ensure distinctness.
        is_distinct = True
        for k in range(n):
            if k == idx:
                if points[k][0] == new_x and points[k][1] == new_y:
                    is_distinct = False
                    break
        if not is_distinct:
            continue
            
        # Apply move: Update line_counts and point_lines
        # 1. Remove old point from its lines
        old_lines = point_lines[idx]
        # We need to decrement counts for these lines
        # But careful: if a line has count 1, it becomes 0. If count 2, becomes 1.
        # We also need to remove the key from point_lines for the *other* points on that line?
        # No, point_lines[k] contains keys for lines passing through k.
        # When we move point idx, the lines passing through idx change.
        # The lines that *used to* pass through idx (old_lines) will lose one point.
        # The lines that *now* pass through idx (new_lines) will gain one point.
        
        # Decrement old lines
        for key in old_lines:
            line_counts[key] -= 1
            if line_counts[key] == 0:
                # Remove from dict to save space? Or keep. Keeping is faster for updates.
                pass
            
        # Calculate new lines for the new point
        new_lines = []
        for j in range(n):
            if j == idx:
                continue
            key = get_line_key(new_point, points[j])
            new_lines.append(key)
            
        # Increment new lines
        for key in new_lines:
            if key not in line_counts:
                line_counts[key] = 0
            line_counts[key] += 1
            
        # Update point_lines for idx
        point_lines[idx] = new_lines
        
        # Check for 5 collinear
        # Any line with count >= 5 is invalid.
        # We only need to check lines that changed: old_lines and new_lines.
        valid = True
        for key in old_lines:
            if line_counts[key] >= 5:
                valid = False
                break
        if valid:
            for key in new_lines:
                if line_counts[key] >= 5:
                    valid = False
                    break
                    
        if not valid:
            # Revert
            # Decrement new lines
            for key in new_lines:
                line_counts[key] -= 1
                if line_counts[key] == 0:
                    pass
            # Increment old lines
            for key in old_lines:
                line_counts[key] += 1
            # Restore point_lines
            point_lines[idx] = old_lines
            # Restore point
            points[idx] = old_point
            continue
            
        # Calculate new score
        # We can update score incrementally too, but calc_score is O(L) where L is num lines.
        # L is at most 120. So calc_score is fast.
        new_score = calc_score()
        
        if new_score > current_score:
            current_score = new_score
            points[idx] = new_point
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
        else:
            # Accept with probability
            delta = new_score - current_score
            if delta < 0 and random.random() < math.exp(delta / T):
                current_score = new_score
                points[idx] = new_point
            else:
                # Revert
                for key in new_lines:
                    line_counts[key] -= 1
                for key in old_lines:
                    line_counts[key] += 1
                point_lines[idx] = old_lines
                points[idx] = old_point
        
        T = max(T_min, T * cooling)
        
        # Occasional restart
        if random.random() < 0.0005:
            # Restart from best
            points = [p[:] for p in best_points]
            current_score = best_score
            # Rebuild line_counts and point_lines
            line_counts = {}
            point_lines = [[] for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    key = get_line_key(points[i], points[j])
                    if key not in line_counts:
                        line_counts[key] = 0
                    line_counts[key] += 1
                    point_lines[i].append(key)
                    point_lines[j].append(key)
            T = 1.0

    return best_points
