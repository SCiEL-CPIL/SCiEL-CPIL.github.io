import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Define grid size
    # 5x5 is small, 6x6 is larger
    # Let's try 6x6 first, it has more potential for 4-lines
    # But 5x5 is easier to validate quickly.
    # Let's stick to 5x5 for speed, or 6x6 if time permits.
    # Actually, 4x4 is 16 points. 5x5 is 25.
    # Let's search in 5x5 grid.
    
    N = 5
    points = [(x, y) for x in range(N) for y in range(N)]
    
    # Precompute all lines with >= 4 points
    # A line is defined by two points.
    # We want to count lines with exactly 4 points.
    
    def get_line_points(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        g = 0
        if dx == 0:
            g = abs(dy)
        elif dy == 0:
            g = abs(dx)
        else:
            g = 1 # simplified, need gcd for general, but grid is small
            # Actually, for grid lines, slope is rational.
            # We just need to check if a third point is on the line.
            pass
            
    # Better: Precompute all sets of collinear points.
    # For each pair, find all points on the line segment extended to grid bounds?
    # No, just check all points in grid.
    
    lines = {} # key: frozenset of points, value: count
    
    # This is O(N^4) which is 625^2 ~ 400k for 5x5. Fine.
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            p1 = points[i]
            p2 = points[j]
            x1, y1 = p1
            x2, y2 = p2
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction? Not needed for set collection.
            # Find all points on this line
            pts_on_line = [p1, p2]
            for k in range(len(points)):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                # Check if (x1,y1), (x2,y2), (x3,y3) are collinear
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    pts_on_line.append(points[k])
            
            if len(pts_on_line) >= 4:
                key = frozenset(pts_on_line)
                lines[key] = len(pts_on_line)
                
    # We want to select 16 points S such that:
    # 1. No line in 'lines' has all its points in S (if len >= 5).
    #    Actually, we just need to ensure no 5 are collinear.
    #    If a line has 5 points, we cannot select all 5.
    # 2. Maximize number of lines L in 'lines' such that |L \cap S| == 4.
    
    # Note: If a line has 6 points, and we select 4, it counts as 1.
    # If we select 5, it's invalid.
    
    # Let's use a randomized local search.
    
    def is_valid(S):
        # S is a set of points
        # Check if any 5 are collinear
        # We can use the precomputed lines?
        # If a precomputed line has len >= 5, and all its points are in S, invalid.
        # But what about lines not in 'lines'?
        # 'lines' contains all lines with >= 4 points.
        # If a line has 5 points, it is in 'lines'.
        # So checking all lines in 'lines' with len >= 5 is sufficient for validity?
        # Yes, because any 5 collinear points form a line with >= 5 points.
        
        for L in lines:
            if len(L) >= 5:
                if L.issubset(S):
                    return False
        return True

    def get_score(S):
        score = 0
        for L in lines:
            if len(L) == 4:
                if L.issubset(S):
                    score += 1
            elif len(L) > 4:
                # If we have exactly 4 points from a longer line, does it count?
                # "lines containing exactly four of the points"
                # If the geometric line has 6 points, and we picked 4,
                # does the line contain exactly 4 of *our* points? Yes.
                # But wait, if the line has 6 points, and we picked 4,
                # are there other points on that line? Yes, 2.
                # So the line contains 4 of our points.
                # The condition is "no five collinear".
                # So if we pick 4 from a 6-point line, it's valid (4 < 5).
                # And it counts as a 4-point line.
                if len(L & S) == 4:
                    score += 1
        return score

    # Initialize
    # Start with a random subset of 16
    current_S = set(random.sample(points, 16))
    
    # Ensure validity? If not, try to fix or restart.
    # For 5x5, most random sets might be valid?
    # 5-point lines are rare in random sets.
    
    if not is_valid(current_S):
        # Try to repair or restart
        for _ in range(100):
            current_S = set(random.sample(points, 16))
            if is_valid(current_S):
                break
    
    best_score = get_score(current_S)
    best_S = set(current_S)
    
    # Local Search
    while time.time() < deadline:
        # Pick a random point in S and a random point not in S
        p_in = random.choice(list(current_S))
        p_out = random.choice([p for p in points if p not in current_S])
        
        new_S = set(current_S)
        new_S.remove(p_in)
        new_S.add(p_out)
        
        if is_valid(new_S):
            new_score = get_score(new_S)
            if new_score > best_score:
                best_score = new_score
                best_S = new_S
                current_S = new_S
            # Accept worse moves with some probability?
            # Or just hill climbing.
            # Let's add a small probability to accept equal/worse to escape local optima
            # Or just restart if stuck.
        else:
            # Invalid, reject
            pass
            
        # Add some randomization
        if random.random() < 0.05:
            # Restart with a slightly perturbed current best
            current_S = set(best_S)
            for _ in range(5):
                p_in = random.choice(list(current_S))
                p_out = random.choice([p for p in points if p not in current_S])
                new_S = set(current_S)
                new_S.remove(p_in)
                new_S.add(p_out)
                if is_valid(new_S):
                    current_S = new_S

    # Convert to list of lists
    return [[x, y] for x, y in best_S]

# Run once to see if it works quickly?
# No, just return the function.
