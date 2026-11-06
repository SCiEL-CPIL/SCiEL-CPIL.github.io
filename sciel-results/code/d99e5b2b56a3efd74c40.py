import math
import time
import random
from itertools import combinations

def _line_key(p1, p2):
    """Return canonical key for the line through p1, p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line: a*x + b*y + c = 0
    # a = dy, b = -dx, c = dx*y1 - dy*x1
    a = dy
    b = -dx
    c = dx * y1 - dy * x1
    # Normalize sign
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a = -a
        b = -b
        c = -c
    g2 = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    return (a, b, c)

def _score_and_map(points):
    """
    Calculate score and build line map.
    Returns (score, line_map)
    score is number of 4-point lines.
    line_map is dict: line_key -> list of indices
    """
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in line_map:
                line_map[key] = []
            line_map[key].append(i)
            line_map[key].append(j)
            # Wait, this adds i and j every time. This is wrong for building the map.
            # We need unique sets.
    
    # Correct way to build map
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
            
    # Convert sets to lists for easier manipulation? Or keep as sets.
    # Sets are fine.
    
    count_4 = 0
    max_collinear = 0
    for key, pts in line_map.items():
        k = len(pts)
        if k == 4:
            count_4 += 1
        if k > max_collinear:
            max_collinear = k
            
    # Penalty for 5+ collinear
    if max_collinear >= 5:
        # Heavy penalty to ensure we avoid this
        # The task says "no five collinear" is a constraint.
        # If we violate it, score should be very low.
        return count_4 - 10000 * (max_collinear - 4), line_map, max_collinear
        
    return count_4, line_map, max_collinear

def construct():
    deadline = time.time() + 140
    
    # Initial configurations
    inits = []
    
    # 1. 4x4 Grid
    inits.append([[x, y] for x in range(4) for y in range(4)])
    
    # 2. 5x5 Grid subset (remove 9 points? No, 25-16=9. Maybe remove center 3x3? No, 3x3=9. So keep border? Border is 16 points!)
    # Border of 5x5: (0,0)..(4,4). Points where x=0 or x=4 or y=0 or y=4.
    # Count: 5+5+5+5 - 4 (corners counted twice) = 16.
    border_5x5 = []
    for x in range(5):
        for y in range(5):
            if x == 0 or x == 4 or y == 0 or y == 4:
                border_5x5.append([x, y])
    if len(border_5x5) == 16:
        inits.append(border_5x5)
        
    # 3. 6x5 Grid subset
    # 6x5 = 30 points. Pick 16.
    # Maybe a 4x4 grid in the corner?
    # Or random sample from 6x5
    grid_6x5 = [[x, y] for x in range(6) for y in range(5)]
    random.shuffle(grid_6x5)
    inits.append(grid_6x5[:16])
    
    # 4. Random points in [-10, 10]
    for _ in range(5):
        pts = []
        while len(pts) < 16:
            p = [random.randint(-10, 10), random.randint(-10, 10)]
            if p not in pts:
                pts.append(p)
        inits.append(pts)
        
    best_points = None
    best_score = -1
    best_line_map = None
    
    # Evaluate inits
    for pts in inits:
        s, lm, max_c = _score_and_map(pts)
        if s > best_score:
            best_score = s
            best_points = [p[:] for p in pts]
            best_line_map = lm
            # Note: best_line_map needs to be updated as we move points, 
            # so we shouldn't just store the initial one if we modify best_points.
            # We will recompute or update incrementally.
            # For simplicity in the loop below, we will recompute score from scratch 
            # if we don't implement incremental update, OR we implement incremental.
            # Given the complexity, let's try incremental update in the main loop.
            # But for the initial selection, full recompute is fine.
            
    if best_points is None:
        # Fallback
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score, best_line_map, _ = _score_and_map(best_points)

    # Local Search with Incremental Updates
    # We need to maintain the line_map consistent with current_points.
    current_points = [p[:] for p in best_points]
    current_score, current_line_map, _ = _score_and_map(current_points)
    
    # Helper to update score and map when moving point i from old_pos to new_pos
    def move_point(points, line_map, score, i, new_pos):
        old_pos = points[i]
        if old_pos == new_pos:
            return score, line_map, False
        if [new_pos[0], new_pos[1]] in points:
            return score, line_map, False
            
        # 1. Remove i from all lines containing it
        lines_to_update = []
        for key, pts_set in line_map.items():
            if i in pts_set:
                lines_to_update.append(key)
                
        for key in lines_to_update:
            pts_set = line_map[key]
            old_size = len(pts_set)
            if old_size == 4:
                score -= 1
            # Remove i
            pts_set.discard(i)
            if len(pts_set) == 0:
                del line_map[key]
            else:
                # Check if new size is 4
                if len(pts_set) == 4:
                    score += 1
                # Note: We don't track 5s explicitly in score unless we penalize.
                # The task requires NO 5 collinear.
                # So if we create a 5-collinear, we must penalize.
                # Let's track max_collinear or just check at the end?
                # No, we need to avoid it during search.
                # If we create a 5-collinear, the score should drop.
                # Let's modify score logic: 
                # Score = count_4 - 10000 * count_5_plus
                # So we need to track count_5_plus or similar.
                
        # This logic is getting messy with just 'score'. 
        # Let's return (score, line_map, valid) where valid is False if 5+ collinear.
        # Actually, let's just recompute the score from line_map after the move.
        # It's O(L) where L is number of lines. L is ~ N^2/2 = 120.
        # Recomputing score from map is fast.
        
        # 2. Add i to all lines it forms with other points
        for j in range(len(points)):
            if i == j:
                continue
            key = _line_key(new_pos, points[j])
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
            
        # Update point position
        points[i] = new_pos
        
        # Recalculate score from map
        count_4 = 0
        max_c = 0
        for key, pts_set in line_map.items():
            k = len(pts_set)
            if k == 4:
                count_4 += 1
            if k > max_c:
                max_c = k
                
        if max_c >= 5:
            new_score = count_4 - 10000 * (max_c - 4)
        else:
            new_score = count_4
            
        return new_score, line_map, True

    # Main Loop
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline and T > T_min:
        # Pick a random point
        i = random.randint(0, 15)
        old_x, old_y = current_points[i]
        
        # Propose move
        # Random step in [-5, 5]
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        new_x = old_x + dx
        new_y = old_y + dy
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            T *= cooling
            continue
            
        new_pos = [new_x, new_y]
        
        # Check duplicate
        if new_pos in current_points:
            T *= cooling
            continue
            
        # Perform move on a copy? Or undo?
        # Since move_point modifies in place, we need to be careful.
        # Let's make a deep copy of line_map? Sets are cheap to copy?
        # Or just revert if rejected.
        
        # Save old state for revert
        old_pos = current_points[i][:]
        # Copying line_map is expensive. 
        # Alternative: Do the move, get new score. If rejected, revert.
        # Reverting requires undoing the move.
        
        # Let's just recompute from scratch for simplicity and safety, 
        # but limit the number of evaluations.
        # With 140s, we can do ~100,000 full recomputes if each takes 1ms.
        # Full recompute: 256 keys + dict build + count. ~10-20µs? 
        # 100,000 * 20µs = 2s. 
        # So we can do 100,000 full recomputes easily.
        # Let's stick to full recompute for robustness.
        
        # Generate candidate
        candidate = [p[:] for p in current_points]
        candidate[i] = new_pos
        
        new_score, _, _ = _score_and_map(candidate)
        
        if new_score > current_score or random.random() < math.exp((new_score - current_score) / T):
            current_points = candidate
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        # else: keep current
        
        T *= cooling
        
    # Final local refinement on best_points
    # Try all small moves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(16):
            old_x, old_y = best_points[i]
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    nx = old_x + dx
                    ny = old_y + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    if [nx, ny] in best_points:
                        continue
                    candidate = [p[:] for p in best_points]
                    candidate[i] = [nx, ny]
                    s, _, _ = _score_and_map(candidate)
                    if s > best_score:
                        best_score = s
                        best_points = candidate
                        improved = True
                    if time.time() > deadline:
                        return [[int(x), int(y)] for x, y in best_points]
                        
    return [[int(x), int(y)] for x, y in best_points]
