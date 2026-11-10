import random
import time
import math

def construct():
    n = 16
    # Start with 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    def get_score(pts):
        """Returns penalized score. Higher is better.
        Score = (# lines with exactly 4 points) - 100 * (# lines with >= 5 points)
        """
        lines = {}
        
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i+1, n):
                x2, y2 = pts[j]
                
                A = y2 - y1
                B = -(x2 - x1)
                C = x2*y1 - x1*y2
                
                g = math.gcd(A, math.gcd(B, C))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                else:
                    continue
                    
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                    C = -C
                elif A == 0 and B == 0:
                    continue
                
                key = (A, B, C)
                lines[key] = lines.get(key, 0) + 1
        
        score = 0
        penalty = 0
        
        for count in lines.values():
            # k=4 -> 6 pairs
            # k=5 -> 10 pairs
            # k=6 -> 15 pairs
            if count == 6:
                score += 1
            elif count >= 10:
                penalty += 1
                
        return score - 100 * penalty

    def random_point():
        return [random.randint(-1000, 1000), random.randint(-1000, 1000)]

    def egcd(a, b):
        if b == 0:
            return (1, 0, a)
        x1, y1, d = egcd(b, a % b)
        return (y1, x1 - (a // b) * y1, d)

    def get_line_points(pts, A, B, C):
        """Returns list of indices of points on the line Ax+By+C=0"""
        pts_on_line = []
        for i in range(n):
            x, y = pts[i]
            if A*x + B*y + C == 0:
                pts_on_line.append(i)
        return pts_on_line

    def try_complete_line(pts, line_key):
        """
        Try to move a point to complete a 3-point line to a 4-point line.
        line_key: (A, B, C)
        Returns (new_pts, moved_index, new_pos) or None
        """
        A, B, C = line_key
        # Get points on this line
        pts_on_line = get_line_points(pts, A, B, C)
        
        # We only want to complete lines with exactly 3 points
        if len(pts_on_line) != 3:
            return None
            
        # Pick a random point to move (could be one of the 3, or another one)
        # Ideally, move a point that is NOT on this line, to a new spot on this line.
        # Or move one of the 3 to a different spot on the same line? No, that keeps it 3.
        # We need to add a 4th point.
        # So we pick a point k NOT in pts_on_line.
        candidates_to_move = [i for i in range(n) if i not in pts_on_line]
        if not candidates_to_move:
            return None
            
        k = random.choice(candidates_to_move)
        
        # Find a new position for k on the line Ax+By+C=0
        # Use one of the existing points on the line to define the direction
        # Let p1, p2 be two points on the line.
        # Direction vector d = p2 - p1.
        # New point = p1 + t * d.
        
        p1_idx = pts_on_line[0]
        p2_idx = pts_on_line[1]
        x1, y1 = pts[p1_idx]
        x2, y2 = pts[p2_idx]
        
        dx = x2 - x1
        dy = y2 - y1
        
        # General integer solution for points on the line
        # We can just use the parametric form relative to p1
        # x = x1 + t * (dx/g), y = y1 + t * (dy/g) where g = gcd(dx, dy)
        # But we need to ensure (x,y) is integer.
        # If we step by (dx/g, dy/g), we hit all integer points on the line?
        # Yes, if we start from an integer point.
        
        g = math.gcd(dx, dy)
        if g == 0:
            return None
        step_x = dx // g
        step_y = dy // g
        
        # Current t for p1 is 0.
        # We need a t such that the new point is distinct from all existing points.
        # And within bounds.
        
        # Try a range of t values
        # The existing points on the line correspond to specific t values.
        # Let's just try random t in a wider range, say [-20, 20]
        # This ensures we don't pick the same point.
        
        for _ in range(50):
            t = random.randint(-20, 20)
            if t == 0: continue # p1
            # Check if this t corresponds to p2 or p3?
            # We just check distinctness against all points.
            
            nx = x1 + t * step_x
            ny = y1 + t * step_y
            
            if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                # Check distinctness
                new_pos = [nx, ny]
                is_distinct = True
                for l in range(n):
                    if pts[l] == new_pos:
                        is_distinct = False
                        break
                if is_distinct:
                    new_pts = [p[:] for p in pts]
                    new_pts[k] = new_pos
                    return new_pts, k, new_pos
        return None

    start_time = time.time()
    time_limit = 110.0
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    current_pts = points
    current_score = get_score(current_pts)
    best_pts = [p[:] for p in current_pts]
    best_score = current_score
    
    while time.time() - start_time < time_limit:
        new_pts = None
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to complete a 3-line
            # Find all 3-lines
            lines = {}
            for i in range(n):
                x1, y1 = current_pts[i]
                for j in range(i+1, n):
                    x2, y2 = current_pts[j]
                    A = y2 - y1
                    B = -(x2 - x1)
                    C = x2*y1 - x1*y2
                    g = math.gcd(A, math.gcd(B, C))
                    if g != 0:
                        A //= g; B //= g; C //= g
                    if A < 0 or (A == 0 and B < 0):
                        A = -A; B = -B; C = -C
                    key = (A, B, C)
                    lines[key] = lines.get(key, 0) + 1
            
            three_lines = [key for key, count in lines.items() if count == 3]
            if three_lines:
                line_key = random.choice(three_lines)
                res = try_complete_line(current_pts, line_key)
                if res:
                    new_pts, _, _ = res
            else:
                # Fallback to random move
                k = random.randint(0, n-1)
                new_pts = [p[:] for p in current_pts]
                new_pts[k] = random_point()
                
        elif move_type < 0.7:
            # Random move
            k = random.randint(0, n-1)
            new_pts = [p[:] for p in current_pts]
            new_pts[k] = random_point()
            
        else:
            # Local move
            k = random.randint(0, n-1)
            new_pts = [p[:] for p in current_pts]
            new_pts[k][0] += random.randint(-10, 10)
            new_pts[k][1] += random.randint(-10, 10)
            new_pts[k][0] = max(-1000, min(1000, new_pts[k][0]))
            new_pts[k][1] = max(-1000, min(1000, new_pts[k][1]))
        
        if new_pts is None:
            continue
            
        new_score = get_score(new_pts)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = [p[:] for p in current_pts]
        
        T = max(T_min, T * cooling_rate)
        
    # Final check: ensure best_pts is valid (no 5+ collinear)
    # The penalty function might have allowed a state with 5+ collinear if the score was high enough?
    # No, penalty is -100 per 5-line. Score is max ~15. So any 5-line makes score <= -100.
    # So the best state should be valid.
    # But to be safe, we can verify.
    
    return best_pts
