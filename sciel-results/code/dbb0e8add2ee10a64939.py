import time
import random
from collections import defaultdict
from math import gcd

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for 150s limit
    
    # Initial configuration: 4x4 grid
    # This gives 10 lines of exactly 4 points.
    # We will try to improve this.
    
    def get_points():
        # Start with a 4x4 grid, centered to allow movement
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i - 1.5, j - 1.5]) # Not integer, let's use integers
        # Let's use integers directly
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i, j])
        return pts

    def canonical_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return (0, 0, 0) # Should not happen for distinct points
        dx //= g
        dy //= g
        # Normalize direction
        if dx < 0:
            dx = -dx
            dy = -dy
        elif dx == 0 and dy < 0:
            dy = -dy
        # Line equation: ax + by = c
        # a = dy, b = -dx
        a = dy
        b = -dx
        c = a * x1 + b * y1
        return (a, b, c)

    def score(points):
        """
        Returns (num_4_lines, num_5_plus_lines, details)
        We want to maximize num_4_lines and minimize num_5_plus_lines.
        """
        line_counts = defaultdict(int)
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                key = canonical_line(points[i], points[j])
                line_counts[key] += 1
        
        num_4 = 0
        num_5_plus = 0
        for key, count in line_counts.items():
            # Count is the number of pairs on this line.
            # If k points are collinear, number of pairs is k*(k-1)/2.
            # We need to map pair count back to point count.
            # Pairs: 1->2, 3->3, 6->4, 10->5, 15->6, ...
            # We only care about k=4 (6 pairs) and k>=5 (10+ pairs).
            if count == 6:
                num_4 += 1
            elif count >= 10:
                num_5_plus += 1
        return num_4, num_5_plus

    def evaluate(points):
        """
        Score: High is good.
        Penalty for 5+ collinear is severe.
        """
        n4, n5 = score(points)
        # We want to maximize n4.
        # If n5 > 0, it's invalid for the "no 5 collinear" constraint.
        # The task says "no five collinear". So n5 must be 0.
        # Score = n4 - 1000 * n5
        return n4 - 1000 * n5

    best_points = get_points()
    best_score = evaluate(best_points)
    
    # Local Search
    # Move one point at a time to a random nearby integer coordinate.
    # Or swap two points?
    # Let's try moving one point.
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Precompute some random candidates?
    # Just use random jumps.
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Try a few random moves
        improved = False
        for _ in range(10):
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
            
            # Check bounds
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
            
            # Check distinctness
            # Simple check: is (new_x, new_y) already in current_points?
            # We can maintain a set for O(1) lookup, but 16 points is small.
            is_distinct = True
            for k, p in enumerate(current_points):
                if k != idx and p[0] == new_x and p[1] == new_y:
                    is_distinct = False
                    break
            
            if not is_distinct:
                continue
                
            # Apply move
            current_points[idx] = [new_x, new_y]
            new_score = evaluate(current_points)
            
            if new_score > current_score:
                current_score = new_score
                improved = True
                # If we found a better score, update best
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
                break # Restart search from this new state
            else:
                # Revert move
                current_points[idx] = [old_x, old_y]
        
        # If no improvement in 10 tries, maybe do a larger jump or restart?
        # Let's just continue.
        
        # To avoid getting stuck, occasionally restart from best?
        # Or just keep moving.
        
    # Ensure final result is valid (no 5 collinear)
    # If best_score < 0, it means we had 5 collinear.
    # But we started with a valid grid (score 10).
    # So best_score should be >= 10.
    
    return best_points

# Test briefly
if __name__ == "__main__":
    pts = construct()
    print(pts)
    print(len(pts))
    # Verify no 5 collinear
    from collections import defaultdict
    def canonical_line(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        from math import gcd
        g = gcd(abs(dx), abs(dy))
        if g == 0: return (0,0,0)
        dx //= g
        dy //= g
        if dx < 0:
            dx = -dx
            dy = -dy
        elif dx == 0 and dy < 0:
            dy = -dy
        a = dy
        b = -dx
        c = a * x1 + b * y1
        return (a, b, c)
    
    line_counts = defaultdict(int)
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            line_counts[canonical_line(pts[i], pts[j])] += 1
            
    max_collinear = 0
    for k, v in line_counts.items():
        # v is pairs. k points -> k(k-1)/2 pairs
        # 1->2, 3->3, 6->4, 10->5
        if v == 10: max_collinear = max(max_collinear, 5)
        elif v >= 15: max_collinear = max(max_collinear, 6)
        elif v == 6: max_collinear = max(max_collinear, 4)
        elif v == 3: max_collinear = max(max_collinear, 3)
        elif v == 1: max_collinear = max(max_collinear, 2)
        
    print("Max collinear:", max_collinear)
    # Count 4-lines
    count_4 = 0
    for k, v in line_counts.items():
        if v == 6:
            count_4 += 1
    print("4-lines:", count_4)
