import random
import math
from collections import defaultdict
import time

def construct():
    # Start with a 4x4 grid
    # Scale it to be within bounds and centered
    # 0..3 is fine, but let's center it to -1.5..1.5? No, integers.
    # Let's use -2, -1, 0, 1
    base = [-2, -1, 0, 1]
    points = [[x, y] for x in base for y in base]
    
    best_points = points[:]
    best_score = -1
    
    # Function to count lines with exactly 4 points
    # Returns (score, is_valid)
    def evaluate(pts):
        # Check for 5+ collinear first
        line_counts = defaultdict(int)
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                # Line equation: (y2 - y1)x - (x2 - x1)y + (x2 - x1)y1 - (y2 - y1)x2 = 0
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                
                # Normalize
                g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                
                # Sign convention
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                
                line_counts[(A, B, C)] += 1
        
        # Check for invalid lines (5+ points)
        # A line with k points contributes C(k, 2) pairs.
        # If C(k, 2) > C(4, 2) = 6, then k >= 5.
        # Actually, just check the max count.
        # But we need the count of lines with EXACTLY 4 points.
        
        score = 0
        for key, pairs in line_counts.items():
            # pairs = k * (k-1) / 2
            # We need to find k.
            # k=2 -> 1 pair
            # k=3 -> 3 pairs
            # k=4 -> 6 pairs
            # k=5 -> 10 pairs
            # k=6 -> 15 pairs
            if pairs == 6:
                score += 1
            elif pairs > 6:
                # This implies k >= 5
                return -1, False
        
        return score, True

    cur_score, cur_valid = evaluate(points)
    if cur_valid:
        best_score = cur_score
        best_points = points[:]
    else:
        # Grid should be valid, but just in case
        best_score = 0
        best_points = points[:]

    start_time = time.time()
    timeout = 110  # Leave 10s for overhead
    
    # Hill Climbing
    # Neighbors: 8 directions + stay? No, just move to a new spot.
    # To avoid getting stuck, we can allow "random jumps" occasionally.
    
    for _ in range(100000):
        if time.time() - start_time > timeout:
            break
        
        # Pick a random point
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        # Try a random move within a small radius
        # Radius 1 is too small to escape grid structure?
        # Radius 2 or 3 might be better.
        # Let's try radius 2.
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        if dx == 0 and dy == 0:
            continue
            
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Bounds check
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        # Check if point already exists
        # For efficiency, we can just swap and check validity, 
        # but duplicate points are invalid (distinct points required).
        # Simple check:
        is_dup = False
        for i in range(16):
            if i != idx and points[i][0] == new_x and points[i][1] == new_y:
                is_dup = True
                break
        if is_dup:
            continue
            
        # Perform move
        points[idx] = [new_x, new_y]
        
        new_score, new_valid = evaluate(points)
        
        if new_valid:
            if new_score > best_score:
                best_score = new_score
                best_points = points[:]
            # Accept move if it improves current score or with some probability?
            # For simplicity, strict hill climbing on current state, 
            # but we keep track of best global.
            # If new_score > cur_score:
            #    cur_score = new_score
            # else:
            #    # revert
            #    points[idx] = [old_x, old_y]
            
            # To escape local maxima, let's use a simple simulated annealing approach
            # or just revert if not better.
            if new_score <= cur_score:
                points[idx] = [old_x, old_y]
            else:
                cur_score = new_score
        else:
            # Revert
            points[idx] = [old_x, old_y]

    return best_points
