import random
import math
from collections import defaultdict
import time
import numpy as np

def construct():
    # Initialize with 4x4 grid
    # Points are (x, y)
    base = [-2, -1, 0, 1]
    points = np.array([[x, y] for x in base for y in base], dtype=int)
    
    best_points = points.copy()
    best_score = -1
    
    def evaluate(pts):
        # pts is a numpy array of shape (16, 2)
        n = len(pts)
        line_counts = defaultdict(int)
        
        # To speed up, we can use a set of tuples for points
        # But we need to count lines.
        # O(N^2) pairs. For each pair, compute line key.
        # N=16, so 120 pairs. This is very fast.
        
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                
                # Line equation: A*x + B*y + C = 0
                # A = y2 - y1
                # B = x1 - x2
                # C = x2*y1 - x1*y2
                
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                
                # Normalize
                g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                else:
                    # Should not happen for distinct points
                    continue
                
                # Sign convention
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                
                line_counts[(A, B, C)] += 1
        
        score = 0
        for key, pairs in line_counts.items():
            # pairs = k * (k-1) / 2
            # We want lines with EXACTLY 4 points.
            # k=4 -> 6 pairs.
            if pairs == 6:
                score += 1
            elif pairs > 6:
                # k >= 5, invalid
                return -1
        
        return score

    # Evaluate initial
    cur_score = evaluate(points)
    if cur_score > best_score:
        best_score = cur_score
        best_points = points.copy()
    
    # Simulated Annealing
    start_time = time.time()
    timeout = 110
    T = 10.0
    cooling = 0.9999
    min_T = 0.01
    
    # Pre-compute a list of possible moves?
    # Or just random jumps.
    
    while time.time() - start_time < timeout:
        if T < min_T:
            break
            
        # Pick a random point
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        # Random move
        # Use a Gaussian-like distribution or uniform in a box
        # Let's use uniform in [-5, 5]
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        
        if dx == 0 and dy == 0:
            continue
            
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Bounds check
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        # Check duplicate
        is_dup = False
        for i in range(16):
            if i != idx and points[i][0] == new_x and points[i][1] == new_y:
                is_dup = True
                break
        if is_dup:
            continue
            
        # Apply move
        points[idx] = [new_x, new_y]
        
        new_score = evaluate(points)
        
        # Update best
        if new_score > best_score:
            best_score = new_score
            best_points = points.copy()
            
        # Acceptance
        if new_score >= cur_score:
            cur_score = new_score
        else:
            # Delta E is cur_score - new_score (positive)
            delta = cur_score - new_score
            # Accept with probability exp(-delta / T)
            if random.random() < math.exp(-delta / T):
                cur_score = new_score
            else:
                # Revert
                points[idx] = [old_x, old_y]
                
        T *= cooling

    # Convert numpy array to list of lists
    return best_points.tolist()
