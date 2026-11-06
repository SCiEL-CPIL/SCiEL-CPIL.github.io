import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # Grid bounds for search
    GRID_MIN = -10
    GRID_MAX = 10
    
    # Initial configuration: 4x4 grid centered at 0
    # This gives score 10
    def initial_config():
        pts = []
        for x in range(-2, 2):
            for y in range(-2, 2):
                pts.append([x, y])
        return pts

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        # A = y1 - y2
        # B = x2 - x1
        # C = x1*y2 - x2*y1
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        
        # Normalize by GCD to get unique key for parallel lines
        if A == 0 and B == 0:
            return None # Should not happen for distinct points
            
        g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
        if g == 0:
            g = 1
            
        A //= g
        B //= g
        C //= g
        
        # Ensure consistent sign (e.g., first non-zero is positive)
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
            
        return (A, B, C)

    def evaluate(points):
        """
        Returns (score, is_valid)
        score: number of lines with exactly 4 points
        is_valid: False if any line has >= 5 points or points are not distinct
        """
        n = len(points)
        # Check distinctness
        if len(set(tuple(p) for p in points)) != n:
            return 0, False
            
        line_points = defaultdict(set)
        
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                if key is None:
                    return 0, False
                line_points[key].add(i)
                line_points[key].add(j)
        
        score = 0
        for key, pts in line_points.items():
            count = len(pts)
            if count >= 5:
                return 0, False
            if count == 4:
                score += 1
                
        return score, True

    # Simulated Annealing
    best_points = initial_config()
    best_score, _ = evaluate(best_points)
    
    current_points = [p[:] for p in best_points]
    current_score, _ = evaluate(current_points)
    
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    # Pre-generate some random moves or just use random
    while time.time() < deadline:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.7:
            # Shift a point
            idx = random.randint(0, 15)
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            new_points = [p[:] for p in current_points]
            new_points[idx][0] += dx
            new_points[idx][1] += dy
            
            # Check bounds
            if abs(new_points[idx][0]) > 1000 or abs(new_points[idx][1]) > 1000:
                continue
            # Check if it stays within our search grid to maintain density
            # If it goes out of [-10, 10], it's likely less useful, but let's allow it 
            # but penalize or just reject if too far to keep search local
            if abs(new_points[idx][0]) > GRID_MAX or abs(new_points[idx][1]) > GRID_MAX:
                 # Optional: allow escape but less frequently
                 if random.random() > 0.1:
                     continue
                     
        elif move_type < 0.9:
            # Swap coordinates of two points
            i, j = random.sample(range(16), 2)
            new_points = [p[:] for p in current_points]
            new_points[i], new_points[j] = new_points[j], new_points[i]
            
        else:
            # Replace a point with a random point in grid
            idx = random.randint(0, 15)
            new_points = [p[:] for p in current_points]
            new_points[idx] = [random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX)]

        new_score, is_valid = evaluate(new_points)
        
        if not is_valid:
            # Reject invalid moves
            # Optionally, we could try to repair, but simple rejection is robust
            pass
        else:
            delta = new_score - current_score
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_points = new_points
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
        
        T *= cooling
        if T < min_T:
            T = min_T

    return best_points

# EVOLVE-BLOCK-END
