import random
import math
import time

def _score(points):
    """
    Returns the number of lines containing exactly 4 points.
    Returns 0 if any line contains 5 or more points, or if points are invalid.
    """
    n = len(points)
    if n != 16:
        return 0
    
    # Check duplicates
    if len(set(tuple(p) for p in points)) != n:
        return 0
        
    # Check bounds
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return 0
            
    # Map line keys to count of points on that line
    # Line key: (A, B, C) for Ax + By + C = 0, normalized
    # Or simpler: for each pair (i,j), find the line, then count.
    
    # To be efficient, we can iterate pairs, determine the line, and then count points on it.
    # But we must count each line only once.
    
    lines = {} # key: line_id, value: count
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Line through (x1,y1) and (x2,y2)
            # A = y2 - y1
            # B = x1 - x2
            # C = x2*y1 - x1*y2
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            # Normalize (A, B, C)
            # Divide by GCD of A, B, C
            # Handle 0s
            g = 0
            for val in [A, B, C]:
                if val != 0:
                    g = math.gcd(abs(g), abs(val))
            if g == 0:
                g = 1
            
            A //= g
            B //= g
            C //= g
            
            # Canonical sign: make first non-zero of A,B positive
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            
            if key not in lines:
                lines[key] = 0
            lines[key] += 1 # This increments for each pair, but we need unique point count
            
    # The above logic is flawed for counting unique points.
    # A line with k points has C(k,2) pairs.
    # If we just count pairs, a line with 4 points has 6 pairs.
    # A line with 5 points has 10 pairs.
    # We need to know the number of DISTINCT points on each line.
    
    # Let's restart the counting logic properly.
    # For each unique line key, we need to count how many of the 16 points lie on it.
    
    exact_4_lines = 0
    invalid = False
    
    for key in lines.keys():
        # Count points on this line
        A, B, C = key
        count = 0
        for x, y in points:
            if A * x + B * y + C == 0:
                count += 1
        
        if count >= 5:
            return 0 # Invalid configuration
        
        if count == 4:
            exact_4_lines += 1
            
    return exact_4_lines

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial points: 4x4 grid
    # Centered to allow movement
    pts = [[x - 2, y - 2] for x in range(4) for y in range(4)]
    
    best_pts = [p[:] for p in pts]
    best_score = _score(best_pts)
    
    current_pts = [p[:] for p in pts]
    current_score = best_score
    
    T = 10.0
    cooling = 0.999
    
    # Pre-generate some random moves? No, just random integers.
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_pts[idx]
        
        # Random step
        # Mix of small and large steps
        if random.random() < 0.1:
            step_x = random.randint(-50, 50)
            step_y = random.randint(-50, 50)
        else:
            step_x = random.randint(-5, 5)
            step_y = random.randint(-5, 5)
            
        new_x = old_x + step_x
        new_y = old_y + step_y
        
        # Bounds check
        if new_x < -1000 or new_x > 1000 or new_y < -1000 or new_y > 1000:
            continue
            
        # Collision check (simple)
        # Check if (new_x, new_y) is already in current_pts (excluding idx)
        collision = False
        for k in range(16):
            if k != idx:
                if current_pts[k][0] == new_x and current_pts[k][1] == new_y:
                    collision = True
                    break
        if collision:
            continue
            
        current_pts[idx] = [new_x, new_y]
        new_score = _score(current_pts)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = [p[:] for p in current_pts]
        else:
            # Simulated Annealing acceptance
            delta = new_score - current_score
            if delta < 0:
                prob = math.exp(delta / T)
                if random.random() > prob:
                    # Reject
                    current_pts[idx] = [old_x, old_y]
                else:
                    # Accept, but score is lower (or same)
                    current_score = new_score
            else:
                # Should not happen if we handled > above, but if equal, accept?
                # If new_score == current_score, delta=0, exp(0)=1, always accept.
                # This helps wander the plateau.
                current_score = new_score
        
        T *= cooling
        
        # If stuck for a while, restart?
        # For now, just keep going.
        
    return best_pts
