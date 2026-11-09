import random
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial configuration: 4x4 grid
    # Shifted to center? No, [0,3] is fine.
    current = [[x, y] for x in range(4) for y in range(4)]
    best = [p[:] for p in current]
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return None # Should not happen for distinct points
            
        g = math.gcd(abs(dx), abs(dy))
        dx /= g
        dy /= g
        
        # Normalize direction
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
            
        # Line equation: A*x + B*y = C
        # A = dy, B = -dx
        A = dy
        B = -dx
        C = A * x1 + B * y1
        
        # Normalize A, B, C
        g_abc = math.gcd(abs(int(A)), abs(int(B)))
        if g_abc == 0:
             g_abc = 1
        # C is integer if A,B,x,y are integers
        C = int(C)
        
        # Ensure unique sign
        if A < 0 or (A == 0 and B < 0):
            A = -A
            B = -B
            C = -C
            
        # Final normalization of A, B, C by their gcd?
        # A and B are coprime. C might share factors.
        # But (A,B,C) and (kA, kB, kC) represent same line.
        # Since A,B coprime, k must be 1 if we normalize A,B first.
        # So (A,B,C) is unique.
        
        return (A, B, C)

    def score(points):
        pair_counts = defaultdict(int)
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                key = get_line_key(points[i], points[j])
                if key is not None:
                    pair_counts[key] += 1
        
        score_val = 0
        for count in pair_counts.values():
            # Solve k(k-1)/2 = count
            # k^2 - k - 2*count = 0
            # k = (1 + sqrt(1 + 8*count)) / 2
            disc = 1 + 8 * count
            sqrt_disc = math.isqrt(disc)
            if sqrt_disc * sqrt_disc == disc:
                k = (1 + sqrt_disc) / 2
                if k == int(k):
                    k = int(k)
                    if k == 4:
                        score_val += 1
                    elif k > 4:
                        return -1 # Invalid: 5+ collinear
        return score_val

    current_score = score(current)
    best_score = current_score
    
    T = 1.0
    T_min = 0.01
    alpha = 0.9995
    
    # Local move: change one coordinate of one point by small delta
    while time.time() < deadline:
        # Propose move
        i = random.randint(0, 15)
        x, y = current[i]
        
        # Random delta
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        
        new_x = x + dx
        new_y = y + dy
        
        # Check bounds
        if new_x < -1000 or new_x > 1000 or new_y < -1000 or new_y > 1000:
            continue
            
        new_points = [p[:] for p in current]
        new_points[i] = [new_x, new_y]
        
        # Check distinctness
        valid = True
        for j in range(16):
            if j == i: continue
            if new_points[j] == new_points[i]:
                valid = False
                break
        
        if not valid:
            continue
            
        new_score = score(new_points)
        
        if new_score == -1:
            # Invalid, reject (or accept with tiny prob? No, hard constraint)
            continue
            
        if new_score > current_score:
            current = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = [p[:] for p in current]
        else:
            delta = new_score - current_score
            if delta == 0:
                # Accept neutral moves to explore
                if random.random() < 0.5:
                    current = new_points
                    current_score = new_score
            else:
                # delta < 0
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current = new_points
                    current_score = new_score
        
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best
