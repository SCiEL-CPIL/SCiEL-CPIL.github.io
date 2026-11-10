import math
import random

def construct():
    from math import gcd
    
    def score(pts):
        # pts is a list of 16 [x, y]
        # Check validity: distinct, bounds, no 5 collinear
        # Return count of 4-point lines, or -1 if invalid
        
        # Check distinct
        if len(set(map(tuple, pts))) != 16:
            return -1000000
            
        # Check bounds
        for p in pts:
            if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                return -1000000

        line_counts = {}
        
        for i in range(16):
            x1, y1 = pts[i]
            for j in range(i+1, 16):
                x2, y2 = pts[j]
                
                A = y2 - y1
                B = x1 - x2
                C = A * x1 + B * y1
                
                # Normalize
                # gcd(A, B, C)
                # Note: gcd(0, 0) is 0, but A and B cannot both be 0 for distinct points.
                g = gcd(gcd(A, B), C)
                if g == 0:
                    g = 1 # Should not happen for distinct points
                
                A_n = A // g
                B_n = B // g
                C_n = C // g
                
                # Canonical sign
                if A_n < 0 or (A_n == 0 and B_n < 0) or (A_n == 0 and B_n == 0 and C_n < 0):
                    A_n = -A_n
                    B_n = -B_n
                    C_n = -C_n
                    
                key = (A_n, B_n, C_n)
                line_counts[key] = line_counts.get(key, 0) + 1
        
        final_score = 0
        for pair_count in line_counts.values():
            # Check if pair_count is a triangular number k*(k-1)/2
            # k=2 -> 1, k=3 -> 3, k=4 -> 6, k=5 -> 10
            if pair_count == 1:
                continue
            elif pair_count == 3:
                continue
            elif pair_count == 6:
                final_score += 1
            elif pair_count >= 10:
                return -1000000 # 5 or more collinear
            else:
                # Other triangular numbers? 
                # 1, 3, 6, 10, 15...
                # If it's not 1, 3, 6, 10, it might be invalid or we missed something.
                # For 16 points, max k is 16.
                # But we return -1 if k>=5.
                pass
        
        return final_score

    # Simulated Annealing
    # Initialize with a random configuration
    
    def generate_valid_start():
        pts = []
        while len(pts) < 16:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        return pts

    current_pts = generate_valid_start()
    current_score = score(current_pts)
    
    best_pts = [p[:] for p in current_pts]
    best_score = current_score
    
    # Annealing parameters
    T = 100.0
    cooling = 0.995
    min_T = 0.1
    max_iter = 50000
    
    for i in range(max_iter):
        if T < min_T:
            break
            
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_point = current_pts[idx][:]
        
        # Move it to a new random position
        new_point = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
        
        # Bounds check (already in range, but good practice)
        # Distinctness check (fast)
        distinct = True
        for j, p in enumerate(current_pts):
            if j != idx and p == new_point:
                distinct = False
                break
        if not distinct:
            continue
            
        current_pts[idx] = new_point
        new_score = score(current_pts)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = [p[:] for p in current_pts]
        else:
            # Accept with probability
            delta = new_score - current_score
            # If new_score is invalid, reject
            if new_score < -100000:
                current_pts[idx] = old_point
                continue
                
            prob = math.exp(delta / T)
            if random.random() < prob:
                current_score = new_score
            else:
                current_pts[idx] = old_point
                
        T *= cooling
        
    return best_pts
