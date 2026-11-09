import time
import random
import math
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def is_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def line_key(p1, p2):
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
        a = dy
        b = -dx
        c = -(dy * x1 - dx * y1)
        g2 = math.gcd(abs(a), math.gcd(abs(b), abs(c)))
        if g2 > 0:
            a //= g2
            b //= g2
            c //= g2
        if a < 0 or (a == 0 and (b < 0 or (b == 0 and c < 0))):
            a, b, c = -a, -b, -c
        return (a, b, c)

    def count_triples(points):
        """
        Returns (num_triples, max_collinear)
        If max_collinear >= 4, it's invalid.
        """
        lines = defaultdict(set)
        for p1, p2 in combinations(points, 2):
            k = line_key(p1, p2)
            if k is None: continue
            lines[k].add(p1)
            lines[k].add(p2)
        
        total = 0
        max_c = 0
        for k, pts in lines.items():
            n = len(pts)
            if n > max_c:
                max_c = n
            if n == 3:
                total += 1
            elif n > 3:
                # Invalid configuration
                return -1, n
        return total, max_c

    def max_3_free_subset(points):
        """
        Exact branch and bound for the largest subset with no three collinear.
        Returns the size of the subset.
        """
        n = len(points)
        if n <= 2:
            return n
        
        # Precompute collinear masks
        # collinear_with_pair[i][j] is a bitmask of points k such that i, j, k are collinear
        collinear_with_pair = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                mask = 0
                for k in range(n):
                    if k != i and k != j and is_collinear(points[i], points[j], points[k]):
                        mask |= (1 << k)
                collinear_with_pair[i][j] = mask
                collinear_with_pair[j][i] = mask
        
        best = [0]
        
        # Helper to check if adding point i to subset_mask is valid
        # A point i is valid if for all pairs (a, b) in subset_mask, i is not collinear with a and b.
        # This is equivalent to: for all a in subset_mask, (collinear_with_pair[a][i] & subset_mask) == 0
        # Note: We don't need to check pairs involving i yet, only pairs already in subset.
        
        def backtrack(i, subset_mask, size):
            if size > best[0]:
                best[0] = size
            if i == n:
                return
            
            # Upper bound
            if size + (n - i) <= best[0]:
                return
            
            # Option 1: exclude point i
            backtrack(i + 1, subset_mask, size)
            
            # Option 2: include point i
            can_add = True
            temp_mask = subset_mask
            while temp_mask:
                a_bit = temp_mask & (-temp_mask)
                a = a_bit.bit_length() - 1
                if collinear_with_pair[a][i] & subset_mask:
                    can_add = False
                    break
                temp_mask ^= a_bit
            
            if can_add:
                backtrack(i + 1, subset_mask | (1 << i), size + 1)
        
        backtrack(0, 0, 0)
        return best[0]

    def generate_initial_points():
        """
        Generate a set of 20 points that are valid (no 4 collinear) 
        and have a high number of 3-collinear triples.
        We use a grid-like structure with perturbations.
        """
        random.seed(42)
        
        # Try several grid configurations
        best_pts = None
        best_triples = -1
        
        for attempt in range(10):
            # Define a grid
            # e.g. 5x4 or 4x5
            # Let's try 5x4
            xs = random.sample(range(-20, 21), 5)
            ys = random.sample(range(-20, 21), 4)
            
            # Shuffle to avoid obvious patterns?
            # Actually, we want to create lines.
            # Let's just take the Cartesian product.
            pts = []
            for x in xs:
                for y in ys:
                    pts.append((x, y))
            
            # Perturb
            for i in range(20):
                if random.random() < 0.5:
                    pts[i] = (pts[i][0] + random.randint(-2, 2), pts[i][1] + random.randint(-2, 2))
            
            # Ensure distinct
            unique_pts = []
            for p in pts:
                if p not in unique_pts:
                    unique_pts.append(p)
            
            # If we lost points due to collisions, refill
            while len(unique_pts) < 20:
                p = (random.randint(-50, 50), random.randint(-50, 50))
                if p not in unique_pts:
                    unique_pts.append(p)
            
            pts = unique_pts[:20]
            
            # Check validity and count triples
            triples, max_c = count_triples(pts)
            if triples != -1 and triples > best_triples:
                best_triples = triples
                best_pts = pts[:]
                
        if best_pts is None:
            # Fallback: Random
            best_pts = []
            for _ in range(20):
                p = (random.randint(-100, 100), random.randint(-100, 100))
                while p in best_pts:
                    p = (random.randint(-100, 100), random.randint(-100, 100))
                best_pts.append(p)
            # Fix if invalid
            while True:
                triples, max_c = count_triples(best_pts)
                if triples != -1:
                    break
                # Move a random point
                idx = random.randint(0, 19)
                best_pts[idx] = (random.randint(-100, 100), random.randint(-100, 100))
                # Ensure distinct
                for i in range(20):
                    if i != idx and best_pts[i] == best_pts[idx]:
                         best_pts[idx] = (best_pts[idx][0] + 1, best_pts[idx][1])
                         
        return best_pts

    # 1. Initialize
    current_pts = generate_initial_points()
    current_triples, _ = count_triples(current_pts)
    
    best_pts = current_pts[:]
    best_score = max_3_free_subset(current_pts)
    
    # 2. Local Search (Simulated Annealing style)
    # Objective: Maximize triples (proxy for minimizing score)
    # Constraint: No 4 collinear
    
    temp = 5.0
    min_temp = 0.5
    cooling = 0.99
    
    while time.time() < deadline and temp > min_temp:
        # Pick a random point
        idx = random.randint(0, 19)
        old_pt = current_pts[idx]
        
        # Generate new point
        # Bias towards nearby points to maintain structure?
        # Or global random?
        # Let's try a mix.
        if random.random() < 0.5:
             # Local move
             new_pt = (old_pt[0] + random.randint(-5, 5), old_pt[1] + random.randint(-5, 5))
        else:
             # Global move
             new_pt = (random.randint(-100, 100), random.randint(-100, 100))
        
        # Ensure distinct
        is_dup = False
        for i in range(20):
            if i != idx and current_pts[i] == new_pt:
                is_dup = True
                break
        if is_dup:
            continue
            
        # Ensure bounds
        if abs(new_pt[0]) > 100 or abs(new_pt[1]) > 100:
            continue

        # Apply move
        current_pts[idx] = new_pt
        
        new_triples, max_c = count_triples(current_pts)
        
        if new_triples == -1:
            # Invalid
            current_pts[idx] = old_pt
            continue
            
        delta = new_triples - current_triples
        
        if delta >= 0:
            # Accept
            current_triples = new_triples
            # Check score
            # Score calculation is expensive, so only do it if we improved triples significantly
            # or periodically?
            # Since score is the actual metric, we should check it when we find a new local optimum or periodically.
            # But checking score every step is too slow.
            # Let's check score if we improved triples, or every 50 steps.
            
            if delta > 0 or random.random() < 0.05:
                 score = max_3_free_subset(current_pts)
                 if score < best_score:
                     best_score = score
                     best_pts = current_pts[:]
        else:
            # Worse
            prob = math.exp(delta / temp)
            if random.random() < prob:
                current_triples = new_triples
            else:
                current_pts[idx] = old_pt
        
        temp *= cooling
        
        # Safety check
        if time.time() > deadline - 10:
            break

    # Final check
    final_score = max_3_free_subset(best_pts)
    if final_score < best_score:
        best_score = final_score

    return [[p[0], p[1]] for p in best_pts]
