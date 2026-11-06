import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = math.gcd(A, math.gcd(B, C))
        if g == 0:
            return None
        A //= g
        B //= g
        C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def has_4_collinear(pts):
        n = len(pts)
        if n < 4:
            return False
        seen_lines = set()
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is None or key in seen_lines:
                    continue
                seen_lines.add(key)
                A, B, C = key
                count = 0
                for k in range(n):
                    x3, y3 = pts[k]
                    if A * x3 + B * y3 + C == 0:
                        count += 1
                if count >= 4:
                    return True
        return False

    def calculate_score(pts):
        n = len(pts)
        if n < 3:
            return n
        
        # Precompute collinear triples
        # conflicts[i] = list of (j, k) such that i, j, k are collinear
        # We only need to check pairs in the current set against the new point.
        # For branch and bound, we need to know if adding point p to set S creates a triple.
        # This happens if there exist a, b in S such that p, a, b are collinear.
        
        # Precompute for each point i, the list of pairs (j, k) with j < k such that i, j, k collinear.
        # To speed up checking, for each point i, we can store a set of "conflict masks" or just lists.
        # Since n=20, bitmask is efficient.
        
        # Let's precompute a 20x20 matrix or similar?
        # Actually, for the B&B, when we consider adding point idx to current_mask:
        # We need to check if there are any a, b in current_mask such that idx, a, b are collinear.
        # We can precompute for each idx, a list of pairs (a, b) where a < b and idx, a, b are collinear.
        
        triples_with_idx = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                # For each k > j
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        # i, j, k are collinear
                        triples_with_idx[i].append((j, k))
                        triples_with_idx[j].append((i, k))
                        triples_with_idx[k].append((i, j))
                        
        best_score = 0
        
        # Bitmask branch and bound
        # Order points by some heuristic? Maybe degree of collinearity?
        # Points with more collinear triples are "harder" to add? 
        # Actually, points that form many triples are "good" for minimizing score 
        # because they force exclusions. 
        # Let's sort indices by descending number of triples they participate in.
        # This might help pruning: if we include a "dense" point early, we block many others.
        
        order = sorted(range(n), key=lambda i: len(triples_with_idx[i]), reverse=True)
        
        def bbb(step, current_mask, current_size):
            nonlocal best_score
            if elapsed() > deadline - 5:
                return
            if current_size > best_score:
                best_score = current_size
            if step == n:
                return
            
            # Pruning: even if we add all remaining points, can we beat best?
            if current_size + (n - step) <= best_score:
                return
                
            idx = order[step]
            
            # Option 1: Exclude idx
            bbb(step + 1, current_mask, current_size)
            
            # Option 2: Include idx
            # Check if adding idx creates a triple with any two points in current_mask
            valid = True
            for a, b in triples_with_idx[idx]:
                if (current_mask & (1 << a)) and (current_mask & (1 << b)):
                    valid = False
                    break
            
            if valid:
                bbb(step + 1, current_mask | (1 << idx), current_size + 1)
                
        bbb(0, 0, 0)
        return best_score

    def generate_grid_points(size, step, offset_x, offset_y):
        pts = []
        for i in range(size):
            for j in range(size):
                x = offset_x + i * step
                y = offset_y + j * step
                if -100 <= x <= 100 and -100 <= y <= 100:
                    pts.append((x, y))
        return pts

    def greedy_construct(candidate_pool, target_size=20):
        """
        Greedily pick points from candidate_pool to maximize collinear triples
        while maintaining no 4 collinear.
        """
        current_set = []
        current_set_set = set()
        # Precompute lines in the pool? 
        # For each point in pool, we want to know how many triples it forms with current_set.
        
        # To make this fast, we can precompute for the whole pool the lines passing through each point.
        # But current_set changes.
        
        # Let's just iterate.
        # For each candidate p in pool not in current_set:
        #   Count how many pairs (a, b) in current_set are collinear with p.
        #   Also check if adding p creates 4 collinear.
        
        # Optimization: 
        # Maintain a dictionary of lines in current_set? 
        # If p is on a line with 2 points in current_set, it forms a triple.
        # If p is on a line with 3 points in current_set, it creates 4 collinear (invalid).
        
        # Let's precompute for the pool: for each point p, list of other points q in pool collinear with p.
        # This is O(N^2) for pool size N.
        # Then for each step, for each candidate p, we check its neighbors in current_set.
        
        # Actually, simpler:
        # For each candidate p:
        #   For each pair (a, b) in current_set:
        #       if p, a, b collinear:
        #           check if there is a 4th point c in current_set on the same line.
        
        # Since current_set is small (<= 20), and pool is small (<= 25-100), this is fast enough.
        
        while len(current_set) < target_size:
            best_candidate = None
            best_triples = -1
            
            for p in candidate_pool:
                if p in current_set_set:
                    continue
                
                # Check validity: no 4 collinear
                # For each pair in current_set, check if p is collinear and if that line has a 3rd point in current_set
                valid = True
                triples_count = 0
                
                # To avoid O(k^2) for each p, we can group current_set points by line?
                # Or just do O(k^2) since k <= 20.
                
                for i in range(len(current_set)):
                    a = current_set[i]
                    for j in range(i + 1, len(current_set)):
                        b = current_set[j]
                        # Check if p, a, b collinear
                        x1, y1 = a
                        x2, y2 = b
                        xp, yp = p
                        if (x2 - x1) * (yp - y1) == (y2 - y1) * (xp - x1):
                            # They are collinear. Check if there is a 3rd point c in current_set on this line
                            # If yes, adding p makes 4 collinear.
                            # If no, this is a new triple.
                            
                            # Find if any other c in current_set is on this line
                            # We can check against all other points in current_set
                            found_third = False
                            for k in range(len(current_set)):
                                if k == i or k == j:
                                    continue
                                c = current_set[k]
                                xc, yc = c
                                if (x2 - x1) * (yc - y1) == (y2 - y1) * (xc - x1):
                                    found_third = True
                                    break
                            
                            if found_third:
                                valid = False
                                break
                            else:
                                triples_count += 1
                if not valid:
                    continue
                    
                if triples_count > best_triples:
                    best_triples = triples_count
                    best_candidate = p
            
            if best_candidate is None:
                # No valid candidate found? Should not happen if pool is large enough
                # Try to relax or return what we have
                break
                
            current_set.append(best_candidate)
            current_set_set.add(best_candidate)
            
        return current_set

    best_points = None
    best_val = float('inf')
    
    # Strategy 1: Greedy on various grids
    # Grid sizes that have enough points
    for size in range(5, 11):
        if elapsed() > deadline - 20:
            break
            
        # Try different steps and offsets to get integer coordinates within [-100, 100]
        # Step must be integer for integer points? Not necessarily, but points must be integers.
        # So step should be integer.
        
        for step in range(1, 15):
            if elapsed() > deadline - 20:
                break
            
            # Max range is 200. size * step <= 200
            if size * step > 200:
                continue
                
            # Try a few offsets
            for ox in range(-10, 11, 5):
                for oy in range(-10, 11, 5):
                    if elapsed() > deadline - 20:
                        break
                        
                    pts = generate_grid_points(size, step, ox, oy)
                    if len(pts) < 20:
                        continue
                        
                    # Greedy construct
                    cand = greedy_construct(pts, 20)
                    if len(cand) == 20:
                        if has_4_collinear(cand):
                            continue
                        s = calculate_score(cand)
                        if s < best_val:
                            best_val = s
                            best_points = cand
                            
    # Strategy 2: Local search improvement on best_points
    if best_points is not None:
        current_pts = [tuple(p) for p in best_points]
        current_val = best_val
        
        # Try to swap points to improve
        # Define a neighborhood: replace one point with a random point from a local grid or random
        # If we replace a point, we must re-check 4-collinear and recalculate score.
        # Score calculation is expensive (B&B).
        # We can use a heuristic objective for local search:
        # Minimize score, but score is hard to approximate.
        # Maybe just try random swaps and check full score?
        # Given time budget, we can afford a few hundred full score calculations.
        
        for _ in range(50):
            if elapsed() > deadline - 10:
                break
                
            # Pick a random point to replace
            idx = random.randint(0, 19)
            old_p = current_pts[idx]
            
            # Generate a new point
            # Try points near old_p or random
            if random.random() < 0.5:
                # Local move
                new_p = (old_p[0] + random.randint(-5, 5), old_p[1] + random.randint(-5, 5))
            else:
                # Random move
                new_p = (random.randint(-100, 100), random.randint(-100, 100))
                
            if new_p in current_pts:
                continue
            if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                continue
                
            # Create new set
            new_pts = current_pts[:]
            new_pts[idx] = new_p
            
            if has_4_collinear(new_pts):
                continue
                
            s = calculate_score(new_pts)
            if s < current_val:
                current_val = s
                current_pts = new_pts
                best_val = s
                best_points = current_pts
            elif s == current_val and random.random() < 0.1:
                # Accept equal score with small probability to escape local optima?
                # Only if we have time
                if elapsed() < deadline - 30:
                    current_pts = new_pts
                    best_points = current_pts

    # Fallback if no good points found
    if best_points is None:
        # Just return a valid random set
        while True:
            pts = []
            while len(pts) < 20:
                p = (random.randint(-100, 100), random.randint(-100, 100))
                if p not in pts:
                    pts.append(p)
            if not has_4_collinear(pts):
                best_points = pts
                break

    return [[x, y] for x, y in best_points]

if __name__ == "__main__":
    pts = construct()
    print(pts)
    print("Score:", calculate_score(pts))
    print("Has 4 collinear:", has_4_collinear(pts))
