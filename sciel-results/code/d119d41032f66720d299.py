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

    def calculate_score_fast(pts):
        """Faster score calculation using bitmask-based B&B with precomputed conflict masks."""
        n = len(pts)
        if n < 3:
            return n

        # Precompute for each point, which pairs of other points form a collinear triple with it
        # conflict_mask[i] = bitmask of points that cannot be in the same independent set as i
        # Actually, we need: for each point i, if i is selected, which other points j,k are forbidden together
        # Better: for each point i, build a bitmask of "conflict partners" - if i is in set, 
        # then for every line through i with two other points, those two are mutually exclusive
        
        # Alternative faster approach: precompute all collinear triples as bitmasks
        # For each point i, compute mask_i = set of points j such that there exists k where i,j,k collinear
        # But we need: if i is selected, then for any pair (j,k) collinear with i, we can't select both j and k
        
        # Let's use the approach: for each point i, precompute a list of (j, k) pairs that are collinear with i
        # Then in B&B, when we add i, we need to check that no such (j,k) pair is both already selected
        
        # Even faster: use a different formulation. 
        # Precompute: for each pair (i,j), is there a k such that i,j,k collinear? 
        # If yes, then i and j cannot both be in the independent set.
        # This gives us a graph where edges represent "cannot be together"
        # Then the problem is Maximum Independent Set on this graph!
        
        # Build conflict graph: edge (i,j) exists if there's a point k collinear with both i and j
        conflict = [0] * n  # bitmask of conflicting points
        
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                # Check if any other point is collinear with i and j
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)
                        break
        
        best_score = 0
        
        # Order by degree (number of conflicts) descending - more constrained first
        order = sorted(range(n), key=lambda i: bin(conflict[i]).count('1'), reverse=True)
        
        def bbb(step, current_mask, current_size):
            nonlocal best_score
            if elapsed() > deadline - 5:
                return
            if current_size > best_score:
                best_score = current_size
            if step == n:
                return
            # Upper bound: current size + remaining points
            if current_size + (n - step) <= best_score:
                return
            idx = order[step]
            # Try not selecting idx
            bbb(step + 1, current_mask, current_size)
            # Try selecting idx if it doesn't conflict with any selected point
            if (conflict[idx] & current_mask) == 0:
                bbb(step + 1, current_mask | (1 << idx), current_size + 1)
                
        bbb(0, 0, 0)
        return best_score

    def greedy_construct(candidate_pool, target_size=20):
        current_set = []
        current_set_set = set()
        
        while len(current_set) < target_size:
            best_candidate = None
            best_triples = -1
            
            for p in candidate_pool:
                if p in current_set_set:
                    continue
                
                valid = True
                triples_count = 0
                
                for i in range(len(current_set)):
                    a = current_set[i]
                    for j in range(i + 1, len(current_set)):
                        b = current_set[j]
                        x1, y1 = a
                        x2, y2 = b
                        xp, yp = p
                        if (x2 - x1) * (yp - y1) == (y2 - y1) * (xp - x1):
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
                break
                
            current_set.append(best_candidate)
            current_set_set.add(best_candidate)
            
        return current_set

    def gen_parabola_points():
        """Generate points on a perturbed parabola y = ax^2 + bx + c."""
        pts = []
        # Use different parabolas to avoid collinearity
        for a in [1, 2, 3]:
            for b in range(-5, 6, 2):
                for c in range(-20, 21, 5):
                    for x in range(-10, 11):
                        y = a * x * x + b * x + c
                        if -100 <= y <= 100:
                            p = (x, y)
                            if p not in pts:
                                pts.append(p)
        return pts

    def gen_random_perturbed_grid(size, base_step):
        """Generate a grid with random perturbations to break collinearity."""
        pts = []
        for i in range(size):
            for j in range(size):
                x = i * base_step + random.randint(-2, 2)
                y = j * base_step + random.randint(-2, 2)
                if -100 <= x <= 100 and -100 <= y <= 100:
                    p = (x, y)
                    if p not in pts:
                        pts.append(p)
        return pts

    best_points = None
    best_val = float('inf')
    
    # Strategy 1: Greedy on various grids with more diverse parameters
    for size in range(5, 12):
        if elapsed() > deadline - 30:
            break
            
        for step in range(1, 15):
            if elapsed() > deadline - 30:
                break
            if size * step > 200:
                continue
                
            for ox in range(-15, 16, 5):
                for oy in range(-15, 16, 5):
                    if elapsed() > deadline - 30:
                        break
                        
                    pts = []
                    for i in range(size):
                        for j in range(size):
                            x = ox + i * step
                            y = oy + j * step
                            if -100 <= x <= 100 and -100 <= y <= 100:
                                pts.append((x, y))
                    
                    if len(pts) < 20:
                        continue
                        
                    cand = greedy_construct(pts, 20)
                    if len(cand) == 20:
                        if has_4_collinear(cand):
                            continue
                        s = calculate_score_fast(cand)
                        if s < best_val:
                            best_val = s
                            best_points = [tuple(p) for p in cand]
    
    # Strategy 2: Parabola-based initialization
    if elapsed() < deadline - 25:
        para_pts = gen_parabola_points()
        if len(para_pts) >= 20:
            cand = greedy_construct(para_pts, 20)
            if len(cand) == 20:
                if not has_4_collinear(cand):
                    s = calculate_score_fast(cand)
                    if s < best_val:
                        best_val = s
                        best_points = [tuple(p) for p in cand]
    
    # Strategy 3: Random initialization with local search
    if elapsed() < deadline - 20:
        for _ in range(15):
            if elapsed() > deadline - 15:
                break
            pts = []
            while len(pts) < 20:
                p = (random.randint(-100, 100), random.randint(-100, 100))
                if p not in pts:
                    pts.append(p)
            if not has_4_collinear(pts):
                s = calculate_score_fast(pts)
                if s < best_val:
                    best_val = s
                    best_points = [tuple(p) for p in pts]
    
    # Strategy 4: Perturbed grid initialization
    if elapsed() < deadline - 20:
        for _ in range(10):
            if elapsed() > deadline - 15:
                break
            size = random.randint(5, 8)
            step = random.randint(3, 8)
            grid_pts = gen_random_perturbed_grid(size, step)
            if len(grid_pts) >= 20:
                cand = greedy_construct(grid_pts, 20)
                if len(cand) == 20:
                    if not has_4_collinear(cand):
                        s = calculate_score_fast(cand)
                        if s < best_val:
                            best_val = s
                            best_points = [tuple(p) for p in cand]
    
    # Strategy 5: Local search improvement (simulated annealing style)
    if best_points is not None:
        current_pts = list(best_points)
        current_val = best_val
        
        max_iterations = 500
        for iteration in range(max_iterations):
            if elapsed() > deadline - 10:
                break
                
            idx = random.randint(0, 19)
            old_p = current_pts[idx]
            
            # Diverse move generation
            r = random.random()
            if r < 0.3:
                # Small local move
                new_p = (old_p[0] + random.randint(-2, 2), old_p[1] + random.randint(-2, 2))
            elif r < 0.6:
                # Medium local move
                new_p = (old_p[0] + random.randint(-8, 8), old_p[1] + random.randint(-8, 8))
            elif r < 0.85:
                # Larger local move
                new_p = (old_p[0] + random.randint(-20, 20), old_p[1] + random.randint(-20, 20))
            else:
                # Random global move
                new_p = (random.randint(-100, 100), random.randint(-100, 100))
                
            if new_p in current_pts:
                continue
            if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                continue
                
            new_pts = current_pts[:]
            new_pts[idx] = new_p
            
            if has_4_collinear(new_pts):
                continue
                
            s = calculate_score_fast(new_pts)
            
            # Simulated annealing acceptance
            if s < current_val:
                current_val = s
                current_pts = new_pts
                best_val = s
                best_points = current_pts
            elif s == current_val:
                if random.random() < 0.1:
                    current_pts = new_pts
                    best_points = current_pts
            else:
                # Accept worse solutions with decreasing probability
                temp = max(0.1, 1.0 - iteration / max_iterations)
                if random.random() < math.exp(-(s - current_val) / temp):
                    current_pts = new_pts
                    best_points = current_pts

    # Fallback
    if best_points is None:
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
