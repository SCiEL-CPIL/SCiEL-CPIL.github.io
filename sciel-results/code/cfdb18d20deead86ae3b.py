import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)
        # Note: Fixed typo in thought process, actual code below is correct
    
    def are_collinear_correct(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

    def has_4_collinear(points):
        n = len(points)
        # Count points on each line defined by pairs
        # Since n=20, O(n^3) is 8000 ops, very fast.
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear_correct(points[i], points[j], points[k]):
                            count += 1
                            if count >= 4:
                                return True
        return False

    def max_no_3_collinear(points):
        n = len(points)
        # Precompute collinearity masks
        # coll_mask[i][j] is a bitmask of k such that i, j, k are collinear (k != i, j)
        coll_mask = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                mask = 0
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear_correct(points[i], points[j], points[k]):
                            mask |= (1 << k)
                coll_mask[i][j] = mask
                coll_mask[j][i] = mask
        
        best = [0]
        
        # Backtracking with bitmask
        # A: selected points (bitmask)
        # C: candidate points (bitmask)
        
        def backtrack(A, C, depth):
            if depth + C.bit_count() <= best[0]:
                return
            if C == 0:
                if depth > best[0]:
                    best[0] = depth
                return
            
            # Heuristic: Pick a point from C
            # We want to pick a point that will prune the search space effectively.
            # A point that conflicts with many points in C given A is good.
            # Or simply the first bit.
            # Let's try picking the point that maximizes the number of removals from C if added.
            # This requires calculating new_C for each candidate, which is expensive.
            # Let's stick to a simpler heuristic: pick the point with the highest index in C?
            # Or lowest?
            # Let's pick the lowest set bit for now, but maybe sort C by some static property?
            # Static property: number of collinear partners in the full set?
            
            p_bit = C & -C
            p = p_bit.bit_length() - 1
            
            # Option 1: Exclude p
            backtrack(A, C ^ p_bit, depth)
            
            # Option 2: Include p
            # Check compatibility with A
            # p is incompatible if there exists a1, a2 in A such that p, a1, a2 are collinear.
            # This is equivalent to: for some a in A, (coll_mask[p][a] & A) != 0.
            compatible = True
            temp_A = A
            while temp_A:
                a_bit = temp_A & -temp_A
                a = a_bit.bit_length() - 1
                if coll_mask[p][a] & A:
                    compatible = False
                    break
                temp_A ^= a_bit
            
            if compatible:
                new_A = A | p_bit
                new_C = C ^ p_bit
                # Remove from C any q such that p, q, a are collinear for some a in A.
                # q is removed if q is in coll_mask[p][a] for some a in A.
                remove_mask = 0
                temp_A = A
                while temp_A:
                    a_bit = temp_A & -temp_A
                    a = a_bit.bit_length() - 1
                    remove_mask |= coll_mask[p][a]
                    temp_A ^= a_bit
                
                new_C &= ~remove_mask
                
                backtrack(new_A, new_C, depth + 1)
        
        backtrack(0, (1 << n) - 1, 0)
        return best[0]

    best_score = 20
    best_config = None
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Generate candidates
        # Strategy 1: Perturbed 5x4 Grid
        # Strategy 2: Perturbed 4x5 Grid
        # Strategy 3: Random in small box
        
        if attempts % 3 == 1:
            # 5x4 Grid
            pts = []
            for i in range(5):
                for j in range(4):
                    di = random.choice([-1, 0, 0, 1])
                    dj = random.choice([-1, 0, 0, 1])
                    pts.append([i + di, j + dj])
        elif attempts % 3 == 2:
            # 4x5 Grid
            pts = []
            for i in range(4):
                for j in range(5):
                    di = random.choice([-1, 0, 0, 1])
                    dj = random.choice([-1, 0, 0, 1])
                    pts.append([i + di, j + dj])
        else:
            # Random in [-5, 5]
            pts = []
            for _ in range(20):
                x = random.randint(-5, 5)
                y = random.randint(-5, 5)
                pts.append([x, y])
        
        # Check distinct
        if len(set(map(tuple, pts))) < 20:
            continue
        
        # Check no 4 collinear
        if has_4_collinear(pts):
            continue
        
        # Compute score
        score = max_no_3_collinear(pts)
        if score < best_score:
            best_score = score
            best_config = pts
            if best_score <= 9:
                break
    
    if best_config is None:
        # Fallback: Parabola (score 20)
        return [[x, x * x] for x in range(-10, 10)]
    
    return best_config
