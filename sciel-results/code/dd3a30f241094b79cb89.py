import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)
    
    def has_4_collinear(points):
        n = len(points)
        seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            count += 1
                if count >= 4:
                    return True
        return False
    
    def max_no_3_collinear(points):
        n = len(points)
        # Precompute for each pair (i,j) the bitmask of k collinear with i,j
        coll_mask = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                mask = 0
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            mask |= (1 << k)
                coll_mask[i][j] = mask
                coll_mask[j][i] = mask
        
        # Also precompute: for each point p, and each other point a, 
        # the mask of points q such that p, a, q are collinear
        # This is already coll_mask[p][a]
        
        best = [0]
        
        # Backtracking with bitmask
        # A: current selected set (bitmask)
        # C: candidates (bitmask)
        # For efficiency, track conflicts incrementally
        
        # Better approach: use recursion with careful state
        # State: A (bitmask of selected), C (bitmask of available)
        # When adding p to A, new C = C & ~union of coll_mask[p][a] for all a in A
        
        def backtrack(A, C, depth):
            if depth + bin(C).count('1') <= best[0]:
                return
            if C == 0:
                if depth > best[0]:
                    best[0] = depth
                return
            
            # Pick a candidate - heuristic: pick point with fewest remaining options
            # or just first bit
            # For speed, pick the lowest set bit
            p_bit = C & -C
            p = p_bit.bit_length() - 1
            
            # Option 1: exclude p
            backtrack(A, C ^ p_bit, depth)
            
            # Option 2: include p
            # Check if p is compatible with A
            # p is compatible if for no a1, a2 in A are p, a1, a2 collinear
            # i.e., for each a in A, p is not collinear with a and any other a' in A
            # Actually: adding p to A creates a triple if there exist a1, a2 in A with p,a1,a2 collinear
            # This means: for some a in A, a is in coll_mask[p][a'] for some a' in A... 
            # Simpler: for each pair (a1, a2) in A, check if p collinear with them
            # But that's O(|A|^2). 
            # Alternative: precompute for each p, the "forbidden pairs" in A
            # Actually, we can check: is there any a in A such that coll_mask[p][a] & A has any bit set?
            # No - coll_mask[p][a] contains points q collinear with p and a. 
            # We need: does there exist a1, a2 in A such that p, a1, a2 collinear?
            # This is equivalent to: for some a1 in A, a2 in A, a2 != a1, and a2 in coll_mask[p][a1]
            # i.e., (coll_mask[p][a1] & A) has a bit other than possibly a1 itself
            # But a1 is not in coll_mask[p][a1] by construction (k != i and k != j)
            # So: for some a1 in A, (coll_mask[p][a1] & A) != 0
            
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
                # New C: remove p and all q that would form a triple with p and some a in A
                new_C = C ^ p_bit
                temp_A = A
                while temp_A:
                    a_bit = temp_A & -temp_A
                    a = a_bit.bit_length() - 1
                    new_C &= ~coll_mask[p][a]
                    temp_A ^= a_bit
                
                backtrack(A | p_bit, new_C, depth + 1)
        
        backtrack(0, (1 << n) - 1, 0)
        return best[0]
    
    # Try a structured construction first
    # Use points that create many collinear triples
    # Strategy: take a 5x4 grid and perturb to eliminate 4-collinearities
    # while preserving many 3-collinearities
    
    best_score = 20
    best_config = None
    
    # Method 1: Structured grid with perturbations
    base_grid = []
    for i in range(5):
        for j in range(4):
            base_grid.append([i, j])
    
    # Try various perturbations
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        if attempts <= 500:
            # Small perturbations of grid
            pts = []
            for i, j in [(i, j) for i in range(5) for j in range(4)]:
                di = random.choice([-1, 0, 0, 0, 1])
                dj = random.choice([-1, 0, 0, 0, 1])
                pts.append([i + di, j + dj])
        else:
            # Random points in small region
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
    
    # Try a specific known-good construction
    # Points on a parabola with some moved to create collinearities
    if best_score > 10:
        # Try: take 20 points, place them so that ~10 triples are collinear
        # Use a 4x5 grid and swap some points
        specific_configs = []
        
        # Config 1: Perturbed grid
        pts1 = [[0,0],[1,0],[2,0],[3,0],[4,0],
                [0,1],[1,1],[2,1],[3,1],[4,1],
                [0,2],[1,2],[2,2],[3,2],[4,2],
                [0,3],[1,3],[2,3],[3,3],[4,3]]
        # Perturb to break 4-collinear in rows/cols
        # Move (4,0) to (5,1), (4,1) to (5,2), (4,2) to (5,3), (4,3) to (5,4)
        # But check for new 4-collinear
        pts1[4] = [5, 1]
        pts1[9] = [5, 2]  
        pts1[14] = [5, 3]
        pts1[19] = [5, 4]
        if len(set(map(tuple, pts1))) == 20 and not has_4_collinear(pts1):
            score1 = max_no_3_collinear(pts1)
            if score1 < best_score:
                best_score = score1
                best_config = pts1
        
        # Config 2: Different perturbation
        pts2 = [[0,0],[1,0],[2,0],[3,0],[4,0],
                [0,1],[1,1],[2,1],[3,1],[4,1],
                [0,2],[1,2],[2,2],[3,2],[4,2],
                [0,3],[1,3],[2,3],[3,3],[4,3]]
        # Move some corner points
        pts2[0] = [-1, 0]
        pts2[4] = [5, 0]
        pts2[10] = [-1, 2]
        pts2[14] = [5, 2]
        pts2[15] = [0, 4]
        pts2[19] = [4, 4]
        if len(set(map(tuple, pts2))) == 20 and not has_4_collinear(pts2):
            score2 = max_no_3_collinear(pts2)
            if score2 < best_score:
                best_score = score2
                best_config = pts2
    
    if best_config is None:
        # Fallback: parabola (score 20, but valid)
        return [[x, x * x] for x in range(-10, 10)]
    
    return best_config
