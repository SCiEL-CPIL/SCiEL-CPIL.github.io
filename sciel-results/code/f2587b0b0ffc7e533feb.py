import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_score = 20
    best_config = None
    
    # Helper: Check if 3 points are collinear
    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

    # Helper: Check if a set of points has no 4 collinear
    def has_4_collinear(points):
        n = len(points)
        # For each pair, count collinear points
        for i in range(n):
            for j in range(i + 1, n):
                count = 0
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            count += 1
                if count >= 2: # Total 4 points
                    return True
        return False

    # Helper: Calculate max subset size with no 3 collinear
    def max_no_3_collinear(points):
        n = len(points)
        # Precompute collinear triples as bitmasks or lists
        # coll[i][j] = bitmask of points k such that i, j, k are collinear
        coll = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                mask = 0
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            mask |= (1 << k)
                coll[i][j] = mask
                coll[j][i] = mask

        best = 0
        
        # Backtracking
        # A: current set of indices
        # C: bitmask of available candidates
        def backtrack(A_mask, C_mask, depth):
            nonlocal best
            # Pruning
            if depth + bin(C_mask).count('1') <= best:
                return
            
            if C_mask == 0:
                if depth > best:
                    best = depth
                return
            
            # Pick a point from C
            # Heuristic: Pick point with most constraints? Or just first.
            # Let's pick the first set bit in C_mask
            p = (C_mask & -C_mask).bit_length() - 1
            
            # Option 1: Exclude p
            backtrack(A_mask, C_mask ^ (1 << p), depth)
            
            # Option 2: Include p
            # Check if p conflicts with A
            valid = True
            # If we add p, we need to ensure no two points in A form a line with p
            # i.e., for any a1, a2 in A, are_collinear(p, a1, a2) must be False
            # We can check this by iterating through bits in A_mask
            a_indices = []
            temp_A = A_mask
            while temp_A:
                a_idx = (temp_A & -temp_A).bit_length() - 1
                a_indices.append(a_idx)
                temp_A ^= (1 << a_idx)
                
            for i in range(len(a_indices)):
                for j in range(i + 1, len(a_indices)):
                    a1 = a_indices[i]
                    a2 = a_indices[j]
                    if are_collinear(points[p], points[a1], points[a2]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # New C_mask: remove any q in C such that there exists a in A with p, q, a collinear
                new_C_mask = C_mask ^ (1 << p)
                for a in a_indices:
                    # If p, a, q are collinear, q cannot be in the set
                    # coll[p][a] gives mask of such q
                    new_C_mask &= ~coll[p][a]
                
                backtrack(A_mask | (1 << p), new_C_mask, depth + 1)

        backtrack(0, (1 << n) - 1, 0)
        return best

    # Generation Strategy
    # 1. Try some structured grids with perturbations
    # 2. Random search
    
    def generate_candidate():
        # Try to create points with many collinear triples
        # e.g., take a grid and shuffle
        pts = []
        # Use a 5x4 grid base
        base = []
        for i in range(5):
            for j in range(4):
                base.append([i, j])
        
        # Perturb to break 4-collinearities
        # Randomly swap or shift
        random.shuffle(base)
        
        # Assign random coordinates within [-100, 100] but keep them somewhat clustered
        # to increase chance of collinearity?
        # Actually, random integers in [-100, 100] are sparse, unlikely to have 3 collinear.
        # We WANT 3 collinear.
        # So we should use a denser region or specific structures.
        
        # Let's try points on a few lines.
        # Line 1: 10 points
        # Line 2: 10 points
        # But no 4 collinear.
        # So max 3 per line.
        # We need 20 points. 20/3 = 6.6. So at least 7 lines.
        
        # Construct:
        # Pick 7 lines. Distribute 20 points among them (3,3,3,3,3,3,2).
        # Ensure no 4 on any line.
        # Ensure points are distinct.
        
        lines = []
        # Define 7 lines with integer slopes/intercepts
        # y = m*x + c
        candidates = [
            (0, 0), # y=0
            (0, 1), # y=1
            (1, 0), # y=x
            (-1, 0), # y=-x
            (1, 1), # y=x+1
            (-1, 1), # y=-x+1
            (2, 0)  # y=2x
        ]
        
        pts = []
        # Assign 3 points to first 6 lines, 2 to last
        counts = [3, 3, 3, 3, 3, 3, 2]
        
        for (m, c), count in zip(candidates, counts):
            # Pick 'count' distinct x values
            # Ensure they don't overlap with existing points
            # And ensure the line doesn't have 4 points total (from other lines? No, we define the set)
            # But we must ensure that the union of points doesn't create a 4-collinear set on ANY line, not just these.
            # This is tricky.
            
            # For now, just pick random x values in [-10, 10]
            xs = random.sample(range(-10, 11), count)
            for x in xs:
                y = m * x + c
                pts.append([x, y])
        
        # Check for duplicates
        if len(set(map(tuple, pts))) < 20:
            return None
            
        # Check no 4 collinear
        if has_4_collinear(pts):
            return None
            
        return pts

    # Run search
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        pts = generate_candidate()
        if pts is None:
            continue
            
        # Also try random perturbation of a grid
        if iterations % 10 == 0:
            # Random grid perturbation
            pts = []
            for i in range(5):
                for j in range(4):
                    # Perturb i, j
                    di = random.randint(-2, 2)
                    dj = random.randint(-2, 2)
                    pts.append([i + di, j + dj])
            
            # Check distinct
            if len(set(map(tuple, pts))) < 20:
                continue
            # Check no 4 collinear
            if has_4_collinear(pts):
                continue

        score = max_no_3_collinear(pts)
        if score < best_score:
            best_score = score
            best_config = pts
            # Print for debugging? No, just track.
            # If we find a very good score, we can stop early?
            # Theoretical min?
            # If we find 8, that's very good.
            
    if best_config is None:
        # Fallback to parabola
        return [[x, x * x] for x in range(-10, 10)]
        
    return best_config
