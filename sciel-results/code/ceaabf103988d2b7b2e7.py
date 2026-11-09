import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    # Fast check for 4 collinear
    def has_four_collinear(points):
        # Map slope/intercept to count of points
        # Since N=20, O(N^2) lines is fine.
        lines = defaultdict(list)
        for i in range(20):
            for j in range(i+1, 20):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                # Normalize slope
                g = __import__('math').gcd(dx, dy)
                if g != 0:
                    dx //= g
                    dy //= g
                # Handle vertical lines
                if dx == 0:
                    key = (0, x1)
                else:
                    # y = mx + c -> c = y - mx
                    # Use integer arithmetic: y1*dx - x1*dy is constant for the line?
                    # No, standard form: A x + B y + C = 0
                    # A = dy, B = -dx, C = -dy*x1 + dx*y1
                    A = dy
                    B = -dx
                    C = -A*x1 - B*y1
                    # Normalize A, B, C
                    g2 = __import__('math').gcd(A, B)
                    if g2 != 0:
                        A //= g2
                        B //= g2
                        C //= g2
                    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                        A = -A
                        B = -B
                        C = -C
                    key = (A, B, C)
                lines[key].append(i)
                lines[key].append(j)
        
        # Check if any line has >= 4 unique points
        for pts in lines.values():
            if len(set(pts)) >= 4:
                return True
        return False

    def get_triple_masks(points):
        masks = []
        for i in range(20):
            for j in range(i+1, 20):
                for k in range(j+1, 20):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1):
                        masks.append((1 << i) | (1 << j) | (1 << k))
        return masks

    def max_independent_set(triple_masks):
        if not triple_masks:
            return 20
        # B&B for 3-uniform hypergraph independent set
        # State: remaining vertices mask, current count
        # Bound: count + popcount(mask)
        # Pruning: If we have a triple fully in mask, we must exclude at least one.
        
        n = 20
        full_mask = (1 << n) - 1
        best = 0
        
        # Precompute popcount for all 2^20? No, too big. Use bin(x).count('1') or lookup table for 16-bit chunks.
        popcount_table = [bin(i).count('1') for i in range(1 << 16)]
        def popcount(x):
            return popcount_table[x & 0xFFFF] + popcount_table[(x >> 16) & 0xFFFF]

        # Sort triple masks to help pruning?
        # Triple masks are small.
        
        def dfs(mask, count):
            nonlocal best
            if count + popcount(mask) <= best:
                return
            if mask == 0:
                if count > best:
                    best = count
                return
            
            # Find a triple fully contained in mask
            found_triple = None
            for t in triple_masks:
                if (t & mask) == t:
                    found_triple = t
                    break
            
            if found_triple is None:
                # All remaining vertices are independent
                best = max(best, count + popcount(mask))
                return
            
            # Branch: Exclude one vertex from found_triple
            # To be efficient, pick the vertex in found_triple that appears in the most other triples within mask?
            # For now, just iterate the 3 bits.
            bits = []
            tmp = found_triple
            while tmp:
                lsb = tmp & -tmp
                bits.append(lsb)
                tmp ^= lsb
                
            # Heuristic: Try excluding the bit that removes the most "conflict" potential?
            # Or just try all 3.
            for bit in bits:
                new_mask = mask & ~bit
                dfs(new_mask, count)
                
        dfs(full_mask, 0)
        return best

    def score(points):
        masks = get_triple_masks(points)
        return max_independent_set(masks)

    best_points = None
    best_score = 20
    
    # Generate initial points on a parabola
    # y = x^2. Scale to fit [-100, 100].
    # x in [-10, 10] -> y in [0, 100].
    # We need 20 points.
    # Let's use x = -10, -9, ..., 9.
    # y = x*x.
    # Points: (-10, 100), (-9, 81), ..., (9, 81).
    # Check 4 collinear?
    # Parabola has no 3 collinear? No, 3 points on a parabola are not collinear.
    # So this initial config has 0 triples. Score 20.
    # We need to "break" the parabola to create triples.
    
    # Better initial: Grid with perturbation?
    # Or just start with random and use the "Snap to line" move.
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        # Random start
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append((x, y))
        
        if has_four_collinear(points):
            continue
            
        current_score = score(points)
        if current_score < best_score:
            best_score = current_score
            best_points = points[:]
            
        # Local Search with "Snap to line"
        temp = 50.0
        current_points = points[:]
        
        # Precompute lines for current points? 
        # We will update incrementally or recompute if needed.
        # Given N=20, recomputing lines is O(N^2) = 400 ops. Cheap.
        
        for step in range(10000):
            if time.time() > deadline:
                break
                
            # Choose a move type
            if random.random() < 0.7:
                # Snap to line
                # Pick two points A, B
                idx_a = random.randint(0, 19)
                idx_b = random.randint(0, 19)
                if idx_a == idx_b:
                    continue
                A = current_points[idx_a]
                B = current_points[idx_b]
                
                # Find integer points on line AB within bounds
                # Line equation: y = m x + c
                # Or parametric: P = A + t * (B - A)
                # We want integer solutions.
                # Vector v = B - A = (dx, dy)
                # Points are A + k * v / g where g = gcd(dx, dy)?
                # Actually, the integer points on the line segment or line are 
                # A + k * (dx/g, dy/g) for integer k.
                
                dx = B[0] - A[0]
                dy = B[1] - A[1]
                g = __import__('math').gcd(dx, dy)
                if g == 0: continue
                step_x = dx // g
                step_y = dy // g
                
                # Find range of k such that point is in [-100, 100]
                # x = A[0] + k * step_x
                # y = A[1] + k * step_y
                # -100 <= x <= 100
                # -100 <= y <= 100
                
                # Solve for k bounds
                # This is a bit tedious. Just generate a few candidates around A and B.
                candidates = set()
                for k in range(-5, 6):
                    nx = A[0] + k * step_x
                    ny = A[1] + k * step_y
                    if -100 <= nx <= 100 and -100 <= ny <= 100:
                        candidates.add((nx, ny))
                        
                if not candidates:
                    continue
                    
                # Pick a random point P to move
                idx_p = random.randint(0, 19)
                if idx_p == idx_a or idx_p == idx_b:
                    # Moving A or B would break the line definition, but maybe we want to move a 3rd point onto it.
                    # If we move A, we lose the line. 
                    # We want to create a triple. So we should move a point NOT in {A, B} to a position on line AB.
                    # If we move A, we might create a new line with B and some C.
                    # Let's enforce moving a point not A or B to form a triple with A and B.
                    continue
                    
                # Pick a random candidate position
                new_pos = random.choice(list(candidates))
                
                # Check if new_pos is already occupied
                if new_pos in [current_points[i] for i in range(20) if i != idx_p]:
                    continue
                    
                candidate_points = current_points[:]
                candidate_points[idx_p] = new_pos
                
                if has_four_collinear(candidate_points):
                    continue
                    
                new_score = score(candidate_points)
                delta = new_score - current_score
                
                if delta < 0 or random.random() < pow(2.718, -delta / temp):
                    current_points = candidate_points
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_points = current_points[:]
                        
            else:
                # Random move
                idx_p = random.randint(0, 19)
                nx = random.randint(-100, 100)
                ny = random.randint(-100, 100)
                if (nx, ny) == current_points[idx_p]:
                    continue
                if (nx, ny) in [current_points[i] for i in range(20) if i != idx_p]:
                    continue
                    
                candidate_points = current_points[:]
                candidate_points[idx_p] = (nx, ny)
                
                if has_four_collinear(candidate_points):
                    continue
                    
                new_score = score(candidate_points)
                delta = new_score - current_score
                
                if delta < 0 or random.random() < pow(2.718, -delta / temp):
                    current_points = candidate_points
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_points = current_points[:]
                        
            temp *= 0.999

    if best_points:
        return [list(p) for p in best_points]
    else:
        return [[x, x * x] for x in range(-10, 10)]
