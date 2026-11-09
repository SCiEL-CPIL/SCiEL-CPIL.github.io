"""New program: Optimized B&B for score + Simulated Annealing for search."""
import random
import time
import math
from collections import defaultdict

def _precompute_triples(points):
    """Precompute all collinear triples and conflict masks."""
    n = len(points)
    # collinear_triples[i] is a set of (j, k) such that i, j, k are collinear
    # Actually, for the B&B, we need:
    # For a point p, and a set of selected points S, 
    # p conflicts with S if there exists q, r in S such that p, q, r are collinear.
    # This is equivalent to: p conflicts with S if there exists q in S such that 
    # the line pq contains another point r in S.
    
    # Let's precompute for each pair (i, j), the set of k such that i, j, k are collinear.
    # line_points[i][j] = set of k
    line_points = [[set() for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = points[i]
            xj, yj = points[j]
            # Check all k
            for k in range(n):
                if k == i or k == j:
                    continue
                xk, yk = points[k]
                if (xj - xi) * (yk - yi) == (yj - yi) * (xk - xi):
                    line_points[i][j].add(k)
                    line_points[j][i].add(k)
    
    return line_points

def _largest_no_three_collinear(points):
    """Find the largest subset of points with no three collinear using optimized B&B with bitsets."""
    n = len(points)
    if n == 0:
        return 0
        
    line_points = _precompute_triples(points)
    
    # We want to find the max size of a set S such that no 3 points in S are collinear.
    # This means for any p, q, r in S, they are not collinear.
    
    best = [0]
    
    # Precompute conflict masks?
    # If we select p and q, we cannot select any r in line_points[p][q].
    # So, if we have selected S, and we want to add p,
    # we must ensure that for all q in S, line_points[p][q] does not intersect S.
    # Actually, simpler:
    # When adding p to S, we remove from candidates any point r such that 
    # there exists q in S where r is in line_points[p][q].
    # This ensures that no future point r will form a triple with p and any q in S.
    # Also, we must ensure p itself doesn't form a triple with two points in S.
    # If p forms a triple with q, r in S, then r is in line_points[p][q].
    # So if we maintain the invariant that candidates are "safe" to add given S,
    # we just need to check if p is safe to add to S.
    
    # Let's use a recursive function.
    # selected: bitset
    # candidates: bitset
    # 
    # To check if p can be added to selected:
    # For all q in selected, if there exists r in selected such that p, q, r collinear -> invalid.
    # This check is O(|S|^2) if done naively.
    
    # Optimization:
    # When we add p to selected, we update candidates.
    # New candidates = (old candidates \ {p}) \ { r | exists q in selected, r in line_points[p][q] }
    # This guarantees that any point in new candidates cannot form a triple with p and any q in selected.
    # Does it guarantee that the new candidates are mutually compatible?
    # Yes, because they were compatible before, and we only removed points.
    # But wait, if we add p, do we need to check if p forms a triple with two existing points?
    # If p forms a triple with q, r in selected, then r is in line_points[p][q].
    # Since r is in selected, and selected is a subset of (old selected U {p})...
    # Actually, the update rule `remove r if exists q in selected, r in line_points[p][q]` 
    # removes r from candidates. It does NOT prevent p from being added if p, q, r are already in selected.
    # So we MUST check if p is compatible with selected.
    
    # How to check compatibility of p with selected?
    # For each q in selected, check if line_points[p][q] intersects selected.
    # If yes, invalid.
    
    # To speed this up, we can precompute for each p, a list of (q, mask) where mask is the bitmask of points collinear with p and q.
    # Then check: for all q in selected, (mask[p][q] & selected) == 0.
    
    # Let's build these masks.
    # mask[p][q] is a bitmask of k such that p, q, k are collinear.
    masks = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            m = 0
            for k in line_points[i][j]:
                m |= (1 << k)
            masks[i][j] = m
    
    def backtrack(selected, candidates):
        # Upper bound
        if selected.bit_count() + candidates.bit_count() <= best[0]:
            return
            
        if candidates == 0:
            if selected.bit_count() > best[0]:
                best[0] = selected.bit_count()
            return
        
        # Pick a point to branch on.
        # Heuristic: Pick the point in candidates that is most "constrained" or has highest index?
        # Let's pick the lowest set bit for simplicity, or random?
        # Randomization can help escape local optima in B&B? No, B&B is exact.
        # We want to prune fast.
        # Points that are part of many lines should be decided early?
        
        # Let's just pick the first available point.
        # To improve, we could sort candidates by degree.
        # For now, lowest bit.
        lsb = candidates & -candidates
        p_idx = lsb.bit_length() - 1
        p_bit = 1 << p_idx
        
        # Branch 1: Exclude p
        backtrack(selected, candidates ^ p_bit)
        
        # Branch 2: Include p
        # Check if p is compatible with selected
        compatible = True
        # Iterate through bits in selected
        s = selected
        while s:
            q_lsb = s & -s
            q_idx = q_lsb.bit_length() - 1
            # Check if any point in selected is collinear with p and q
            if masks[p_idx][q_idx] & selected:
                compatible = False
                break
            s ^= q_lsb
            
        if compatible:
            # New selected
            new_selected = selected | p_bit
            
            # Update candidates
            # Remove p
            new_candidates = candidates ^ p_bit
            
            # Remove any r that forms a triple with p and any q in new_selected
            # We only need to check q in new_selected?
            # If r forms a triple with p and q, and q is in new_selected, then r is invalid.
            # We iterate q in new_selected.
            # But note: if r is in selected, it's already handled? No, selected is fixed.
            # We only care about future points (in candidates).
            
            # Optimization:
            # For each q in new_selected, the points that conflict with p and q are masks[p_idx][q_idx].
            # We want to remove (new_candidates & masks[p_idx][q_idx]) for all q in new_selected.
            # Union of these masks.
            
            conflict_mask = 0
            s = new_selected
            while s:
                q_lsb = s & -s
                q_idx = q_lsb.bit_length() - 1
                conflict_mask |= masks[p_idx][q_idx]
                s ^= q_lsb
            
            new_candidates &= ~conflict_mask
            
            backtrack(new_selected, new_candidates)
            
    backtrack(0, (1 << n) - 1)
    return best[0]

def _no_four_collinear(points):
    """Check that no four points are collinear."""
    n = len(points)
    # Count points on each line determined by pairs
    # Use normalized line representation
    def line_rep(i, j):
        x1, y1 = points[i]
        x2, y2 = points[j]
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        g = math.gcd(math.gcd(abs(A), abs(B)), abs(C)) if (A != 0 or B != 0) else 1
        if g == 0: g = 1
        A //= g; B //= g; C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)
    
    line_to_points = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            rep = line_rep(i, j)
            line_to_points[rep].add(i)
            line_to_points[rep].add(j)
            
    for rep, pts in line_to_points.items():
        if len(pts) >= 4:
            return False
    return True

def construct():
    """Construct 20 distinct integer points with |x|,|y| <= 100, no four collinear,
    minimizing the largest no-three-collinear subset."""
    
    random.seed(42)
    start_time = time.time()
    max_time = 115 # Leave 5s buffer
    
    best_score = 21
    best_points = None
    
    # Initial configuration: Random points, ensuring no 4 collinear
    # We can generate random points and check.
    # Or use a grid subset.
    
    def generate_random_config():
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) in used:
                continue
            pts.append([x, y])
            used.add((x, y))
        
        # Ensure no 4 collinear
        # If it fails, retry
        attempts = 0
        while not _no_four_collinear(pts) and attempts < 100:
            # Perturb one point
            i = random.randint(0, 19)
            pts[i] = [random.randint(-100, 100), random.randint(-100, 100)]
            # Check distinctness
            if len(set(map(tuple, pts))) < 20:
                continue
            attempts += 1
            
        if not _no_four_collinear(pts):
            return None
        return pts
    
    # Try a few initial random configs
    for _ in range(10):
        if time.time() - start_time > max_time:
            break
        cfg = generate_random_config()
        if cfg:
            score = _largest_no_three_collinear(cfg)
            if score < best_score:
                best_score = score
                best_points = [p[:] for p in cfg]
                
    if best_points is None:
        # Fallback: Parabola
        best_points = [[x, x*x] for x in range(-10, 10)]
        best_score = _largest_no_three_collinear(best_points)
        
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.99
    
    while time.time() - start_time < max_time:
        if T < T_min:
            break
            
        # Pick a random point to move
        i = random.randint(0, 19)
        old_x, old_y = current_points[i]
        
        # Move to a random nearby point
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        if dx == 0 and dy == 0:
            continue
            
        new_x, new_y = old_x + dx, old_y + dy
        if abs(new_x) > 100 or abs(new_y) > 100:
            continue
            
        new_points = [p[:] for p in current_points]
        new_points[i] = [new_x, new_y]
        
        # Check validity
        if len(set(map(tuple, new_points))) != 20:
            continue
        if not _no_four_collinear(new_points):
            continue
            
        new_score = _largest_no_three_collinear(new_points)
        
        # Acceptance criterion
        if new_score < current_score:
            current_points = new_points
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            # Accept with probability exp(-delta / T)
            delta = new_score - current_score
            if delta > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_points = new_points
                    current_score = new_score
                    
        T *= cooling
        
    return best_points

# EVOLVE-BLOCK-START
def construct():
    """Construct 20 distinct integer points with |x|,|y| <= 100, no four collinear,
    minimizing the largest no-three-collinear subset."""
    
    random.seed(42)
    
    best_score = 21
    best_points = None
    
    # Start with a known valid configuration and try to improve via local search
    # Initial: points on parabola y = x^2
    initial = [[x, x * x] for x in range(-10, 10)]
    
    if not _no_four_collinear(initial):
        return initial
    
    current_points = [p[:] for p in initial]
    current_score = _largest_no_three_collinear(current_points)
    
    best_score = current_score
    best_points = [p[:] for p in current_points]
    
    # Local search: try to move individual points to reduce the score
    start_time = time.time()
    max_time = 100  # seconds, leave buffer for the evaluation
    
    iteration = 0
    while time.time() - start_time < max_time:
        iteration += 1
        improved = False
        
        # Try moving each point to a nearby location
        for i in range(len(current_points)):
            if time.time() - start_time > max_time:
                break
            
            x, y = current_points[i]
            # Try nearby points
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                    
                    new_points = [p[:] for p in current_points]
                    new_points[i] = [nx, ny]
                    
                    # Check distinctness
                    if len(set(map(tuple, new_points))) != 20:
                        continue
                    
                    # Check no four collinear
                    if not _no_four_collinear(new_points):
                        continue
                    
                    new_score = _largest_no_three_collinear(new_points)
                    
                    if new_score < current_score:
                        current_points = new_points
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_points = [p[:] for p in current_points]
                        break
                else:
                    continue
                break
        
        if not improved:
            # Try random perturbations
            for _ in range(100):
                if time.time() - start_time > max_time:
                    break
                
                i = random.randint(0, 19)
                x, y = current_points[i]
                nx = random.randint(max(-100, x - 10), min(100, x + 10))
                ny = random.randint(max(-100, y - 10), min(100, y + 10))
                
                new_points = [p[:] for p in current_points]
                new_points[i] = [nx, ny]
                
                if len(set(map(tuple, new_points))) != 20:
                    continue
                if not _no_four_collinear(new_points):
                    continue
                
                new_score = _largest_no_three_collinear(new_points)
                
                if new_score < current_score:
                    current_points = new_points
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_points = [p[:] for p in current_points]
                    break
    
    return best_points
# EVOLVE-BLOCK-END
