import itertools
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    # --- Helper: check if 4 points are collinear ---
    def has_4_collinear(pts):
        # pts is a list of (x, y)
        n = len(pts)
        if n < 4:
            return False
        # Use set of slopes from each point to others
        # For each point, check if any other point shares the same slope with 2 others
        # Actually, simpler: for each pair, define line, count points on it.
        # But O(n^3) is fine for n=20.
        # Let's use the line equation method.
        seen_lines = set()
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                A = y1 - y2
                B = x2 - x1
                C = x1 * y2 - x2 * y1
                # Normalize
                g = math.gcd(A, math.gcd(B, C))
                if g == 0:
                    continue
                A //= g
                B //= g
                C //= g
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                key = (A, B, C)
                if key in seen_lines:
                    continue
                seen_lines.add(key)
                # Count points on this line
                count = 0
                for k in range(n):
                    x3, y3 = pts[k]
                    if A * x3 + B * y3 + C == 0:
                        count += 1
                if count >= 4:
                    return True
        return False

    # --- Helper: Calculate score (max subset with no 3 collinear) ---
    def calculate_score(pts):
        n = len(pts)
        # Precompute collinear triples
        # For each pair (i, j), find all k such that i, j, k are collinear.
        # Then, for the branch and bound, we need to know if adding a point creates a triple.
        # Let's precompute for each point i, a bitmask of points that are collinear with i? No, that's not enough.
        # We need to know if {i, j, k} is a triple.
        # Let's build a list of triples.
        triples = []
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        triples.append((i, j, k))
        
        # For each point, store the list of triples it belongs to.
        # Also, for faster checking, maybe store for each pair (i, j), the list of k's?
        # When adding point p to a set S, we must ensure no triple (p, a, b) with a, b in S exists.
        # So for each p, we can precompute a list of pairs (a, b) such that (p, a, b) is a triple.
        # Let's call this conflicts[p] = list of (a, b)
        conflicts = [[] for _ in range(n)]
        for (i, j, k) in triples:
            # i, j, k are collinear
            conflicts[i].append((j, k))
            conflicts[j].append((i, k))
            conflicts[k].append((i, j))
            
        best_score = 0
        # Bitmask for current set
        # Branch and bound
        def bbb(idx, current_mask, current_size):
            nonlocal best_score
            if elapsed() > deadline - 5:
                return
            if current_size > best_score:
                best_score = current_size
            # Pruning
            if current_size + (n - idx) <= best_score:
                return
            if idx == n:
                return
            
            # Option 1: Exclude idx
            bbb(idx + 1, current_mask, current_size)
            
            # Option 2: Include idx
            # Check if adding idx creates a collinear triple with any two points in current_mask
            # We check conflicts[idx]. For each (a, b) in conflicts[idx], if a and b are in current_mask, invalid.
            valid = True
            for a, b in conflicts[idx]:
                if (current_mask & (1 << a)) and (current_mask & (1 << b)):
                    valid = False
                    break
            if valid:
                bbb(idx + 1, current_mask | (1 << idx), current_size + 1)
                
        bbb(0, 0, 0)
        return best_score

    # --- Search Strategy ---
    # 1. Start with a known good configuration or a random one.
    # 2. Use simulated annealing or random restarts with local search.
    # Objective: Minimize score, subject to no 4 collinear.
    # If 4 collinear, penalize heavily.
    
    def objective(pts):
        # Returns (penalty, score). We want to minimize.
        # If 4 collinear, return (1, 0) or something high.
        # Actually, we can just return a tuple that sorts correctly.
        # Let's return a single float: score + 100 * (1 if 4_collinear else 0)
        if has_4_collinear(pts):
            return 100.0 + calculate_score(pts)
        else:
            return calculate_score(pts)

    # Initial configuration: Parabola
    # Points on y = x^2 have no 3 collinear?
    # Wait, 3 points on a parabola y=x^2:
    # (x1, x1^2), (x2, x2^2), (x3, x3^2)
    # Slope 1-2: x1+x2. Slope 1-3: x1+x3.
    # Collinear if x1+x2 = x1+x3 => x2=x3.
    # So no 3 distinct points on y=x^2 are collinear.
    # Thus score is 20.
    # But we need to minimize score. So we want to INTRODUCE collinearities.
    # The current program tries to find a 5x5 grid subset with no 4 collinear, which might have score < 20.
    # The inspiration says "Random search with local improvement may find configurations with more collinear triples, reducing the max cap-free subset from 20 to ~12-14".
    
    # Let's try to find a configuration with many collinear triples.
    # A good structure is a grid. A 5x5 grid has many collinearities.
    # But we must avoid 4 collinear.
    # The current program does exactly this: finds a 5x5 grid subset with no 4 collinear.
    
    # Let's implement a random search that starts from a perturbed grid or random points.
    # We will try to find a set of 20 points in [-100, 100] x [-100, 100] with no 4 collinear, minimizing score.
    
    best_points = None
    best_val = float('inf')
    
    # Try the 5x5 grid approach first as a baseline (fast)
    # Re-implement the grid logic from the seed program but optimized?
    # The seed program's grid logic is O(25^2 * 25) for lines, which is fast.
    # Let's just use the seed program's logic to get a good starting point.
    
    points_5x5 = [(i, j) for i in range(5) for j in range(5)]
    lines = []
    seen_lines = set()
    for p1 in range(25):
        for p2 in range(p1 + 1, 25):
            x1, y1 = points_5x5[p1]
            x2, y2 = points_5x5[p2]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = math.gcd(A, math.gcd(B, C))
            if g == 0: continue
            A //= g; B //= g; C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A; B = -B; C = -C
            key = (A, B, C)
            if key in seen_lines: continue
            seen_lines.add(key)
            line_pts = []
            for p3 in range(25):
                x3, y3 = points_5x5[p3]
                if A * x3 + B * y3 + C == 0:
                    line_pts.append(p3)
            if len(line_pts) >= 4:
                lines.append(frozenset(line_pts))
                
    # Find hitting set of size 5
    valid_removals = []
    def backtrack(idx, current_removal, remaining_lines):
        if elapsed() > deadline - 20: return
        if not remaining_lines:
            valid_removals.append(set(current_removal))
            return
        if len(current_removal) == 5: return
        line = remaining_lines[0]
        rest_lines = remaining_lines[1:]
        candidates = [p for p in line if p not in current_removal]
        if not candidates: return
        for p in candidates:
            new_removal = list(current_removal) + [p]
            new_remaining = [l for l in rest_lines if p not in l]
            backtrack(idx + 1, new_removal, new_remaining)
            
    backtrack(0, [], lines)
    
    if valid_removals:
        # Evaluate the best few
        for rem in valid_removals[:5]:
            if elapsed() > deadline - 10: break
            rem_set = set(rem)
            pts = [points_5x5[i] for i in range(25) if i not in rem_set]
            if not has_4_collinear(pts):
                s = calculate_score(pts)
                if s < best_val:
                    best_val = s
                    best_points = pts
    
    # Now, try to improve via local search / random restarts
    # We will generate random points and try to "fix" them or improve them.
    # Or start from best_points and perturb.
    
    if best_points is None:
        # Fallback: random
        best_points = [(random.randint(-100, 100), random.randint(-100, 100)) for _ in range(20)]
        # Ensure distinct
        while len(set(best_points)) < 20:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p not in best_points:
                best_points.append(p)
                best_points.pop(0)
        best_val = objective(best_points)
        
    # Local Search: Swap points
    # Try to swap a point with a random new point to reduce objective.
    # If current config has 4 collinear, we must fix it.
    
    current_pts = [tuple(p) for p in best_points]
    current_val = best_val
    
    # Simulated Annealing or Hill Climbing
    # Let's do Hill Climbing with random restarts if stuck.
    # Since score is integer, we can try to find any valid config with lower score.
    
    # Strategy:
    # 1. If current has 4 collinear, try to move points to break them.
    # 2. If current is valid, try to move points to increase collinearities (reduce score).
    
    # To increase collinearities, we can try to align points.
    # This is hard to do randomly.
    # Instead, let's try to search for configurations in small grids.
    # e.g., 6x6 grid, 7x7 grid, etc.
    
    # Let's iterate over grid sizes
    for grid_size in range(4, 11):
        if elapsed() > deadline - 10: break
        
        # Generate grid points
        # Scale to fit in [-100, 100]
        # Step size = 200 / (grid_size - 1)
        step = 200.0 / (grid_size - 1)
        grid_pts = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = int(round(-100 + i * step))
                y = int(round(-100 + j * step))
                grid_pts.append((x, y))
                
        # If grid_size^2 < 20, we can't pick 20 distinct points.
        if len(grid_pts) < 20:
            continue
            
        # We need to pick 20 points from grid_pts such that no 4 are collinear.
        # This is a constraint satisfaction problem.
        # Let's use a heuristic: start with all points, remove points that cause 4 collinearities.
        # Or, pick 20 points randomly and check.
        
        # Random sampling from grid
        for _ in range(100):
            if elapsed() > deadline - 5: break
            if len(grid_pts) == 20:
                cand = grid_pts[:]
            else:
                cand = random.sample(grid_pts, 20)
                
            if has_4_collinear(cand):
                continue
                
            s = calculate_score(cand)
            if s < best_val:
                best_val = s
                best_points = cand
                current_pts = cand
                current_val = s
                
    # Finally, try some random points in the full range
    for _ in range(500):
        if elapsed() > deadline - 2: break
        cand = []
        while len(cand) < 20:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p not in cand:
                cand.append(p)
        
        if has_4_collinear(cand):
            continue
            
        s = calculate_score(cand)
        if s < best_val:
            best_val = s
            best_points = cand
            
    return [[x, y] for x, y in best_points]

# Test run
if __name__ == "__main__":
    pts = construct()
    print(pts)
    print("Score:", calculate_score(pts))
    print("Has 4 collinear:", has_4_collinear(pts))
