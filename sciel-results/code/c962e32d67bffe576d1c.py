import time
import random
import math
from collections import defaultdict

def _count_4_lines(points):
    """
    Counts the number of lines containing exactly 4 of the given points.
    Also returns True if any line contains 5 or more points (invalid).
    """
    n = len(points)
    if n < 4:
        return 0, False
    
    line_counts = defaultdict(int)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            if A == 0 and B == 0:
                continue
                
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g > 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0:
                A = -A
                B = -B
                C = -C
            elif A == 0:
                if B < 0:
                    B = -B
                    C = -C
                elif B == 0:
                    if C < 0:
                        C = -C
            
            line_counts[(A, B, C)] += 1

    count_4 = 0
    has_5 = False
    for count in line_counts.values():
        if count == 4:
            count_4 += 1
        elif count >= 5:
            has_5 = True
            
    return count_4, has_5

def _check_validity(points):
    """
    Fast check for no 5-collinear.
    Returns True if valid, False otherwise.
    """
    n = len(points)
    line_counts = defaultdict(int)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            if A == 0 and B == 0:
                continue
                
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g > 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0:
                A = -A
                B = -B
                C = -C
            elif A == 0:
                if B < 0:
                    B = -B
                    C = -C
                elif B == 0:
                    if C < 0:
                        C = -C
            
            line_counts[(A, B, C)] += 1
            if line_counts[(A, B, C)] >= 5:
                return False
    return True

def _construct_via_line_packing(start_time, deadline, max_grid_size=8, max_lines_to_consider=500):
    """
    Greedy line packing strategy.
    1. Generate all 4-point lines in a grid.
    2. Shuffle them.
    3. Greedily select lines that don't cause 5-collinear violations with already selected points.
    4. Collect points from selected lines.
    5. If we have <= 16 points, fill in random points to reach 16.
    6. If we have > 16 points, we can't use this selection directly, but we might prune.
       Actually, if we just add points, we might exceed 16. 
       Better: Select lines, collect unique points. If count <= 16, add randoms.
       If count > 16, this specific selection is too dense? No, we want 16 points.
       If we select too many lines, we get too many points. 
       We should stop adding lines once we have enough points to form 16?
       No, we want to MAXIMIZE 4-lines. 
       If we have a set of points P, we count 4-lines in P.
       
       Alternative:
       Generate candidate lines.
       Start with empty set of points.
       Iterate through lines (shuffled).
       If adding the 4 points of this line to current set:
           - No duplicates
           - Total points <= 16
           - No 5-collinear violation
       Then add them.
       
       This might stop too early. 
       Let's try: 
       1. Generate all 4-lines in a box.
       2. Shuffle.
       3. Greedily pick lines. 
       4. At the end, if we have < 16 points, add randoms.
       5. If we have > 16 points, we need to drop some? 
          Actually, if we just pick lines, the points accumulate. 
          If we pick 5 lines, we might have 20 points. 
          We need exactly 16.
          
       Revised Strategy:
       1. Generate all 4-lines.
       2. Shuffle.
       3. Greedily select a subset of lines such that the union of their points has size <= 16.
       4. Fill remaining slots with random points to reach 16.
       5. Count 4-lines in the final set.
       
       To get more 4-lines, we want to select lines that share points (overlap) to keep the point count low while increasing line count?
       Yes! Overlapping lines are good. 
       So we prefer lines that share 2 or 3 points with already selected lines.
       
       Let's implement:
       - current_points = set()
       - selected_lines = []
       - For each line in shuffled_lines:
           - if len(current_points) + 4 > 16: continue (can't add this line without exceeding 16)
             Wait, if it shares points, len(current_points) + (4 - shared) might be <= 16.
           - new_points = current_points | line_points
           - if len(new_points) > 16: continue
           - Check validity of new_points (no 5-collinear)
           - If valid, add to current_points, add line to selected_lines.
       
       This is a heuristic. It might get stuck.
       Let's add a random restart mechanism.
    """
    
    # Generate grid
    # Use a slightly larger grid to find more potential lines, but keep it small for speed
    # 8x8 = 64 points. C(64, 4) is ~ 635k. Too many to generate all lines explicitly and store?
    # Actually, we only need lines with exactly 4 points from the GRID? No, lines in the final set.
    # But the final set is a subset of the grid.
    # So any 4-line in the final set is a 4-line in the grid (or a superset of a grid line? No, if 4 points are collinear in grid, they are in final set).
    # So we can enumerate all 4-point collinear sets in the grid.
    
    grid_points = []
    for x in range(-max_grid_size, max_grid_size + 1):
        for y in range(-max_grid_size, max_grid_size + 1):
            grid_points.append((x, y))
            
    # Map point to index
    point_to_idx = {p: i for i, p in enumerate(grid_points)}
    
    # Find all 4-point lines in the grid
    # This is expensive: O(N^4). N=17x17=289. 289^4 is ~ 7e9. Too slow.
    
    # Alternative: 
    # Instead of enumerating all lines, use the local search but with a better initial state?
    # The prompt suggests "Maximum weight independent set on a conflict graph".
    # But implementing that from scratch in 150s is risky.
    
    # Let's stick to a smarter local search.
    # The previous local search was slow because it recalculated the score from scratch.
    # We can optimize the score calculation or use incremental updates.
    
    # Let's try a different initialization: 
    # Start with a known good configuration (e.g. 4x4 grid) and do a more aggressive 
    # simulated annealing with a better move set.
    
    # Move set: 
    # 1. Swap two points.
    # 2. Move one point to a random location.
    # 3. "Line completion": If there are 3 points on a line, move a 4th point to complete it?
    
    # Let's implement a faster local search with incremental updates if possible, 
    # or just a very fast full recount.
    
    # Full recount is O(N^2) = 120 ops. 
    # 100,000 iterations = 12M ops. Should be fast enough in Python? 
    # 12M ops in Python is ~ 1-2 seconds. 
    # So we can do ~100k-200k iterations in 150s.
    
    # The current program does ~50-200 iterations of "try_complete_grid" which is slow.
    # Let's do a direct simulated annealing with many iterations.
    
    best_points = None
    best_score = 0
    
    # Initial configurations
    inits = []
    
    # 1. 4x4 Grid
    grid_4x4 = [[x, y] for x in range(4) for y in range(4)]
    inits.append([p[:] for p in grid_4x4])
    
    # 2. 4x4 Grid shifted/rotated
    for _ in range(5):
        g = [p[:] for p in grid_4x4]
        # Random affine transform?
        # Just random small perturbations
        for i in range(16):
            if random.random() < 0.5:
                g[i][0] += random.randint(-1, 1)
                g[i][1] += random.randint(-1, 1)
        # Ensure distinct
        unique = []
        for p in g:
            if p not in unique:
                unique.append(p)
        if len(unique) == 16:
            inits.append(unique)
            
    # 3. Random points in 10x10 box
    for _ in range(10):
        pts = []
        while len(pts) < 16:
            p = [random.randint(-5, 5), random.randint(-5, 5)]
            if p not in pts:
                pts.append(p)
        inits.append(pts)
        
    # 4. "Parabola" like: y = x^2 mod something?
    # (0,0), (1,1), (2,4), (3,9), (4,16)... 
    # Let's just use random inits for now.
    
    for init in inits:
        if time.time() > deadline:
            break
            
        current_points = [p[:] for p in init]
        current_score, is_invalid = _count_4_lines(current_points)
        if is_invalid:
            continue
            
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
        # Simulated Annealing
        T = 10.0
        T_min = 0.1
        cooling = 0.9995 # Slow cooling
        
        # Run for a fixed number of steps or until time
        steps = 20000
        for i in range(steps):
            if time.time() > deadline:
                break
                
            # Choose a move
            move_type = random.random()
            
            if move_type < 0.7:
                # Move one point
                idx = random.randint(0, 15)
                ox, oy = current_points[idx]
                # Move to nearby
                nx = ox + random.randint(-3, 3)
                ny = oy + random.randint(-3, 3)
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                # Check distinct
                is_dup = False
                for k in range(16):
                    if k != idx and current_points[k] == [nx, ny]:
                        is_dup = True
                        break
                if is_dup:
                    continue
                    
                new_points = [p[:] for p in current_points]
                new_points[idx] = [nx, ny]
                
            elif move_type < 0.9:
                # Swap two points
                idx1, idx2 = random.sample(range(16), 2)
                new_points = [p[:] for p in current_points]
                new_points[idx1], new_points[idx2] = new_points[idx2][:], new_points[idx1][:]
            else:
                # Jump
                idx = random.randint(0, 15)
                nx = random.randint(-10, 10)
                ny = random.randint(-10, 10)
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                is_dup = False
                for k in range(16):
                    if k != idx and current_points[k] == [nx, ny]:
                        is_dup = True
                        break
                if is_dup:
                    continue
                new_points = [p[:] for p in current_points]
                new_points[idx] = [nx, ny]
                
            new_score, is_invalid = _count_4_lines(new_points)
            if is_invalid:
                # Reject invalid
                continue
                
            delta = new_score - current_score
            if delta > 0 or random.random() < math.exp(delta / T):
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
                    
            T *= cooling
            if T < T_min:
                T = T_min
                
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    random.seed(42)
    
    best_points_global, best_score_global = _construct_via_line_packing(start_time, deadline)
    
    # If we didn't find anything, fallback
    if best_points_global is None:
        best_points_global = [[x, y] for x in range(4) for y in range(4)]
        
    return best_points_global
