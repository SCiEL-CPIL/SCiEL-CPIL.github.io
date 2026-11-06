"""Combined program for 20-point no-4-collinear configuration."""
import time
import random

def construct():
    start_time = time.time()
    timeout = 135  # Leave 15s margin for safety before 150s kill

    def are_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1])

    def has_4_collinear(points):
        n = len(points)
        # For each pair, count how many other points are collinear with them
        for i in range(n):
            for j in range(i + 1, n):
                count = 0
                for k in range(n):
                    if k != i and k != j and are_collinear(points[i], points[j], points[k]):
                        count += 1
                if count >= 2:
                    return True
        return False

    def get_collinear_triples(points):
        n = len(points)
        triples = set()
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples.add((i, j, k))
        return triples

    def max_no_3_collinear_exact(points):
        """
        Exact branch and bound for max subset with no 3 collinear.
        Uses bitmasks and triple-branching.
        """
        n = len(points)
        if n <= 2:
            return n

        triples = get_collinear_triples(points)
        if not triples:
            return n

        # Map triples to bitmasks for fast intersection checks
        triple_masks = [(1 << u) | (1 << v) | (1 << w) for u, v, w in triples]
        
        best_size = 0
        
        # Precompute conflict pairs for each point to check validity of inclusion
        # conflict_pairs[i] is a list of (j, k) such that (i, j, k) is a triple
        conflict_pairs = [[] for _ in range(n)]
        for u, v, w in triples:
            conflict_pairs[u].append((v, w))
            conflict_pairs[v].append((u, w))
            conflict_pairs[w].append((u, v))

        # Heuristic: Order of branching. We branch on the "most constrained" point?
        # Or just iterate. With n=20, simple bit manipulation is fast.
        
        # State: available_mask, current_size
        # Optimization: If we include a point v, we must ensure no triple (v, u, w) is fully in the set.
        # This is handled by the recursion: if we pick v, we don't immediately remove u or w,
        # but we must ensure that in the future, we don't pick both u and w.
        # Wait, the standard "branch on triple" approach is often better for hypergraphs.
        # Approach: 
        # 1. Find a triple (u, v, w) in available_mask.
        # 2. Branch: Exclude u, Exclude v, Exclude w.
        # This guarantees that in any leaf, no triple is fully included.
        
        def solve(available_mask, current_size):
            nonlocal best_size
            
            # Pruning
            if current_size + bin(available_mask).count('1') <= best_size:
                return
            
            if available_mask == 0:
                if current_size > best_size:
                    best_size = current_size
                return
            
            # Find a triple contained in available_mask
            # To speed up, we can precompute for each mask? No, too large.
            # Just scan triples.
            found_triple = None
            for tm in triple_masks:
                if (available_mask & tm) == tm:
                    found_triple = tm
                    break
            
            if found_triple is None:
                # No triples left in available_mask, so all remaining points can be taken
                size = bin(available_mask).count('1')
                if current_size + size > best_size:
                    best_size = current_size + size
                return
            
            # Extract indices from found_triple
            # We need to exclude one of the three points in the triple
            # Let u, v, w be the points.
            # We can find them by checking bits.
            # Since n <= 20, we can iterate.
            bits = []
            for i in range(n):
                if found_triple & (1 << i):
                    bits.append(i)
            
            # Branch 1: Exclude bits[0]
            solve(available_mask ^ (1 << bits[0]), current_size)
            
            # Branch 2: Exclude bits[1]
            solve(available_mask ^ (1 << bits[1]), current_size)
            
            # Branch 3: Exclude bits[2]
            solve(available_mask ^ (1 << bits[2]), current_size)
            
        solve((1 << n) - 1, 0)
        return best_size

    def score_config(points):
        return max_no_3_collinear_exact(points)

    best_score = 20
    best_points = None

    # --- Configuration Generation Strategies ---

    def generate_grid_perturbed(seed_offset=0):
        """
        Generate a 5x4 grid and perturb points to break 4-collinear lines.
        Start with a regular grid, then nudge points slightly.
        """
        random.seed(1234 + seed_offset)
        pts = []
        used = set()
        
        # 5 columns, 4 rows
        # Base grid
        xs = [0, 20, 40, 60, 80]
        ys = [0, 30, 60, 90]
        
        for y in ys:
            for x in xs:
                # Add random perturbation
                # Must ensure no 4 collinear.
                # Small perturbations might keep collinearities if they are integer.
                # Let's try larger random jumps within a local box.
                nx = x + random.randint(-5, 5)
                ny = y + random.randint(-5, 5)
                
                # Ensure bounds
                if nx < -100 or nx > 100 or ny < -100 or ny > 100:
                    continue
                if (nx, ny) in used:
                    # Try a few times
                    found = False
                    for _ in range(10):
                        nx = x + random.randint(-10, 10)
                        ny = y + random.randint(-10, 10)
                        if -100 <= nx <= 100 and -100 <= ny <= 100 and (nx, ny) not in used:
                            found = True
                            break
                    if not found:
                        continue
                
                pts.append([nx, ny])
                used.add((nx, ny))
        
        # If we didn't get 20, fill with random
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                used.add((x, y))
                pts.append([x, y])
        
        return pts[:20]

    def generate_parabola_clusters():
        """
        Points on parabolas have no 3 collinear.
        We want to MINIMIZE score, so we want MAX collinearities.
        This strategy is bad for minimization. We want points ON lines.
        So we use lines with exactly 3 points.
        """
        random.seed(999)
        pts = []
        used = set()
        
        # Try to form 6 lines of 3 points + 2 singles? Or 5 lines of 4? No, max 3 per line.
        # 20 points.
        # Strategy: Pick 3 points on a line, add them. Repeat.
        # Ensure no 4th point falls on any existing line.
        
        lines = []
        
        for _ in range(8): # Try to create 8 lines of 3? 8*3=24, too many.
            # We need 20 points.
            # 6 lines of 3 = 18 points. 2 singles.
            # Or 5 lines of 4 is forbidden.
            # Let's aim for 6 lines of 3 points, sharing some points if possible?
            # If lines share points, we use fewer points.
            # We want to use exactly 20 points.
            
            pass # This generic construction is hard to guarantee validity easily.
            # Let's stick to grid perturbation and random search.
            break

    candidates = []
    
    # 1. Perturbed grids
    for i in range(5):
        pts = generate_grid_perturbed(i)
        if len(pts) == 20 and not has_4_collinear(pts):
            candidates.append(pts)
    
    # 2. Random points (dense)
    for i in range(20):
        random.seed(1000 + i)
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            if (x, y) not in used:
                used.add((x, y))
                pts.append([x, y])
        if not has_4_collinear(pts):
            candidates.append(pts)

    # 3. Specific structured attempt: 
    # Use a "star" or "wheel" like structure but with integer coords.
    # Center (0,0). 
    # Pairs of points symmetric?
    # If we use points (x, y) such that they lie on lines through origin?
    # Line y=x: (1,1), (2,2), (3,3) -> 3 pts.
    # Line y=-x: (1,-1), (2,-2), (3,-3) -> 3 pts.
    # Line y=2x: (1,2), (2,4), (3,6) -> 3 pts.
    # Line y=-2x: (1,-2), (2,-4), (3,-6) -> 3 pts.
    # Line x=0: (0,1), (0,2), (0,3) -> 3 pts.
    # Line x=1: (1,0), (1,1)... (1,1) is used. 
    # This requires careful bookkeeping.
    
    def generate_line_clusters():
        random.seed(42)
        pts = []
        used = set()
        
        # Define 6 lines with 3 points each
        # Line 1: y = 0. Points: (-10,0), (0,0), (10,0)
        # Line 2: y = 1. Points: (-9,1), (0,1), (9,1)
        # Line 3: y = 2. Points: (-8,2), (0,2), (8,2)
        # Line 4: y = 3. Points: (-7,3), (0,3), (7,3)
        # Line 5: y = 4. Points: (-6,4), (0,4), (6,4)
        # Line 6: y = 5. Points: (-5,5), (0,5), (5,5)
        # Total 18 points.
        # Check verticals:
        # x=0 has 6 points: (0,0)..(0,5). BAD (4 collinear).
        
        # We must stagger x-coordinates for x=0.
        # Let's use random x for each row, but ensure no 3 in a column or diagonal.
        
        pts = []
        used = set()
        
        for y in range(6):
            # Pick 3 distinct x's
            xs = random.sample(range(-20, 21), 3)
            # Sort them for consistency? Doesn't matter.
            for x in xs:
                if (x, y) not in used:
                    pts.append([x, y])
                    used.add((x, y))
        
        # Add 2 more points
        while len(pts) < 20:
            x = random.randint(-20, 20)
            y = random.randint(6, 10) # New rows
            if (x, y) not in used:
                pts.append([x, y])
                used.add((x, y))
        
        return pts

    for i in range(5):
        random.seed(500 + i)
        pts = generate_line_clusters()
        if len(pts) == 20 and not has_4_collinear(pts):
            candidates.append(pts)

    # --- Local Search ---
    
    def local_search(points, max_time=10):
        current_points = [p[:] for p in points]
        current_score = score_config(current_points)
        start_ls = time.time()
        improved = True
        
        while improved and (time.time() - start_ls) < max_time and (time.time() - start_time) < timeout - 5:
            improved = False
            # Shuffle order of points to avoid bias
            indices = list(range(20))
            random.shuffle(indices)
            
            for i in indices:
                if (time.time() - start_time) > timeout - 5:
                    return current_points, current_score
                
                old_x, old_y = current_points[i]
                
                # Try several random moves
                for _ in range(10):
                    # Move in a small neighborhood
                    dx = random.randint(-10, 10)
                    dy = random.randint(-10, 10)
                    new_x = old_x + dx
                    new_y = old_y + dy
                    
                    if new_x < -100 or new_x > 100 or new_y < -100 or new_y > 100:
                        continue
                    if (new_x, new_y) == (old_x, old_y):
                        continue
                    
                    # Check collision
                    collision = False
                    for j, p in enumerate(current_points):
                        if j != i and p[0] == new_x and p[1] == new_y:
                            collision = True
                            break
                    if collision:
                        continue
                    
                    current_points[i] = [new_x, new_y]
                    
                    if has_4_collinear(current_points):
                        current_points[i] = [old_x, old_y]
                        continue
                        
                    new_score = score_config(current_points)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        current_points[i] = [old_x, old_y]
                        
        return current_points, current_score

    # Run local search on candidates
    for cand in candidates:
        if time.time() - start_time > timeout - 15:
            break
        
        # Limit time per candidate
        pts, s = local_search(cand, max_time=5)
        
        if s < best_score:
            best_score = s
            best_points = pts

    # If no improvement found, try a more aggressive random search with shorter local search
    if best_score == 20:
        for i in range(10):
            if time.time() - start_time > timeout - 10:
                break
            random.seed(2000 + i)
            pts = []
            used = set()
            while len(pts) < 20:
                x = random.randint(-80, 80)
                y = random.randint(-80, 80)
                if (x, y) not in used:
                    used.add((x, y))
                    pts.append([x, y])
            if not has_4_collinear(pts):
                pts, s = local_search(pts, max_time=2)
                if s < best_score:
                    best_score = s
                    best_points = pts

    # Fallback
    if best_points is None:
        # Simple valid config: Parabola
        pts = []
        for x in range(-5, 6):
            pts.append([x, x*x])
        for x in range(-5, 6):
            pts.append([x, x*x + 20])
        best_points = pts
        best_score = score_config(best_points)

    return best_points

# EVOLVE-BLOCK-END
