import random
import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)
    
    # Corrected are_collinear
    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

    def has_4_collinear(points):
        n = len(points)
        # Use a set of lines to avoid O(N^4)
        lines = set()
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    continue
                g = math.gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                    dx_n = -dx_n
                    dy_n = -dy_n
                # Line key: (dx_n, dy_n, offset)
                # offset = dy_n * x1 - dx_n * y1
                offset = dy_n * x1 - dx_n * y1
                key = (dx_n, dy_n, offset)
                if key in lines:
                    # If we've seen this line before, it already had at least 2 points.
                    # If we see it again with a new pair, it means at least 3 points?
                    # Not necessarily. (0,0)-(1,1) and (2,2)-(3,3) are same line.
                    # We need to count points.
                    # Better: just check count of points on line defined by i,j
                    count = 0
                    for k in range(n):
                        if are_collinear(points[i], points[j], points[k]):
                            count += 1
                    if count >= 4:
                        return True
                else:
                    lines.add(key)
        return False

    def count_triples(points):
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
        return count

    def max_no_3_collinear(points):
        n = len(points)
        # Precompute collinear triples
        triples = []
        collinear_with = [set() for _ in range(n)] # collinear_with[i] contains (j,k) pairs? 
        # Actually, we need: for a given i, and a chosen set C, if we add i, 
        # which candidates become forbidden?
        # Forbidden if there exists r in C such that {i, c, r} is collinear.
        
        # Let's store: collinear_pairs[i] = set of (j, k) such that i,j,k collinear.
        # But we need to check against C.
        # If r in C, and (j, k) is in collinear_pairs[i] and r==j, then k is forbidden?
        # Yes.
        
        # Better structure:
        # For each i, and each j, store the set of k such that i,j,k collinear.
        # collinear[i][j] = set of k.
        
        collinear = [[set() for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        collinear[i][j].add(k)
                        collinear[j][i].add(k)
                        collinear[i][k].add(j)
                        collinear[k][i].add(j)
                        collinear[j][k].add(i)
                        collinear[k][j].add(i)
        
        best = [0]
        
        def backtrack(chosen, candidates, depth):
            if depth + len(candidates) <= best[0]:
                return
            if not candidates:
                if depth > best[0]:
                    best[0] = depth
                return
            
            # Heuristic: pick candidate with highest "conflict" potential
            # Or just pick the first one.
            # To improve pruning, pick a point that is involved in many triples with other candidates?
            # Let's just pick the first one for simplicity, but order candidates by index.
            
            p = candidates[0]
            rest_candidates = candidates[1:]
            
            # Option 1: Skip p
            backtrack(chosen, rest_candidates, depth)
            
            # Option 2: Take p
            # Check if p is collinear with any two in chosen
            valid = True
            for a in chosen:
                for b in chosen:
                    if a == b: continue
                    if b in collinear[a][p]: # a, b, p collinear
                        valid = False
                        break
                if not valid: break
            
            if valid:
                new_chosen = chosen + [p]
                new_candidates = []
                for q in rest_candidates:
                    # If we take p, can we take q?
                    # q is forbidden if there exists r in new_chosen such that p, q, r are collinear.
                    # Note: r can be p? No, distinct points.
                    # So check r in new_chosen (which includes p).
                    # If r == p, check if p, q, p collinear? No, distinct.
                    # So we check r in new_chosen.
                    forbidden = False
                    for r in new_chosen:
                        if q in collinear[p][r]: # p, q, r collinear
                            forbidden = True
                            break
                    if not forbidden:
                        new_candidates.append(q)
                
                backtrack(new_chosen, new_candidates, depth + 1)
        
        backtrack([], list(range(n)), 0)
        return best[0]

    def local_search(points, time_limit):
        pts = [list(p) for p in points]
        current_triples = count_triples(pts)
        
        while time.time() < time_limit:
            improved = False
            # Shuffle points to try different orders
            indices = list(range(20))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > time_limit:
                    break
                orig_x, orig_y = pts[i]
                # Try nearby moves
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = orig_x + dx, orig_y + dy
                        if abs(nx) > 100 or abs(ny) > 100:
                            continue
                        
                        cand = [list(p) for p in pts]
                        cand[i] = [nx, ny]
                        
                        # Check distinct
                        if len(set(map(tuple, cand))) < 20:
                            continue
                        
                        # Check no 4 collinear
                        if has_4_collinear(cand):
                            continue
                        
                        new_triples = count_triples(cand)
                        if new_triples > current_triples:
                            pts = cand
                            current_triples = new_triples
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break
        
        return pts

    best_score = 20
    best_config = None
    
    # Generate initial configurations
    configs = []
    
    # 1. Grid Perturbations
    for seed in range(100):
        random.seed(seed)
        pts = []
        # 5x4 grid
        for i in range(5):
            for j in range(4):
                x = i * 10 + random.randint(-2, 2)
                y = j * 10 + random.randint(-2, 2)
                pts.append([x, y])
        
        # Check validity
        if len(set(map(tuple, pts))) == 20 and not has_4_collinear(pts):
            configs.append(pts)
            
    # 2. Random in small box
    for seed in range(100):
        random.seed(seed + 1000)
        pts = []
        for _ in range(20):
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            pts.append([x, y])
        if len(set(map(tuple, pts))) == 20 and not has_4_collinear(pts):
            configs.append(pts)

    # Evaluate and improve
    for pts in configs:
        if time.time() > deadline - 10:
            break
        
        # Local search
        improved_pts = local_search(pts, deadline - 10)
        
        # Exact score
        score = max_no_3_collinear(improved_pts)
        
        if score < best_score:
            best_score = score
            best_config = improved_pts
            
        if best_score == 10: # Theoretical lower bound for n=20 is around 10-12? 
            # If we hit a very low score, we might stop early if time is tight
            # But let's keep searching to ensure validity or find even lower (unlikely)
            pass

    if best_config is None:
        # Fallback: simple grid
        best_config = [[i*5, j*5] for i in range(4) for j in range(5)]
        # Ensure no 4 collinear? 4x5 grid has 5 in a row. Invalid.
        # Perturb it
        for i in range(20):
            best_config[i][0] += random.randint(-1, 1)
            best_config[i][1] += random.randint(-1, 1)
        # Force distinct
        seen = set()
        for p in best_config:
            while tuple(p) in seen:
                p[0] += 1
            seen.add(tuple(p))
        # This fallback is weak, but better than None

    return best_config
