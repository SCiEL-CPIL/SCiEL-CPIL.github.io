import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(abs(dx), abs(dy))
        if g == 0:
            return None
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        a, b = dy, -dx
        c = a * x1 + b * y1
        return (a, b, c)

    def build_line_map(points):
        n = len(points)
        line_map = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(points[i], points[j])
                if key is None:
                    continue
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
        return line_map

    def has_four_collinear(points):
        line_map = build_line_map(points)
        for pts in line_map.values():
            if len(pts) >= 4:
                return True
        return False

    def count_triples_from_line_map(line_map):
        total = 0
        for pts in line_map.values():
            k = len(pts)
            if k >= 3:
                total += k * (k - 1) * (k - 2) // 6
        return total

    def count_triples(points):
        line_map = build_line_map(points)
        return count_triples_from_line_map(line_map)

    def quick_max_no_three(points):
        """
        Quick branch and bound for largest subset with no three collinear.
        Uses a greedy upper bound: sort by degree (number of collinear pairs),
        and use a simple recursive search with pruning.
        """
        n = len(points)
        # Precompute: for each point i, which pairs (j,k) are collinear with i
        # i.e., if we pick i, we cannot pick both j and k for any (j,k) collinear with i
        collinear_pairs = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if get_line_key(points[i], points[j]) == get_line_key(points[i], points[k]):
                        collinear_pairs[i].add((j, k))
                        collinear_pairs[j].add((i, k))
                        collinear_pairs[k].add((i, j))
        
        best = 0
        
        # Order points by degree (descending) for better pruning
        order = sorted(range(n), key=lambda i: len(collinear_pairs[i]), reverse=True)
        
        def bbound(remaining_mask, current_size, current_set_mask):
            nonlocal best
            if current_size + bin(remaining_mask).count('1') <= best:
                return
            if remaining_mask == 0:
                if current_size > best:
                    best = current_size
                return
            
            # Pick the first point in the ordered list that's in remaining
            for idx in range(n):
                bit = 1 << order[idx]
                if remaining_mask & bit:
                    break
            else:
                return
            
            # Option 1: skip this point
            bbound(remaining_mask & ~bit, current_size, current_set_mask)
            
            # Option 2: include this point
            # Check conflicts: if any pair (j,k) in collinear_pairs[order[idx]] both in current_set_mask or remaining
            new_mask = remaining_mask & ~bit
            can_include = True
            for (j, k) in collinear_pairs[order[idx]]:
                bit_j = 1 << j
                bit_k = 1 << k
                # If both j and k are already in current_set, we can't include this point
                if (current_set_mask & bit_j) and (current_set_mask & bit_k):
                    can_include = False
                    break
                # If one is in current_set and the other is in remaining, we must remove the other from remaining
                if (current_set_mask & bit_j) and (new_mask & bit_k):
                    new_mask &= ~bit_k
                elif (current_set_mask & bit_k) and (new_mask & bit_j):
                    new_mask &= ~bit_j
                # If both are in remaining, we need to ensure we don't pick both later
                # This is handled by the recursive call; we just need to track it
                # For simplicity, we'll just remove one of them to maintain validity
                elif (new_mask & bit_j) and (new_mask & bit_k):
                    # Remove the one with higher degree to maximize pruning
                    if len(collinear_pairs[j]) >= len(collinear_pairs[k]):
                        new_mask &= ~bit_k
                    else:
                        new_mask &= ~bit_j
            
            if can_include:
                bbound(new_mask, current_size + 1, current_set_mask | bit)
        
        all_mask = (1 << n) - 1
        bbound(all_mask, 0, 0)
        return best

    def random_valid_points():
        while True:
            pts = []
            used = set()
            while len(pts) < 20:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in used:
                    used.add((x, y))
                    pts.append((x, y))
            if not has_four_collinear(pts):
                return pts

    # Generate initial candidates
    best_points = None
    best_score = 999
    best_triples = -1

    for _ in range(30):
        if time.time() > deadline - 15:
            break
        pts = random_valid_points()
        triples = count_triples(pts)
        if best_points is None or triples > best_triples:
            best_points = pts
            best_triples = triples

    if best_points is None:
        # Fallback: construct a valid configuration manually
        return [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2],[3,0],[3,1],[3,2],[3,3],[0,3],[1,3],[2,3],[4,0],[4,1],[4,2],[4,3]]

    current_points = best_points[:]
    current_triples = best_triples
    current_score = quick_max_no_three(current_points)
    best_score = current_score
    best_points = current_points[:]

    # Local search: move points to create more triples
    while time.time() < deadline - 5:
        improved = False
        
        for idx in range(20):
            if time.time() > deadline - 3:
                break
            
            old_point = current_points[idx]
            existing = set(current_points)
            existing.discard(old_point)
            
            # Generate targeted candidates: points on lines with 2 other points
            candidates = set()
            for j in range(20):
                if j == idx:
                    continue
                for k in range(j + 1, 20):
                    if k == idx:
                        continue
                    p_j = current_points[j]
                    p_k = current_points[k]
                    x1, y1 = p_j
                    x2, y2 = p_k
                    dx = x2 - x1
                    dy = y2 - y1
                    g = gcd(abs(dx), abs(dy))
                    if g == 0:
                        continue
                    sx = dx // g
                    sy = dy // g
                    
                    # Find integer points on this line within bounds
                    # t=0 is p_j, t=1 is p_k
                    min_t = -200
                    max_t = 200
                    for t in range(min_t, max_t + 1):
                        if t == 0 or t == 1:
                            continue
                        nx = x1 + t * sx
                        ny = y1 + t * sy
                        if -100 <= nx <= 100 and -100 <= ny <= 100:
                            candidates.add((nx, ny))
            
            # Add some random candidates for diversity
            for _ in range(20):
                candidates.add((random.randint(-100, 100), random.randint(-100, 100)))
            
            # Filter valid candidates
            valid_cands = [c for c in candidates if c not in existing]
            random.shuffle(valid_cands)
            
            best_cand = None
            best_cand_score = current_score
            best_cand_triples = current_triples
            
            for cand in valid_cands:
                if time.time() > deadline - 3:
                    break
                
                new_points = current_points[:]
                new_points[idx] = cand
                
                if has_four_collinear(new_points):
                    continue
                
                new_triples = count_triples(new_points)
                
                # Use triple count as a fast filter; if significantly better, compute exact score
                if new_triples > best_cand_triples + 2 or new_triples == best_cand_triples:
                    new_score = quick_max_no_three(new_points)
                    if new_score < best_cand_score or (new_score == best_cand_score and new_triples > best_cand_triples):
                        best_cand_score = new_score
                        best_cand_triples = new_triples
                        best_cand = cand
                        if new_score < current_score - 1:
                            break
                elif new_triples > best_cand_triples:
                    # Slightly better triples, accept if not worse in score
                    new_score = quick_max_no_three(new_points)
                    if new_score <= best_cand_score:
                        best_cand_score = new_score
                        best_cand_triples = new_triples
                        best_cand = cand
            
            if best_cand is not None:
                current_points[idx] = best_cand
                current_triples = best_cand_triples
                current_score = best_cand_score
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_points = current_points[:]
        
        if not improved:
            # Try perturbation to escape local minimum
            if random.random() < 0.3:
                idx = random.randint(0, 19)
                for _ in range(50):
                    nx = random.randint(-100, 100)
                    ny = random.randint(-100, 100)
                    if (nx, ny) not in set(current_points):
                        new_points = current_points[:]
                        new_points[idx] = (nx, ny)
                        if not has_four_collinear(new_points):
                            new_triples = count_triples(new_points)
                            new_score = quick_max_no_three(new_points)
                            if new_score <= current_score + 1:  # Accept slight worsening
                                current_points = new_points
                                current_triples = new_triples
                                current_score = new_score
                            break

    return [[p[0], p[1]] for p in best_points]
