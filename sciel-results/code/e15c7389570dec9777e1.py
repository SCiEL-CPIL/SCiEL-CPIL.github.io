import time
import random
import math
import itertools

def _neighbors(p):
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _edge_count(points):
    """Count edges in a set of points."""
    pts_set = set(points)
    cnt = 0
    for p in pts_set:
        for q in _neighbors(p):
            if q in pts_set and p < q:
                cnt += 1
    return cnt

def _valid(points):
    """Check validity: 32 distinct points, |a|,|b| <= 64, min distance >= 1."""
    if len(points) != 32:
        return False
    if len(set(points)) != 32:
        return False
    for p in points:
        if abs(p[0]) > 64 or abs(p[1]) > 64:
            return False
    return True

def _generate_hexagon(radius):
    pts = []
    for a in range(-radius, radius + 1):
        for b in range(-radius, radius + 1):
            c = -a - b
            if max(abs(a), abs(b), abs(c)) <= radius:
                pts.append((a, b))
    return pts

def _edge_count_fast(pts_set):
    """Count edges using a precomputed set for O(1) lookups."""
    cnt = 0
    for p in pts_set:
        for q in _neighbors(p):
            if q in pts_set and p < q:
                cnt += 1
    return cnt

def _deterministic_polish(pts_set):
    """Try all single-point swaps to find improvements until no more exist."""
    curr_score = _edge_count_fast(pts_set)
    improved = True
    while improved:
        improved = False
        best_delta = 0
        best_move = None
        # Try removing each point and adding each candidate
        for p_out in list(pts_set):
            for p_in in _neighbors(p_out):
                if p_in in pts_set:
                    continue
                # Check bounds
                if abs(p_in[0]) > 64 or abs(p_in[1]) > 64:
                    continue
                # Compute delta in edge count
                # Remove p_out: lose edges between p_out and its neighbors in set
                lost = sum(1 for n in _neighbors(p_out) if n in pts_set)
                # Add p_in: gain edges between p_in and its neighbors in set (minus p_out)
                new_set_minus = pts_set - {p_out}
                gained = sum(1 for n in _neighbors(p_in) if n in new_set_minus)
                delta = gained - lost
                if delta > best_delta:
                    best_delta = delta
                    best_move = (p_out, p_in)
        if best_move is not None and best_delta > 0:
            p_out, p_in = best_move
            pts_set.remove(p_out)
            pts_set.add(p_in)
            curr_score += best_delta
            improved = True
    return pts_set, curr_score

def _pair_swap_polish(pts_set, pool):
    """Try all pair swaps to find improvements."""
    curr_score = _edge_count_fast(pts_set)
    improved = True
    while improved:
        improved = False
        best_delta = 0
        best_move = None
        pts_list = list(pts_set)
        for i in range(len(pts_list)):
            for j in range(i + 1, len(pts_list)):
                p1_out = pts_list[i]
                p2_out = pts_list[j]
                # Try adding two points from pool
                for p1_in in _neighbors(p1_out):
                    if p1_in in pts_set or p1_in == p2_out:
                        continue
                    if abs(p1_in[0]) > 64 or abs(p1_in[1]) > 64:
                        continue
                    if p1_in not in pool:
                        continue
                    for p2_in in _neighbors(p2_out):
                        if p2_in in pts_set or p2_in == p1_out:
                            continue
                        if p2_in == p1_in:
                            continue
                        if abs(p2_in[0]) > 64 or abs(p2_in[1]) > 64:
                            continue
                        if p2_in not in pool:
                            continue
                        # Compute delta
                        new_set = pts_set - {p1_out, p2_out} | {p1_in, p2_in}
                        new_score = _edge_count_fast(new_set)
                        delta = new_score - curr_score
                        if delta > best_delta:
                            best_delta = delta
                            best_move = (p1_out, p2_out, p1_in, p2_in)
        if best_move is not None and best_delta > 0:
            p1_out, p2_out, p1_in, p2_in = best_move
            pts_set.remove(p1_out)
            pts_set.remove(p2_out)
            pts_set.add(p1_in)
            pts_set.add(p2_in)
            curr_score += best_delta
            improved = True
    return pts_set, curr_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Step 1: Brute force best 32-subset of Hex(3)
    hex3 = _generate_hexagon(3)
    hex3_set = set(hex3)
    
    best_points = None
    best_score = -1
    
    for rem_combo in itertools.combinations(hex3, 5):
        rem_set = set(rem_combo)
        curr_set = hex3_set - rem_set
        score = _edge_count(list(curr_set))
        if score > best_score:
            best_score = score
            best_points = list(curr_set)
            
        if time.time() > start_time + 20.0:
            break

    if best_points is None:
        best_points = hex3[:32]
        best_score = _edge_count(best_points)

    # Step 2: Local search with multiple move types
    pool = set(_generate_hexagon(5))
    
    curr_points = set(best_points)
    curr_score = best_score
    best_global_score = best_score
    best_global_points = set(curr_points)
    
    T = 5.0
    T_min = 0.05
    cooling_rate = 0.999
    
    random.seed(42)
    
    # Precompute neighbor candidates for each point in pool
    # For efficiency, we'll generate candidates on the fly
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Choose move type: 70% single swap, 20% pair swap, 10% random restart-ish
        move_type = random.random()
        
        if move_type < 0.7:
            # Single point swap
            p_in = random.choice(list(curr_points))
            
            # Pick from neighbors of current set for locality
            candidates = []
            for p in curr_points:
                for n in _neighbors(p):
                    if n not in curr_points and n in pool:
                        candidates.append(n)
            
            if not candidates:
                T *= cooling_rate
                continue
                
            p_out = random.choice(candidates)
            
            new_set = set(curr_points)
            new_set.remove(p_in)
            new_set.add(p_out)
            
            new_score = _edge_count(list(new_set))
            
            delta = new_score - curr_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                curr_points = new_set
                curr_score = new_score
                
                if curr_score > best_global_score:
                    best_global_score = curr_score
                    best_global_points = set(curr_points)
                    
        elif move_type < 0.9:
            # Pair swap: remove two points, add two points
            if len(curr_points) < 4:
                T *= cooling_rate
                continue
                
            pts_list = list(curr_points)
            p1 = random.choice(pts_list)
            p2 = random.choice([x for x in pts_list if x != p1])
            
            # Find candidates for replacements
            cand1 = []
            for n in _neighbors(p1):
                if n not in curr_points and n in pool:
                    cand1.append(n)
            cand2 = []
            for n in _neighbors(p2):
                if n not in curr_points and n in pool and n != p1:
                    cand2.append(n)
            
            if not cand1 or not cand2:
                T *= cooling_rate
                continue
                
            r1 = random.choice(cand1)
            r2 = random.choice(cand2)
            
            if r1 == r2:
                T *= cooling_rate
                continue
                
            new_set = set(curr_points)
            new_set.remove(p1)
            new_set.remove(p2)
            new_set.add(r1)
            new_set.add(r2)
            
            new_score = _edge_count(list(new_set))
            
            delta = new_score - curr_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                curr_points = new_set
                curr_score = new_score
                
                if curr_score > best_global_score:
                    best_global_score = curr_score
                    best_global_points = set(curr_points)
                    
        else:
            # Slight perturbation: move one point to a random nearby spot
            p = random.choice(list(curr_points))
            # Pick a random offset within 2 steps
            da = random.randint(-2, 2)
            db = random.randint(-2, 2)
            if da == 0 and db == 0:
                T *= cooling_rate
                continue
            new_p = (p[0] + da, p[1] + db)
            if new_p in curr_points or new_p not in pool:
                T *= cooling_rate
                continue
            if abs(new_p[0]) > 64 or abs(new_p[1]) > 64:
                T *= cooling_rate
                continue
                
            new_set = set(curr_points)
            new_set.remove(p)
            new_set.add(new_p)
            
            new_score = _edge_count(list(new_set))
            
            delta = new_score - curr_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                curr_points = new_set
                curr_score = new_score
                
                if curr_score > best_global_score:
                    best_global_score = curr_score
                    best_global_points = set(curr_points)
                    
        T *= cooling_rate

    # Final deterministic polish
    if time.time() < deadline - 10:
        polished_set, polished_score = _deterministic_polish(set(best_global_points))
        if polished_score > best_global_score:
            best_global_score = polished_score
            best_global_points = polished_set
    
    # Final pair swap polish if time permits
    if time.time() < deadline - 5:
        polished_set2, polished_score2 = _pair_swap_polish(set(best_global_points), pool)
        if polished_score2 > best_global_score:
            best_global_score = polished_score2
            best_global_points = polished_set2

    # Final check
    if not _valid(list(best_global_points)):
        return [[a, b] for a, b in list(best_global_points)[:32]]
        
    return [[a, b] for a, b in best_global_points]
