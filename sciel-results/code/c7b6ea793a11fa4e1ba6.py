import time
import random

def construct():
    deadline = time.time() + 140
    rng = random.Random(42)
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + db*db + da*db

    def is_adjacent(p, q):
        return dist_sq(p, q) == 1

    def count_edges(points):
        cnt = 0
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if dist_sq(points[i], points[j]) == 1:
                    cnt += 1
        return cnt

    # Build hex-2 core (19 points)
    hex2 = set()
    for a in range(-2, 3):
        for b in range(-2, 3):
            if max(abs(a), abs(b), abs(a + b)) <= 2:
                hex2.add((a, b))
    
    # Build shell-3 (18 points)
    shell3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a + b)) == 3:
                shell3.append((a, b))
    
    # Greedily add 13 points from shell3
    current_set = set(hex2)
    selected_shell = set()
    
    for _ in range(13):
        best_point = None
        best_gain = -1
        for p in shell3:
            if p in selected_shell:
                continue
            gain = 0
            for q in current_set:
                if is_adjacent(p, q):
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_point = p
        if best_point is not None:
            selected_shell.add(best_point)
            current_set.add(best_point)
    
    # Now we have 32 points. Compute initial score.
    pts = list(current_set)
    current_score = count_edges(pts)
    
    # Build a larger pool for hill climbing: hex-5
    pool = set()
    for a in range(-5, 6):
        for b in range(-5, 6):
            if max(abs(a), abs(b), abs(a + b)) <= 5:
                pool.add((a, b))
    
    # Precompute adjacency for all pool points
    # adj[p] = set of points in pool adjacent to p
    adj = {}
    for p in pool:
        adj[p] = set()
    for p in pool:
        for q in pool:
            if p != q and is_adjacent(p, q):
                adj[p].add(q)
    
    def get_deg_in(p, s):
        """Number of neighbors of p in set s"""
        return len(adj[p] & s)
    
    def get_edges_in(p, s):
        """Number of edges from p to points in s"""
        return len(adj[p] & s)
    
    def total_edges(s):
        """Total edges in set s"""
        cnt = 0
        for p in s:
            cnt += get_deg_in(p, s)
        return cnt // 2
    
    # Hill climbing: try all single swaps (remove one, add one from pool)
    best_score = current_score
    best_set = set(current_set)
    
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        best_swap_gain = 0
        best_remove = None
        best_add = None
        
        # Precompute neighbor counts for each point in current_set
        deg_in = {}
        for p in current_set:
            deg_in[p] = get_deg_in(p, current_set)
        
        # For each candidate removal and addition, compute net gain
        for p_remove in current_set:
            edges_lost = deg_in[p_remove]
            
            for p_add in pool:
                if p_add in current_set:
                    continue
                # Edges gained when adding p_add to (current_set - {p_remove})
                edges_gained = get_deg_in(p_add, current_set)
                # But if p_add is adjacent to p_remove, we don't count that edge
                if p_add in adj[p_remove]:
                    edges_gained -= 1
                
                net_gain = edges_gained - edges_lost
                if net_gain > best_swap_gain:
                    best_swap_gain = net_gain
                    best_remove = p_remove
                    best_add = p_add
        
        if best_remove is not None and best_add is not None:
            current_set.remove(best_remove)
            current_set.add(best_add)
            current_score += best_swap_gain
            improved = True
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
        
        # 2-swap move: remove two, add two
        if time.time() < deadline - 15:
            best_2swap_gain = 0
            best_remove1 = None
            best_remove2 = None
            best_add1 = None
            best_add2 = None
            
            # Try 2-swap moves
            cur_list = list(current_set)
            pool_list = list(pool - current_set)
            
            # Limit 2-swap search for speed
            max_2swap_pairs = 500
            attempts = 0
            
            for i in range(len(cur_list)):
                for j in range(i + 1, len(cur_list)):
                    if attempts >= max_2swap_pairs:
                        break
                    p_r1 = cur_list[i]
                    p_r2 = cur_list[j]
                    
                    # Edges lost
                    edges_lost = deg_in[p_r1] + deg_in[p_r2]
                    if p_r1 in adj[p_r2]:
                        edges_lost -= 1  # Don't double count the edge between them
                    
                    # Try random additions
                    for _ in range(3):
                        attempts += 1
                        if not pool_list:
                            break
                        a1_idx = rng.randint(0, len(pool_list) - 1)
                        a2_idx = rng.randint(0, len(pool_list) - 1)
                        if a2_idx == a1_idx:
                            a2_idx = (a2_idx + 1) % len(pool_list)
                        p_a1 = pool_list[a1_idx]
                        p_a2 = pool_list[a2_idx]
                        
                        # Edges gained
                        edges_gained = get_deg_in(p_a1, current_set) + get_deg_in(p_a2, current_set)
                        # Subtract edges to removed points
                        if p_a1 in adj[p_r1]:
                            edges_gained -= 1
                        if p_a1 in adj[p_r2]:
                            edges_gained -= 1
                        if p_a2 in adj[p_r1]:
                            edges_gained -= 1
                        if p_a2 in adj[p_r2]:
                            edges_gained -= 1
                        # If both new points are adjacent to each other, count that edge
                        if p_a1 in adj[p_a2]:
                            edges_gained += 1
                        
                        net_gain = edges_gained - edges_lost
                        if net_gain > best_2swap_gain:
                            best_2swap_gain = net_gain
                            best_remove1 = p_r1
                            best_remove2 = p_r2
                            best_add1 = p_a1
                            best_add2 = p_a2
                
                if attempts >= max_2swap_pairs:
                    break
            
            if best_remove1 is not None and best_add1 is not None:
                current_set.remove(best_remove1)
                current_set.remove(best_remove2)
                current_set.add(best_add1)
                current_set.add(best_add2)
                current_score += best_2swap_gain
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
    
    # Return the best set found
    result = [[a, b] for a, b in best_set]
    
    # Safety check: ensure exactly 32 distinct points within bounds
    if len(result) != 32:
        result = [[a, b] for a, b in current_set]
    
    # Verify all points are within bounds and distinct
    seen = set()
    valid = []
    for p in result:
        a, b = p
        if abs(a) <= 64 and abs(b) <= 64 and (a, b) not in seen:
            seen.add((a, b))
            valid.append([a, b])
    
    if len(valid) == 32:
        return valid
    
    # Last resort fallback: known valid 32-point configuration
    hex3 = set()
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a + b)) <= 3:
                hex3.add((a, b))
    
    pts3 = list(hex3)
    for _ in range(5):
        min_deg = float('inf')
        min_idx = 0
        for i, p in enumerate(pts3):
            deg = 0
            for q in pts3:
                if p != q and is_adjacent(p, q):
                    deg += 1
            if deg < min_deg:
                min_deg = deg
                min_idx = i
        pts3.pop(min_idx)
    
    return [[a, b] for a, b in pts3]
