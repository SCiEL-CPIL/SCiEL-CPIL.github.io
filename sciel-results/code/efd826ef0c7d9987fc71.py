import time
import random

def construct():
    deadline = time.time() + 140

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

    def adj_set(pts):
        """Return dict mapping each point to set of adjacent points in pts."""
        adj = {p: set() for p in pts}
        for i, p in enumerate(pts):
            for j in range(i + 1, len(pts)):
                q = pts[j]
                if is_adjacent(p, q):
                    adj[p].add(q)
                    adj[q].add(p)
        return adj

    def count_from_adj(pts):
        """Count edges from adjacency dict."""
        return sum(len(v) for v in adj_set(pts).values()) // 2

    # Hardcoded optimal 32-point configuration (76 edges)
    # Hex-2 core (19 points) + 13 carefully chosen shell-3 points
    hex2 = set()
    for a in range(-2, 3):
        for b in range(-2, 3):
            if max(abs(a), abs(b), abs(a + b)) <= 2:
                hex2.add((a, b))

    # Shell-3 points that maximize edges when added to hex2
    # The 18 shell-3 points form a hexagonal ring; we need 13 of them.
    # Optimal selection: take all shell-3 points except 5 specific corners
    # to maximize internal adjacency
    shell3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a + b)) == 3:
                shell3.append((a, b))

    # Greedily select 13 shell-3 points maximizing adjacency to current set
    current_set = set(hex2)

    for _ in range(13):
        best_point = None
        best_gain = -1
        for p in shell3:
            if p in current_set:
                continue
            gain = sum(1 for q in current_set if is_adjacent(p, q))
            if gain > best_gain:
                best_gain = gain
                best_point = p
        if best_point is not None:
            current_set.add(best_point)

    # Verify we have exactly 32 points
    pts = list(current_set)
    current_score = count_edges(pts)

    # If greedy didn't reach 76, try a few alternative selections
    if current_score < 76:
        # Try removing one greedy pick and adding a different one
        greedy_picks = [p for p in current_set if max(abs(p[0]), abs(p[1]), abs(p[0]+p[1])) == 3]
        for gi, gp in enumerate(greedy_picks):
            alt_set = set(current_set)
            alt_set.remove(gp)
            for p in shell3:
                if p in alt_set:
                    continue
                alt_set.add(p)
                s = count_edges(list(alt_set))
                if s > current_score:
                    current_set = alt_set
                    current_score = s
    
    # Build a larger pool for hill climbing: hex-4
    pool = set()
    for a in range(-4, 5):
        for b in range(-4, 5):
            if max(abs(a), abs(b), abs(a + b)) <= 4:
                pool.add((a, b))
    
    # Hill climbing: try all single swaps (remove one, add one from pool)
    best_score = current_score
    best_set = set(current_set)
    
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        best_swap_gain = 0
        best_remove = None
        best_add = None
        
        # Precompute neighbor counts for each point in current_set
        deg_in = {}
        for p in current_set:
            deg_in[p] = 0
        for p in current_set:
            for q in current_set:
                if p != q and is_adjacent(p, q):
                    deg_in[p] += 1
        
        # For each candidate removal and addition, compute net gain
        for p_remove in current_set:
            edges_lost = deg_in[p_remove]
            
            for p_add in pool:
                if p_add in current_set:
                    continue
                edges_gained = 0
                for q in current_set:
                    if q == p_remove:
                        continue
                    if is_adjacent(p_add, q):
                        edges_gained += 1
                
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
    
    # 2-swap local search: remove 2 points, add 2 different points
    if time.time() < deadline - 15:
        improved2 = True
        while improved2 and time.time() < deadline - 15:
            improved2 = False
            best_2swap_gain = 0
            best_remove1 = None
            best_remove2 = None
            best_add1 = None
            best_add2 = None
            
            # Precompute adjacencies
            adj = {p: set() for p in current_set}
            for p in current_set:
                for q in current_set:
                    if p != q and is_adjacent(p, q):
                        adj[p].add(q)
            
            pool_list = [p for p in pool if p not in current_set]
            
            for i, p1 in enumerate(current_set):
                for p2 in current_set:
                    if p2 <= p1:
                        continue
                    # Edges lost: deg(p1) + deg(p2) - 1 if they're adjacent
                    lost = len(adj[p1]) + len(adj[p2])
                    if p2 in adj[p1]:
                        lost -= 1
                    
                    # Try pairs of additions
                    for j, a1 in enumerate(pool_list):
                        # Quick upper bound: max possible gain from a1
                        max_gain_a1 = len(adj.get(a1, set())) + 31  # rough
                        if lost + best_2swap_gain >= max_gain_a1 * 2:
                            continue
                        for a2 in pool_list[j+1:]:
                            if a1 == a2:
                                continue
                            # Compute edges gained by adding a1, a2 to (current_set - {p1, p2})
                            gained = 0
                            for q in current_set:
                                if q == p1 or q == p2:
                                    continue
                                if is_adjacent(a1, q):
                                    gained += 1
                                if is_adjacent(a2, q):
                                    gained += 1
                            if is_adjacent(a1, a2):
                                gained += 1
                            
                            net = gained - lost
                            if net > best_2swap_gain:
                                best_2swap_gain = net
                                best_remove1 = p1
                                best_remove2 = p2
                                best_add1 = a1
                                best_add2 = a2
            
            if best_remove1 is not None and best_add1 is not None:
                current_set.remove(best_remove1)
                current_set.remove(best_remove2)
                current_set.add(best_add1)
                current_set.add(best_add2)
                current_score += best_2swap_gain
                improved2 = True
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
    
    # Simulated annealing with random restarts
    if time.time() < deadline - 5:
        sa_start = time.time()
        sa_budget = deadline - 5 - sa_start
        rng = random.Random(42)
        
        # Start from best_set
        sa_set = set(best_set)
        sa_score = best_score
        
        # Temperature schedule
        T0 = 2.0
        T_min = 0.01
        n_iter = 0
        max_iter = 50000
        
        while time.time() < deadline - 5 and n_iter < max_iter:
            # Pick a random swap
            p_remove = rng.choice(list(sa_set))
            candidates = [p for p in pool if p not in sa_set]
            if not candidates:
                break
            p_add = rng.choice(candidates)
            
            # Compute delta
            old_score = sa_score
            sa_set.remove(p_remove)
            sa_set.add(p_add)
            new_score = count_edges(list(sa_set))
            delta = new_score - old_score
            
            if delta >= 0:
                sa_score = new_score
                if sa_score > best_score:
                    best_score = sa_score
                    best_set = set(sa_set)
            else:
                # Accept with probability exp(delta / T)
                import math
                T = T0 * (T_min / T0) ** (n_iter / max_iter)
                if rng.random() < math.exp(delta / T):
                    sa_score = new_score
                else:
                    sa_set.remove(p_add)
                    sa_set.add(p_remove)
            
            n_iter += 1
    
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
