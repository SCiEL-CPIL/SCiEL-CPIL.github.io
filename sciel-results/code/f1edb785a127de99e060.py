import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # 1. Helper functions
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + db*db + da*db

    def is_adjacent(p, q):
        return dist_sq(p, q) == 1

    # 2. Generate Candidate Pool
    R = 4
    pool = []
    pool_set = set()
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if max(abs(a), abs(b), abs(a+b)) <= R:
                p = (a, b)
                pool.append(p)
                pool_set.add(p)
    
    n_pool = len(pool)
    pool_index_map = {p: i for i, p in enumerate(pool)}
    
    # 3. Pre-compute adjacency in the pool
    adj = [set() for _ in range(n_pool)]
    for i in range(n_pool):
        p = pool[i]
        for da, db in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
            nb = (p[0]+da, p[1]+db)
            if nb in pool_index_map:
                j = pool_index_map[nb]
                adj[i].add(j)
    
    def get_initial_indices(seed_offset=0):
        """Generate a random or heuristic initial set of 32 indices."""
        indices = set()
        
        # Add Hex-2 core (19 points)
        for i, p in enumerate(pool):
            a, b = p
            if max(abs(a), abs(b), abs(a+b)) <= 2:
                indices.add(i)
                
        # Need 32 - 19 = 13 more points
        remaining = [i for i in range(n_pool) if i not in indices]
        random.shuffle(remaining, random.Random(seed_offset))
        
        # Greedy add to maximize edges
        current_indices = set(indices)
        for _ in range(13):
            best_idx = None
            best_gain = -1
            for j in remaining:
                if j in current_indices:
                    continue
                gain = len(adj[j] & current_indices)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = j
            if best_idx is not None:
                current_indices.add(best_idx)
                remaining.remove(best_idx)
            else:
                if remaining:
                    current_indices.add(remaining.pop())
                    
        return list(current_indices)

    def hill_climb(initial_indices, deadline):
        """Perform 1-swap hill climbing. Returns best indices and score."""
        current_indices = set(initial_indices)
        
        # Calculate initial score and degrees
        deg_in_S = [0] * n_pool
        for i in current_indices:
            for k in adj[i]:
                deg_in_S[k] += 1
                
        score = 0
        for i in current_indices:
            score += deg_in_S[i]
        score //= 2
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_i = None
            best_j = None
            
            # Try all swaps
            for i in current_indices:
                deg_i = deg_in_S[i]
                for j in range(n_pool):
                    if j in current_indices:
                        continue
                    
                    is_adj = 1 if i in adj[j] else 0
                    delta = (deg_in_S[j] - is_adj) - deg_i
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_i is not None:
                current_indices.remove(best_i)
                current_indices.add(best_j)
                
                for k in adj[best_i]:
                    deg_in_S[k] -= 1
                for k in adj[best_j]:
                    deg_in_S[k] += 1
                    
                score += best_delta
                improved = True
                
                if score >= 76:
                    break
                    
        return current_indices, score

    best_score = -1
    best_indices = None
    
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1
        init_idx = get_initial_indices(restart_count)
        idx, score = hill_climb(init_idx, deadline)
        
        if score > best_score:
            best_score = score
            best_indices = idx
            
        if best_score >= 76:
            break
            
    if best_indices is None:
        best_indices = get_initial_indices(0)
        
    final_points = [pool[i] for i in best_indices]
    return [[a, b] for a, b in final_points]
