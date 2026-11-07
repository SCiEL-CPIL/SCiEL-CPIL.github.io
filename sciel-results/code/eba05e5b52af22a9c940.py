"""Triangle-free diameter-four augmentation: combined path-structure + set-cover + SA."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    n = P['n']
    base_edges = P['edges']

    # Build base adjacency as bitmask for fast triangle checks
    base_adj_mask = [0] * n
    for u, v in base_edges:
        base_adj_mask[u] |= (1 << v)
        base_adj_mask[v] |= (1 << u)

    # Precompute candidate edges: (u,v) with |u-v| >= 3
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if v - u >= 3:
                candidates.append((u, v))

    def validate_fast(added_edges):
        """Validate: no duplicates, no base edges, triangle-free, connected, diameter <= 4."""
        if len(added_edges) == 0:
            return False
        
        edge_set = set()
        for u, v in added_edges:
            a, b = (u, v) if u < v else (v, u)
            if a == b:
                return False
            if (a, b) in edge_set:
                return False
            edge_set.add((a, b))
            if base_adj_mask[a] & (1 << b):
                return False
        
        adj_mask = base_adj_mask[:]
        for u, v in added_edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Triangle-free check
        for u in range(n):
            for v in range(u + 1, n):
                if adj_mask[u] & (1 << v):
                    common = adj_mask[u] & adj_mask[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    if common != 0:
                        return False
        
        # BFS for diameter
        max_diam = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = [src]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                du = dist[u]
                m = adj_mask[u]
                while m:
                    v = (m & -m).bit_length() - 1
                    m &= m - 1
                    if dist[v] == -1:
                        dist[v] = du + 1
                        queue.append(v)
            for d in dist:
                if d == -1:
                    return False
                if d > max_diam:
                    max_diam = d
        
        return max_diam <= 4

    # Greedy set-cover approach
    def greedy_initial():
        adj_mask = base_adj_mask[:]
        selected = []
        
        def compute_diameter(adj_mask):
            max_diam = 0
            for src in range(n):
                dist = [-1] * n
                dist[src] = 0
                queue = [src]
                head = 0
                while head < len(queue):
                    u = queue[head]
                    head += 1
                    du = dist[u]
                    m = adj_mask[u]
                    while m:
                        v = (m & -m).bit_length() - 1
                        m &= m - 1
                        if dist[v] == -1:
                            dist[v] = du + 1
                            queue.append(v)
                for d in dist:
                    if d == -1:
                        return -1
                    if d > max_diam:
                        max_diam = d
            return max_diam
        
        current_diam = compute_diameter(adj_mask)
        if current_diam <= 4:
            return []
        
        used_edges = set()
        while current_diam > 4:
            best_edge = None
            best_diam = current_diam
            for u, v in candidates:
                if (u, v) in used_edges or (v, u) in used_edges:
                    continue
                if adj_mask[u] & (1 << v):
                    continue
                common = adj_mask[u] & adj_mask[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                if common != 0:
                    continue
                
                new_adj = adj_mask[:]
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                new_diam = compute_diameter(new_adj)
                if new_diam < best_diam:
                    best_diam = new_diam
                    best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u, v = best_edge
            selected.append([u, v])
            used_edges.add((u, v))
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            current_diam = best_diam
        
        return selected
    
    # Simulated annealing
    def simulated_annealing(initial_edges, max_iters=20000):
        if not initial_edges:
            return []
        
        current = [tuple(e) for e in initial_edges]
        current_cost = len(current)
        best = current[:]
        best_cost = current_cost
        
        all_cands = candidates
        T = 1.0
        cooling = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            move_type = random.random()
            
            if move_type < 0.3 and len(current) > 1:
                idx = random.randint(0, len(current) - 1)
                new_sol = current[:idx] + current[idx+1:]
            elif move_type < 0.7 and len(current) >= 1:
                idx = random.randint(0, len(current) - 1)
                new_sol = current[:idx] + current[idx+1:]
                for _ in range(30):
                    cand = random.choice(all_cands)
                    if cand not in new_sol:
                        new_sol_test = new_sol + [cand]
                        if validate_fast(new_sol_test):
                            new_sol = new_sol_test
                            break
                else:
                    continue
            else:
                if len(current) < 12:
                    for _ in range(30):
                        cand = random.choice(all_cands)
                        if cand not in current:
                            new_sol = current + [cand]
                            if validate_fast(new_sol):
                                break
                    else:
                        continue
                else:
                    continue
            
            new_cost = len(new_sol)
            delta_cost = new_cost - current_cost
            
            if delta_cost <= 0 or random.random() < pow(2.718281828, -delta_cost / max(T, 0.01)):
                current = new_sol
                current_cost = new_cost
                if current_cost < best_cost:
                    best = current[:]
                    best_cost = current_cost
            
            T *= cooling
        
        return [list(e) for e in best]
    
    # Get initial solution
    greedy_sol = greedy_initial()
    if not validate_fast(greedy_sol):
        greedy_sol = [[12, 1], [12, 3], [12, 6], [12, 9], [12, 15], [12, 18], [12, 21], [12, 23]]
        if not validate_fast(greedy_sol):
            greedy_sol = [[0, 4], [4, 8], [8, 12], [12, 16], [16, 20], [20, 23], [1, 5], [21, 23]]
    
    best_sol = greedy_sol
    best_cost = len(greedy_sol)
    
    # Try to find smaller solutions via targeted random search
    for k in range(3, min(best_cost, 7)):
        if time.time() > deadline - 10:
            break
        
        num_trials = 50000
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
            combo = [list(e) for e in random.sample(candidates, k)]
            if validate_fast(combo):
                best_sol = combo
                best_cost = k
                break
        
        if best_cost == k:
            break  # found a solution of size k
    
    # SA to further reduce
    if best_cost > 3 and time.time() < deadline - 10:
        sa_sol = simulated_annealing(best_sol, max_iters=30000)
        if validate_fast(sa_sol) and len(sa_sol) < best_cost:
            best_sol = sa_sol
            best_cost = len(best_sol)
    
    # Final validation
    if not validate_fast(best_sol):
        fallback = [[12, 1], [12, 3], [12, 6], [12, 9], [12, 15], [12, 18], [12, 21], [12, 23]]
        if validate_fast(fallback):
            best_sol = fallback
        else:
            fallback2 = [[0, 4], [4, 8], [8, 12], [12, 16], [16, 20], [20, 23]]
            if validate_fast(fallback2):
                best_sol = fallback2
            else:
                best_sol = [[0, 5], [5, 10], [10, 15], [15, 20], [20, 23], [1, 4], [21, 23], [2, 6]]
    
    return [list(e) for e in best_sol]

# EVOLVE-BLOCK-END
