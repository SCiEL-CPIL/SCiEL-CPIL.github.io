import time
import random
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for evaluation overhead

    def check_validity(n, adj_np):
        """
        Checks if the graph has no regular induced subgraph of size >= 6.
        adj_np: numpy array of shape (n, n) with 0/1.
        """
        def is_reg(indices):
            k = len(indices)
            if k < 2:
                return True
            sub_adj = adj_np[np.ix_(indices, indices)]
            degrees = np.sum(sub_adj, axis=1)
            return np.all(degrees == degrees[0])

        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                if is_reg(comb):
                    return False
        return True

    def count_bad_6(n, adj_np):
        """
        Counts the number of 6-vertex subsets that induce a regular subgraph.
        """
        count = 0
        for comb in itertools.combinations(range(n), 6):
            indices = np.array(comb)
            sub_adj = adj_np[np.ix_(indices, indices)]
            degrees = np.sum(sub_adj, axis=1)
            if np.all(degrees == degrees[0]):
                count += 1
        return count

    def greedy_vertex_addition(target_n, max_restarts=5):
        """
        Greedy sequential vertex addition: start with 5 vertices (no 6-subsets yet),
        then add vertices one by one, choosing edges to minimize new 6-regular violations.
        """
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Start with 5 random vertices
            n_current = 5
            adj = np.zeros((target_n, target_n), dtype=np.uint8)
            for i in range(n_current):
                for j in range(i + 1, n_current):
                    if random.random() < 0.5:
                        adj[i, j] = 1
                        adj[j, i] = 1
            
            # Add vertices 5 through target_n-1
            for new_v in range(n_current, target_n):
                best_edge_pattern = None
                best_new_violations = float('inf')
                
                # Try several random edge patterns to existing vertices
                n_trials = min(50, max(10, (target_n - new_v) * 2))
                for trial in range(n_trials):
                    # Random edge pattern to existing vertices
                    pattern = np.zeros(n_current, dtype=np.uint8)
                    for i in range(n_current):
                        if random.random() < 0.5:
                            pattern[i] = 1
                    
                    # Temporarily add this edge pattern
                    adj[new_v, :n_current] = pattern
                    adj[:n_current, new_v] = pattern
                    
                    # Count new 6-regular violations involving new_v
                    # A 6-set containing new_v and 5 from existing vertices
                    new_violations = 0
                    existing_indices = list(range(n_current))
                    for five_set in itertools.combinations(existing_indices, 5):
                        comb = list(five_set) + [new_v]
                        indices = np.array(comb)
                        sub_adj = adj[np.ix_(indices, indices)]
                        degrees = np.sum(sub_adj, axis=1)
                        if np.all(degrees == degrees[0]):
                            new_violations += 1
                    
                    if new_violations < best_new_violations:
                        best_new_violations = new_violations
                        best_edge_pattern = pattern.copy()
                    
                    # Reset
                    adj[new_v, :n_current] = 0
                    adj[:n_current, new_v] = 0
                
                # Apply best pattern
                if best_edge_pattern is not None:
                    adj[new_v, :n_current] = best_edge_pattern
                    adj[:n_current, new_v] = best_edge_pattern
                else:
                    # All-zero pattern
                    adj[new_v, :n_current] = 0
                    adj[:n_current, new_v] = 0
                
                n_current += 1
            
            # Check if valid
            if count_bad_6(target_n, adj) == 0:
                if check_validity(target_n, adj):
                    return adj[:target_n, :target_n]
        
        return None

    def local_search(n, max_iters=500):
        """
        Attempts to find a valid graph on n vertices using local search.
        Returns adj_np if valid, else None.
        """
        adj_list = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_list[i, j] = 1
                    adj_list[j, i] = 1
        
        if count_bad_6(n, adj_list) == 0:
            if check_validity(n, adj_list):
                return adj_list

        best_adj = adj_list.copy()
        best_score = count_bad_6(n, adj_list)
        current_score = best_score
        
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            u, v = random.choice(edges)
            
            adj_list[u, v] ^= 1
            adj_list[v, u] ^= 1
            
            new_score = count_bad_6(n, adj_list)
            
            if new_score <= current_score:
                current_score = new_score
                if current_score == 0:
                    if check_validity(n, adj_list):
                        return adj_list
            else:
                adj_list[u, v] ^= 1
                adj_list[v, u] ^= 1
                
            if current_score < best_score:
                best_score = current_score
                best_adj = adj_list.copy()
        
        return None

    # Strategy 1: Greedy vertex addition for n=21
    t_start = time.time()
    target_n = 21
    max_restarts = 3
    for i in range(max_restarts):
        if time.time() > t_start + 20:  # 20s budget for n=21
            break
        adj = greedy_vertex_addition(target_n, max_restarts=1)
        if adj is not None:
            return {'n': target_n, 'edges': [[u, v] for u in range(target_n) for v in range(u + 1, target_n) if adj[u, v] == 1]}

    # Strategy 2: Local search for n=21
    t_start = time.time()
    target_n = 21
    max_restarts = 3
    for i in range(max_restarts):
        if time.time() > t_start + 15:  # 15s budget for n=21
            break
        adj = local_search(target_n, max_iters=200)
        if adj is not None:
            return {'n': target_n, 'edges': [[u, v] for u in range(target_n) for v in range(u + 1, target_n) if adj[u, v] == 1]}

    # Strategy 3: Greedy vertex addition for n=20
    t_start = time.time()
    target_n = 20
    max_restarts = 5
    for i in range(max_restarts):
        if time.time() > t_start + 20:  # 20s budget for n=20
            break
        adj = greedy_vertex_addition(target_n, max_restarts=1)
        if adj is not None:
            return {'n': target_n, 'edges': [[u, v] for u in range(target_n) for v in range(u + 1, target_n) if adj[u, v] == 1]}

    # Strategy 4: Local search for n=20
    t_start = time.time()
    target_n = 20
    max_restarts = 5
    for i in range(max_restarts):
        if time.time() > t_start + 20:  # 20s budget for n=20
            break
        adj = local_search(target_n, max_iters=500)
        if adj is not None:
            return {'n': target_n, 'edges': [[u, v] for u in range(target_n) for v in range(u + 1, target_n) if adj[u, v] == 1]}

    # Try n=15 as a safer fallback
    t_start = time.time()
    target_n = 15
    max_restarts = 10
    for i in range(max_restarts):
        if time.time() > t_start + 10:
            break
        adj = local_search(target_n, max_iters=200)
        if adj is not None:
            return {'n': target_n, 'edges': [[u, v] for u in range(target_n) for v in range(u + 1, target_n) if adj[u, v] == 1]}

    # Fallback: Return a known valid small graph.
    n = 10
    adj_list = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj_list[i, j] = 1
                adj_list[j, i] = 1
    adj_list[0, :] = 0
    adj_list[:, 0] = 0
    for _ in range(10):
        if count_bad_6(n, adj_list) == 0:
            if check_validity(n, adj_list):
                return {'n': n, 'edges': [[u, v] for u in range(n) for v in range(u + 1, n) if adj_list[u, v] == 1]}
        u, v = random.randint(0, n-1), random.randint(0, n-1)
        if u != v and u != 0 and v != 0:
            adj_list[u, v] ^= 1
            adj_list[v, u] ^= 1

    # Ultimate fallback: n=6 K6 minus edge
    edges = []
    for u in range(6):
        for v in range(u + 1, 6):
            if not (u == 0 and v == 1):
                edges.append([u, v])
    return {'n': 6, 'edges': edges}
