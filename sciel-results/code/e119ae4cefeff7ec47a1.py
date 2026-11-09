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

    def count_bad_6_fast(n, adj_np, comb_indices=None):
        """
        Counts the number of 6-vertex subsets that induce a regular subgraph.
        Uses precomputed combination indices if provided.
        """
        if comb_indices is None:
            comb_indices = np.array(list(itertools.combinations(range(n), 6)), dtype=np.int32)
        
        count = 0
        for indices in comb_indices:
            sub_adj = adj_np[np.ix_(indices, indices)]
            degrees = sub_adj.sum(axis=1)
            if np.all(degrees == degrees[0]):
                count += 1
        return count

    def get_violating_edges(n, adj_np, comb_indices=None):
        """
        Returns a set of (u,v) edges that appear in at least one violating 6-set.
        """
        if comb_indices is None:
            comb_indices = np.array(list(itertools.combinations(range(n), 6)), dtype=np.int32)
        
        violating_edges = set()
        for indices in comb_indices:
            sub_adj = adj_np[np.ix_(indices, indices)]
            degrees = sub_adj.sum(axis=1)
            if np.all(degrees == degrees[0]):
                # Add all edges within this 6-set
                for i in range(6):
                    for j in range(i + 1, 6):
                        u, v = indices[i], indices[j]
                        violating_edges.add((min(u,v), max(u,v)))
        return violating_edges

    def local_search(n, max_iters=500):
        """
        Attempts to find a valid graph on n vertices using local search.
        Returns adj_np if valid, else None.
        """
        # Precompute combinations
        comb_indices = np.array(list(itertools.combinations(range(n), 6)), dtype=np.int32)
        
        # Initialize random graph
        adj_list = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_list[i, j] = 1
                    adj_list[j, i] = 1
        
        # Check initial
        current_score = count_bad_6_fast(n, adj_list, comb_indices)
        if current_score == 0:
            if check_validity(n, adj_list):
                return adj_list

        best_adj = adj_list.copy()
        best_score = current_score
        
        # Precompute edge list for random flipping
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # If we have violations, bias towards flipping edges in violating sets
            if current_score > 0 and it % 10 == 0:
                # Recompute violating edges periodically
                violating_edges = get_violating_edges(n, adj_list, comb_indices)
                if violating_edges:
                    # Pick from violating edges with high probability
                    if random.random() < 0.8:
                        u, v = random.choice(list(violating_edges))
                    else:
                        u, v = random.choice(edges)
                else:
                    u, v = random.choice(edges)
            else:
                u, v = random.choice(edges)
            
            # Flip edge
            adj_list[u, v] ^= 1
            adj_list[v, u] ^= 1
            
            new_score = count_bad_6_fast(n, adj_list, comb_indices)
            
            # Accept if better or equal
            if new_score <= current_score:
                current_score = new_score
                if current_score == 0:
                    # Check full validity
                    if check_validity(n, adj_list):
                        return adj_list
            else:
                # Revert
                adj_list[u, v] ^= 1
                adj_list[v, u] ^= 1
                
            # Track best
            if current_score < best_score:
                best_score = current_score
                best_adj = adj_list.copy()
        
        return None

    def try_find_graph(target_n, time_budget):
        """Try to find a valid graph on target_n vertices within time_budget seconds."""
        t_start = time.time()
        max_restarts = 10
        for i in range(max_restarts):
            if time.time() > t_start + time_budget:
                break
            adj = local_search(target_n, max_iters=300)
            if adj is not None:
                return {'n': target_n, 'edges': [[u, v] for u in range(target_n) for v in range(u + 1, target_n) if adj[u, v] == 1]}
        return None

    # Try n=21 first
    result = try_find_graph(21, 20)
    if result is not None:
        return result

    # Try n=20
    result = try_find_graph(20, 30)
    if result is not None:
        return result

    # Try n=18
    result = try_find_graph(18, 15)
    if result is not None:
        return result

    # Try n=15 as a safer fallback
    result = try_find_graph(15, 15)
    if result is not None:
        return result

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
    
    comb_indices_10 = np.array(list(itertools.combinations(range(n), 6)), dtype=np.int32)
    for _ in range(20):
        if count_bad_6_fast(n, adj_list, comb_indices_10) == 0:
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
