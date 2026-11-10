import time
import itertools
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    n = 16
    total_possible_edges = n * (n - 1) // 2  # 120
    
    # Q3 structure: 3-cube with 8 vertices, 12 edges
    Q3_EDGES = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    
    # Precompute Q3 adjacency as bitmasks for fast checking
    q3_adj_mask = [0] * 8
    for u, v in Q3_EDGES:
        q3_adj_mask[u] |= (1 << v)
        q3_adj_mask[v] |= (1 << u)
    
    def find_one_q3(adj_mask):
        """
        Fast Q3 detection using bitmask operations.
        Returns (True, edge_list) if a Q3 is found, (False, None) otherwise.
        adj_mask: list of 16 integers, each a bitmask of neighbors.
        """
        # Iterate over all 8-element subsets
        for S in itertools.combinations(range(n), 8):
            S_list = list(S)
            
            # Quick filter: each vertex must have degree >= 3 within S
            ok = True
            for v in S:
                deg = 0
                mask = adj_mask[v]
                for u in S:
                    if u != v and (mask >> u) & 1:
                        deg += 1
                if deg < 3:
                    ok = False
                    break
            if not ok:
                continue
            
            # Backtracking with bitmask pruning
            mapping = [-1] * 8
            used = 0  # bitmask of used vertices in S
            
            def dfs(idx):
                nonlocal used
                if time.time() > deadline - 0.5:
                    return False # Abort if too slow
                
                if idx == 8:
                    return True
                
                # Neighbors of idx in Q3 that are already assigned (index < idx)
                assigned_nbs = []
                for nb in range(8):
                    if nb < idx and (q3_adj_mask[idx] >> nb) & 1:
                        assigned_nbs.append(nb)
                
                # Try each unused vertex in S
                for v in S_list:
                    if (used >> v) & 1:
                        continue
                    
                    # Check if v is adjacent to all already-assigned neighbors of idx
                    valid = True
                    for nb in assigned_nbs:
                        if not (adj_mask[mapping[nb]] >> v) & 1:
                            valid = False
                            break
                    
                    if valid:
                        mapping[idx] = v
                        used |= (1 << v)
                        if dfs(idx + 1):
                            return True
                        used &= ~(1 << v)
                        mapping[idx] = -1
                
                return False
            
            if dfs(0):
                # Extract the 12 edges of the found Q3
                edges = []
                for u, v in Q3_EDGES:
                    edges.append((mapping[u], mapping[v]))
                return (True, edges)
        
        return (False, None)
    
    def get_edge_list(adj_mask):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    edges.append([u, v])
        return edges
    
    # Seed: K_{3,13}
    best_m = 39
    best_adj = [0] * n
    for u in range(3):
        for v in range(3, 16):
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)
    best_edges = get_edge_list(best_adj)
    
    # Strategy: Multiple greedy removals from different starting points
    # with randomized edge selection to escape local minima.
    
    def _make_k16():
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    def _make_k4_12():
        adj = [0] * n
        for u in range(4):
            for v in range(4, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def _make_k5_11():
        adj = [0] * n
        for u in range(5):
            for v in range(5, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def _make_k6_10():
        adj = [0] * n
        for u in range(6):
            for v in range(6, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def _make_random(p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    strategies = [
        ("K16", _make_k16),
        ("K4,12", _make_k4_12),
        ("K5,11", _make_k5_11),
        ("Random0.85", lambda: _make_random(0.85, 42)),
        ("Random0.9", lambda: _make_random(0.9, 123)),
        ("Random0.95", lambda: _make_random(0.95, 777)),
        ("K6,10", _make_k6_10),
    ]
    
    for strat_name, make_fn in strategies:
        if time.time() > deadline - 3.0:
            break
        
        adj = make_fn()
        current_edges = sum(bin(adj[u]).count('1') for u in range(n)) // 2
        
        # Greedy removal loop
        while time.time() < deadline - 3.0 and current_edges > best_m:
            found, edges = find_one_q3(adj)
            
            if not found:
                # Q3-free!
                if current_edges > best_m:
                    best_m = current_edges
                    best_adj = adj[:]
                    best_edges = get_edge_list(adj)
                break
            
            # Remove a random edge from the found Q3
            random.shuffle(edges)
            u, v = edges[0]
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            current_edges -= 1
    
    # Additional strategy: try adding edges to the best found so far
    if time.time() < deadline - 5.0:
        # Start from best and try to add edges
        adj = best_adj[:]
        current_edges = best_m
        
        # Generate candidate edges not in current graph
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    candidates.append((u, v))
        
        random.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() > deadline - 2.0:
                break
            if current_edges <= best_m:
                break
            
            # Try adding edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            current_edges += 1
            
            found, _ = find_one_q3(adj)
            if found:
                # Remove the edge we just added
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges -= 1
            else:
                # Edge addition successful
                best_m = current_edges
                best_adj = adj[:]
                best_edges = get_edge_list(adj)
    
    return best_edges
