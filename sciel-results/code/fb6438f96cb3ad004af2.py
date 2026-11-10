import time
import itertools
import random
import math

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
    
    # All 8-vertex subsets of 0..15
    all_8_subsets = list(itertools.combinations(range(n), 8))
    
    def find_one_q3_fast(adj_mask):
        """
        Fast Q3 detection using bitmask operations.
        Returns (True, edge_list) if a Q3 is found, (False, None) otherwise.
        adj_mask: list of 16 integers, each a bitmask of neighbors.
        """
        # Iterate over all 8-element subsets
        for S in all_8_subsets:
            # Quick filter: each vertex must have degree >= 3 within S
            ok = True
            for v in S:
                mask = adj_mask[v]
                deg = 0
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
                    return False  # Abort if too slow
                
                if idx == 8:
                    return True
                
                # Neighbors of idx in Q3 that are already assigned (index < idx)
                assigned_nbs = []
                for nb in range(8):
                    if nb < idx and (q3_adj_mask[idx] >> nb) & 1:
                        assigned_nbs.append(nb)
                
                # Try each unused vertex in S
                for v in S:
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
    
    def _count_edges(adj):
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2
    
    def _make_complete_bipartite(a, b):
        adj = [0] * n
        # Part A: vertices 0..a-1, Part B: vertices a..a+b-1
        for u in range(a):
            for v in range(a, a + b):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    def _make_random_graph(p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj
    
    # Simulated Annealing for Q3-free graph optimization
    def simulated_annealing(adj_initial, time_limit=10.0):
        """
        Simulated annealing to maximize edges in a Q3-free graph.
        State: adjacency mask (list of 16 ints)
        Move: add or remove a random edge
        Energy: -num_edges (we want to minimize energy = maximize edges)
        Constraint: graph must be Q3-free
        """
        adj = adj_initial[:]
        current_edges = _count_edges(adj)
        
        # Check initial state
        found, _ = find_one_q3_fast(adj)
        if found:
            # Remove Q3s until Q3-free
            while found and time.time() < deadline - 2.0:
                found, edges = find_one_q3_fast(adj)
                if not found:
                    break
                random.shuffle(edges)
                u, v = edges[0]
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges -= 1
        
        best_adj = adj[:]
        best_edges = current_edges
        
        T = 1.0  # initial temperature
        T_min = 0.01
        cooling_rate = 0.995
        
        rng = random.Random(12345)
        
        while time.time() < deadline - 2.0 and T > T_min:
            # Generate a random move: add or remove an edge
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            has_edge = (adj[u] >> v) & 1
            
            if has_edge:
                # Remove edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                delta_energy = 1  # losing an edge increases energy
                new_edges = current_edges - 1
            else:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                delta_energy = -1  # gaining an edge decreases energy
                new_edges = current_edges + 1
            
            # Check if new state is Q3-free
            found, _ = find_one_q3_fast(adj)
            
            if found:
                # Invalid state: revert
                if has_edge:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                # Reject
                pass
            else:
                # Valid state
                if delta_energy < 0:
                    # Better state: always accept
                    current_edges = new_edges
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                else:
                    # Worse state: accept with probability exp(-delta_energy / T)
                    prob = math.exp(-delta_energy / T)
                    if rng.random() < prob:
                        current_edges = new_edges
            
            T *= cooling_rate
        
        return best_adj, best_edges
    
    # Multiple strategies
    best_m = 0
    best_adj = [0] * n
    best_edges = []
    
    # Strategy 1: Start from K_{3,13} (known Q3-free)
    adj = _make_complete_bipartite(3, 13)
    m = _count_edges(adj)
    found, _ = find_one_q3_fast(adj)
    if not found and m > best_m:
        best_m = m
        best_adj = adj[:]
        best_edges = get_edge_list(adj)
    
    # Strategy 2: Simulated annealing from random starts
    for p in [0.75, 0.80, 0.85, 0.90]:
        if time.time() > deadline - 5.0:
            break
        
        adj_init = _make_random_graph(p, int(p * 100))
        sa_adj, sa_m = simulated_annealing(adj_init, time_limit=8.0)
        
        if sa_m > best_m:
            found, _ = find_one_q3_fast(sa_adj)
            if not found:
                best_m = sa_m
                best_adj = sa_adj[:]
                best_edges = get_edge_list(sa_adj)
    
    # Strategy 3: Greedy edge addition from best
    if time.time() < deadline - 5.0 and best_m > 0:
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
            
            # Try adding edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            current_edges += 1
            
            found, _ = find_one_q3_fast(adj)
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
    
    # Strategy 4: Final simulated annealing polish
    if time.time() < deadline - 3.0 and best_m > 0:
        sa_adj, sa_m = simulated_annealing(best_adj[:], time_limit=5.0)
        if sa_m > best_m:
            found, _ = find_one_q3_fast(sa_adj)
            if not found:
                best_m = sa_m
                best_adj = sa_adj[:]
                best_edges = get_edge_list(sa_adj)
    
    return best_edges
