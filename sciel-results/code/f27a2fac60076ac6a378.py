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
    # We don't need explicit edges for the check, just the property:
    # An 8-vertex graph is Q3 iff it is 3-regular and bipartite.
    
    # Precompute all 8-vertex subsets
    all_8_subsets = list(itertools.combinations(range(n), 8))
    
    # Precompute subsets for each edge to speed up incremental check
    # subsets_for_edge[(u, v)] = list of 8-subsets containing both u and v
    subsets_for_edge = {}
    for S in all_8_subsets:
        S_set = set(S)
        # Generate all pairs in S
        S_list = list(S)
        for i in range(8):
            for j in range(i + 1, 8):
                u, v = S_list[i], S_list[j]
                key = (min(u, v), max(u, v))
                if key not in subsets_for_edge:
                    subsets_for_edge[key] = []
                subsets_for_edge[key].append(S)

    def is_q3_induced(adj_mask, S):
        """
        Check if the induced subgraph on S is isomorphic to Q3.
        S is a tuple of 8 vertices.
        adj_mask is a list of 16 ints.
        Q3 on 8 vertices is 3-regular and bipartite.
        """
        # 1. Check 3-regularity within S
        # Create a mapping from S vertices to indices 0..7 for local adj matrix
        # Or just use bitmasks restricted to S
        
        # Local adjacency matrix as bitmasks for the 8 vertices in S
        # Let local[i] be the bitmask of neighbors of S[i] within S
        
        # Faster: just count degrees
        degs = [0] * 8
        local_adj = [0] * 8 # bitmask of neighbors within S for each S[i]
        
        for i in range(8):
            v = S[i]
            mask = adj_mask[v]
            # Count neighbors in S
            deg = 0
            l_adj = 0
            for j in range(8):
                if i == j: continue
                u = S[j]
                if (mask >> u) & 1:
                    deg += 1
                    l_adj |= (1 << j)
            if deg != 3:
                return False
            degs[i] = deg
            local_adj[i] = l_adj
            
        # 2. Check Bipartite
        # BFS on the local graph
        color = [-1] * 8
        color[0] = 0
        queue = deque([0])
        
        while queue:
            curr = queue.popleft()
            mask = local_adj[curr]
            for nb in range(8):
                if (mask >> nb) & 1:
                    if color[nb] == -1:
                        color[nb] = 1 - color[curr]
                        queue.append(nb)
                    elif color[nb] == color[curr]:
                        return False
        return True

    def try_add_edge(adj_mask, u, v):
        """
        Try to add edge (u, v) to adj_mask.
        Returns True if added, False if it creates a Q3.
        Modifies adj_mask in place if added.
        """
        # Tentatively add
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        key = (min(u, v), max(u, v))
        subsets = subsets_for_edge.get(key, [])
        
        for S in subsets:
            if is_q3_induced(adj_mask, S):
                # Revert
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                return False
        
        return True

    def get_edge_list(adj_mask):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def count_edges(adj_mask):
        return sum(bin(adj_mask[u]).count('1') for u in range(n)) // 2

    best_m = 0
    best_edges = []
    
    # List of all edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Run multiple random greedy constructions
    # Estimate time per run:
    # 120 edges * ~3000 subsets * ~100 ops = 36M ops.
    # Python is slow, maybe 1-2s per run.
    # We have 110s. We can do ~50-100 runs.
    
    runs = 0
    while time.time() < deadline - 2.0:
        runs += 1
        rng = random.Random(runs)
        adj = [0] * n
        
        # Shuffle edges
        shuffled_edges = all_edges[:]
        rng.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            try_add_edge(adj, u, v)
            
        m = count_edges(adj)
        if m > best_m:
            best_m = m
            best_edges = get_edge_list(adj)
            
        # Optional: If we find a very good one early, we might try to polish?
        # But let's just keep running to find even better.
        
    return best_edges
