"""Rewritten program using bipartite matching (Hopcroft-Karp) with binary search on max row length."""
# EVOLVE-BLOCK-START
from collections import deque

def construct():
    """
    Returns 101 nonempty lists. Row r contains distinct denominators in 1..10
    whose modular inverses sum to r modulo 101.
    Minimizes the largest row length using a bipartite matching approach where
    each subset is used at most once.
    """
    p = 101
    n_residues = 101
    all_denoms = list(range(1, 11))
    
    # Precompute modular inverses
    inv = [pow(x, -1, p) for x in all_denoms]
    
    # Generate all non-empty subsets of {1..10}
    # Store: (residue, size, terms)
    subsets = []
    for mask in range(1, 1 << 10):
        size = bin(mask).count('1')
        terms = [all_denoms[i] for i in range(10) if mask & (1 << i)]
        residue = sum(inv[i] for i in range(10) if mask & (1 << i)) % p
        subsets.append((residue, size, terms))
        
    def hopcroft_karp(max_k):
        # Build adjacency list for subsets with size <= max_k
        # Left side: 0..100 (residues)
        # Right side: indices of valid subsets
        right_node_indices = []
        for idx, (res, sz, terms) in enumerate(subsets):
            if sz <= max_k:
                right_node_indices.append(idx)
                
        n_right = len(right_node_indices)
        if n_right < n_residues:
            return None
            
        # adj[u] = list of v indices (indices into right_node_indices)
        adj = [[] for _ in range(n_residues)]
        for v_idx, s_idx in enumerate(right_node_indices):
            res = subsets[s_idx][0]
            adj[res].append(v_idx)
            
        n_left = n_residues
        
        # Hopcroft-Karp Implementation
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            queue = deque()
            found = False
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = -1
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    u2 = pair_v[v]
                    if u2 != -1 and dist[u2] == -1:
                        dist[u2] = dist[u] + 1
                        queue.append(u2)
                    elif u2 == -1:
                        found = True
            return found
            
        def dfs(u):
            for v in adj[u]:
                u2 = pair_v[v]
                if u2 == -1 or (dist[u2] == dist[u] + 1 and dfs(u2)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False
            
        matching = 0
        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    if dfs(u):
                        matching += 1
                        
        if matching == n_left:
            result = [None] * n_residues
            for u in range(n_left):
                v = pair_u[u]
                s_idx = right_node_indices[v]
                result[u] = subsets[s_idx][2]
            return result
        else:
            return None
            
    # Binary search for minimum K (linear search is fast enough for 1..10)
    for k in range(1, 11):
        res = hopcroft_karp(k)
        if res is not None:
            return res
            
    # Fallback (should not happen)
    return [list(range(1, 11)) for _ in range(101)]
# EVOLVE-BLOCK-END
