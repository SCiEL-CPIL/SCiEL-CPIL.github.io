import time
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 101 nonempty lists, row r containing distinct denominators in 1..10
    whose modular inverses sum to r modulo 101. Minimize the largest row length.
    """
    P = 101
    deadline = time.time() + 50  # safety margin before 90s limit
    
    # Precompute inverses
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, P)
    
    # Precompute all subsets grouped by size using efficient enumeration
    subsets_by_size = {s: [] for s in range(1, 11)}
    
    for mask in range(1, 1 << 10):
        m = mask
        sz = 0
        s = 0
        while m:
            lsb = m & (-m)
            x = lsb.bit_length()
            sz += 1
            s += inv[x]
            m ^= lsb
        r = s % P
        subset = tuple(x for x in range(1, 11) if mask & (1 << (x-1)))
        subsets_by_size[sz].append((r, subset))
    
    def hopcroft_karp(adj, n_left, n_right):
        """
        Optimized Hopcroft-Karp algorithm for maximum bipartite matching.
        adj: list of lists, adj[u] = list of right nodes adjacent to left node u
        Returns match_left array or None if matching doesn't cover all left nodes.
        """
        match_left = [-1] * n_left
        match_right = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            queue = deque()
            for u in range(n_left):
                if match_left[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = -1
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    mu = match_right[v]
                    if mu == -1:
                        found = True
                    elif dist[mu] == -1:
                        dist[mu] = dist[u] + 1
                        queue.append(mu)
            return found
        
        def dfs(u):
            for v in adj[u]:
                mu = match_right[v]
                if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                    match_left[u] = v
                    match_right[v] = u
                    return True
            dist[u] = -1
            return False
        
        matching = 0
        while bfs():
            for u in range(n_left):
                if match_left[u] == -1:
                    if dfs(u):
                        matching += 1
            if matching == n_left:
                break
        
        if matching < n_left:
            return None
        
        return match_left
    
    def can_cover(max_size):
        """Check if subsets of size <= max_size can cover all 101 residues."""
        # Collect all subsets with size <= max_size
        all_subsets = []
        for sz in range(1, max_size + 1):
            for (r, subset) in subsets_by_size[sz]:
                all_subsets.append((r, subset))
        
        # For each residue, list of subset indices
        residue_to_subsets = [[] for _ in range(P)]
        for idx, (r, subset) in enumerate(all_subsets):
            residue_to_subsets[r].append(idx)
        
        # Check if every residue has at least one subset
        for r in range(P):
            if not residue_to_subsets[r]:
                return None
        
        n_left = P
        n_right = len(all_subsets)
        
        match_left = hopcroft_karp(residue_to_subsets, n_left, n_right)
        if match_left is None:
            return None
        
        # Build result
        result = [None] * P
        for r in range(P):
            idx = match_left[r]
            result[r] = list(all_subsets[idx][1])
        return result
    
    # Try increasing max_size
    for max_size in range(1, 11):
        if time.time() > deadline:
            break
        result = can_cover(max_size)
        if result is not None:
            return result
    
    # Fallback: use the original approach (minimize individual sizes, not max)
    best = [None] * P
    for mask in range(1, 1 << 10):
        m = mask
        sz = 0
        s = 0
        while m:
            lsb = m & (-m)
            x = lsb.bit_length()
            sz += 1
            s += inv[x]
            m ^= lsb
        r = s % P
        subset = tuple(x for x in range(1, 11) if mask & (1 << (x-1)))
        if best[r] is None or sz < best[r][0]:
            best[r] = (sz, subset)
    
    return [list(best[r][1]) for r in range(P)]
# EVOLVE-BLOCK-END
