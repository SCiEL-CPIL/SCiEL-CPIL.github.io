import time
from itertools import combinations
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Precompute modular inverses
    inv = {x: pow(x, -1, 101) for x in range(1, 11)}
    
    # Precompute all non-empty subsets with their residues and sizes
    # subset_list[i] = (residue, size, terms)
    subset_list = []
    for mask in range(1, 1 << 10):
        if time.time() > deadline - 10:
            break
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        residue = sum(inv[x] for x in terms) % 101
        subset_list.append((residue, len(terms), terms))
    
    # Group subsets by size
    # subsets_by_size[s] = list of (residue, terms)
    subsets_by_size = [[] for _ in range(11)]
    for residue, size, terms in subset_list:
        subsets_by_size[size].append((residue, terms))
    
    # Hopcroft-Karp algorithm for bipartite matching
    # Left side: 101 residues (0..100)
    # Right side: subsets (indexed 0..len(subset_list)-1)
    # Edge exists if subset's residue == left node's residue
    
    def hopcroft_karp(max_k):
        """
        Find maximum matching using only subsets of size <= max_k.
        Returns the matching or None if perfect matching not found.
        """
        # Build adjacency: for each residue, which subsets (of size <= max_k) map to it
        # Only include subsets of size <= max_k
        adj = [[] for _ in range(101)]
        for s in range(1, max_k + 1):
            for residue, terms in subsets_by_size[s]:
                adj[residue].append(id(terms))  # use id as unique identifier
                # Actually, we need to track which subset index
                # Let me restructure: assign each subset an index
        
        # Better approach: assign indices to subsets
        # subset_index: maps (size, residue, terms_tuple) -> index
        # Let's rebuild with indices
        pass
    
    # Let me restructure with proper indices
    # subset_indices: list of (residue, size, terms)
    # For each subset, assign a unique index
    
    # Rebuild: all_subsets = list of (residue, size, terms)
    # We already have subset_list as (residue, size, terms)
    
    # For Hopcroft-Karp, we need:
    # Left nodes: 0..100 (residues)
    # Right nodes: 0..N-1 (subsets)
    # Edge (r, s) exists if subset s has residue r and size <= max_k
    
    def find_perfect_matching(max_k):
        """
        Use Hopcroft-Karp to find perfect matching between 101 residues and subsets.
        Only use subsets of size <= max_k.
        Returns assignment dict: residue -> terms, or None if no perfect matching.
        """
        N = len(subset_list)
        
        # Build adjacency list for left nodes
        # adj[r] = list of subset indices j where subset_list[j][0] == r and subset_list[j][1] <= max_k
        adj = [[] for _ in range(101)]
        for j in range(N):
            residue, size, terms = subset_list[j]
            if size <= max_k:
                adj[residue].append(j)
        
        # Hopcroft-Karp
        # pair_u[r] = matched right node for left node r, or -1
        # pair_v[j] = matched left node for right node j, or -1
        pair_u = [-1] * 101
        pair_v = [-1] * N
        dist = [0] * 101
        
        def bfs():
            """Find augmenting paths. Returns True if augmenting path exists."""
            queue = deque()
            for r in range(101):
                if pair_u[r] == -1:
                    dist[r] = 0
                    queue.append(r)
                else:
                    dist[r] = float('inf')
            
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    # v is a right node (subset index)
                    if pair_v[v] == -1:
                        found = True
                    elif dist[pair_v[v]] == float('inf'):
                        dist[pair_v[v]] = dist[u] + 1
                        queue.append(pair_v[v])
            return found
        
        def dfs(u):
            """Try to find augmenting path from left node u."""
            for v in adj[u]:
                if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = float('inf')
            return False
        
        matching_size = 0
        while bfs():
            for r in range(101):
                if pair_u[r] == -1:
                    if dfs(r):
                        matching_size += 1
                        if time.time() > deadline - 5:
                            return None
        
        if matching_size == 101:
            # Perfect matching found
            assignment = {}
            for r in range(101):
                v = pair_u[r]
                assignment[r] = subset_list[v][2]
            return assignment
        return None
    
    # Try increasing max_k until we find a perfect matching
    best_assignment = None
    for max_k in range(1, 11):
        if time.time() > deadline - 5:
            break
        assignment = find_perfect_matching(max_k)
        if assignment is not None:
            best_assignment = assignment
            break
    
    if best_assignment is not None:
        result = []
        for r in range(101):
            result.append(best_assignment[r])
        return result
    
    # Fallback: greedy approach - for each residue, pick smallest size subset
    residues_by_size = [dict() for _ in range(11)]
    for residue, size, terms in subset_list:
        if residue not in residues_by_size[size]:
            residues_by_size[size][residue] = terms
    
    result = []
    for r in range(101):
        found = None
        for s in range(1, 11):
            if r in residues_by_size[s]:
                found = residues_by_size[s][r]
                break
        if found is None:
            found = list(range(1, 11))
        result.append(found)
    
    return result
