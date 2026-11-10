import time
from collections import deque

def construct():
    deadline = time.time() + 85

    # Precompute modular inverses
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)

    # Precompute all subsets and their residues
    subsets = []
    subset_residues = [0] * (1 << 10)
    subset_sizes = [0] * (1 << 10)
    
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        res = sum(inv[x] for x in terms) % 101
        subsets.append(terms)
        subset_residues[mask - 1] = res
        subset_sizes[mask - 1] = len(terms)
    
    # For each residue, list all subsets that produce it, sorted by size
    residue_to_subsets = [[] for _ in range(101)]
    for i in range(1023):
        residue_to_subsets[subset_residues[i]].append(i)
    
    # Sort each residue's subsets by size (then by mask for determinism)
    for r in range(101):
        residue_to_subsets[r].sort(key=lambda i: (subset_sizes[i], i))
    
    # Hopcroft-Karp: find minimum k such that perfect matching exists
    # using only subsets of size <= k
    
    # We'll binary search on k, or just iterate k from 1 to 10
    # For each k, build the bipartite graph and run HK
    
    def hopcroft_karp(max_k):
        # Left: 101 residues (0..100)
        # Right: subsets with size <= max_k
        # Edge: residue r connected to subset i if subset_residues[i] == r and subset_sizes[i] <= max_k
        
        # Build adjacency from left to right
        # left_u -> list of right_v
        adj = [[] for _ in range(101)]
        for r in range(101):
            for i in residue_to_subsets[r]:
                if subset_sizes[i] <= max_k:
                    adj[r].append(i)
        
        # HK algorithm
        # pairU[u] = matched right vertex for left u, or -1
        # pairV[v] = matched left vertex for right v, or -1
        # We need to map right vertices to indices. Right vertices are subset indices 0..1022
        # but only those with size <= max_k are eligible.
        
        # For simplicity, use the subset index as the right vertex ID
        # pairV is an array of size 1023
        pairU = [-1] * 101
        pairV = [-1] * 1023
        dist = [0] * 101
        
        def bfs():
            # Build BFS layers from free left nodes
            q = deque()
            found = False
            for u in range(101):
                if pairU[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            while q:
                u = q.popleft()
                for v in adj[u]:
                    pu = pairV[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
            return found
        
        def dfs(u):
            for v in adj[u]:
                pu = pairV[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pairU[u] = v
                    pairV[v] = u
                    return True
            dist[u] = -1
            return False
        
        matching = 0
        while bfs():
            for u in range(101):
                if pairU[u] == -1:
                    if dfs(u):
                        matching += 1
            if time.time() > deadline - 3:
                return matching, pairU
        
        return matching, pairU
    
    # Try k from 1 to 10
    best_pairU = None
    for k in range(1, 11):
        matching, pairU = hopcroft_karp(k)
        if matching == 101:
            best_pairU = pairU
            break
    
    if best_pairU is not None:
        result = []
        for r in range(101):
            v = best_pairU[r]
            if v >= 0:
                result.append(subsets[v])
            else:
                result.append([1])
        return result
    
    # Fallback: greedy assignment
    # For each residue, pick the smallest subset
    used = [False] * 1023
    result = []
    for r in range(101):
        assigned = False
        for i in residue_to_subsets[r]:
            if not used[i]:
                used[i] = True
                result.append(subsets[i])
                assigned = True
                break
        if not assigned:
            result.append([1])
    return result
