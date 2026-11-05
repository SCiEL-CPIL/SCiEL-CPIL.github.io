import numpy as np
from fractions import Fraction
import itertools

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Generate all non-zero indices of the tensor
    # T[i*m+j, j*p+k, k*n+i] = 1
    non_zero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                non_zero_indices.append((a, b, c))
                
    total_rank = len(non_zero_indices) # 30
    
    # Build a graph where an edge exists between two indices if they can be 
    # covered by a single rank-1 tensor that vanishes on all other non-zero indices.
    # Condition: The box defined by the union of their coordinates contains no other non-zero indices.
    
    # For each pair, check compatibility
    # To make it faster, we can precompute a set of non-zero indices for O(1) lookup
    nz_set = set(non_zero_indices)
    
    edges = []
    for idx1 in range(total_rank):
        a1, b1, c1 = non_zero_indices[idx1]
        for idx2 in range(idx1 + 1, total_rank):
            a2, b2, c2 = non_zero_indices[idx2]
            
            su = {a1, a2}
            sv = {b1, b2}
            sw = {c1, c2}
            
            compatible = True
            # Check all combinations in the box
            for a in su:
                for b in sv:
                    for c in sw:
                        if (a, b, c) in nz_set:
                            if (a, b, c) != (a1, b1, c1) and (a, b, c) != (a2, b2, c2):
                                compatible = False
                                break
                if not compatible:
                    break
            if not compatible:
                continue
                
            if compatible:
                edges.append((idx1, idx2))
                
    # Find maximum matching in the graph
    # Use a simple greedy or bipartite matching? 
    # The graph is not necessarily bipartite.
    # However, for small N (30), we can use a simple backtracking or even just greedy if we want speed.
    # Let's use a simple greedy matching first, then try to improve.
    # Or use Hopcroft-Karp if we can decompose? No, general graph.
    # Given N=30, even O(N^3) or backtracking is fine.
    # Let's use a simple recursive backtracking for max matching.
    
    # Adjacency list
    adj = {i: [] for i in range(total_rank)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Matched status
    matched = [False] * total_rank
    match = [-1] * total_rank
    
    def dfs(u, visited):
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                if match[v] == -1 or dfs(match[v], visited):
                    match[v] = u
                    match[u] = v
                    return True
        return False
    
    max_matching = 0
    for u in range(total_rank):
        if match[u] == -1:
            visited = [False] * total_rank
            if dfs(u, visited):
                max_matching += 1
                
    # Rank is total_rank - max_matching
    # Each edge in matching saves 1 rank (2 nodes covered by 1 rank-1 tensor instead of 2)
    rank = total_rank - max_matching
    
    # If we found a reduction, we need to construct the factors.
    # This is tricky because we need the actual values.
    # For the schoolbook, it's 1s.
    # For the matched pairs, we need to find u, v, w.
    
    # If rank == total_rank, return schoolbook.
    if rank == total_rank:
        U = [[0] * total_rank for _ in range(n*m)]
        V = [[0] * total_rank for _ in range(m*p)]
        W = [[0] * total_rank for _ in range(p*n)]
        for col, (a, b, c) in enumerate(non_zero_indices):
            U[a][col] = 1
            V[b][col] = 1
            W[c][col] = 1
        return {"U": U, "V": V, "W": W}
        
    # Construct factors for the reduced rank
    # We need to assign a column index to each "group" (either a single index or a matched pair)
    # Let's identify the groups.
    groups = []
    used = [False] * total_rank
    for u in range(total_rank):
        if not used[u]:
            if match[u] != -1:
                v = match[u]
                if not used[v]:
                    groups.append((u, v))
                    used[u] = True
                    used[v] = True
            else:
                groups.append((u,))
                used[u] = True
                
    # Each group gets one column in U, V, W
    # Total columns = len(groups) = rank
    
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    
    for col, group in enumerate(groups):
        if len(group) == 1:
            a, b, c = non_zero_indices[group[0]]
            U[a][col] = 1
            V[b][col] = 1
            W[c][col] = 1
        else:
            idx1, idx2 = group
            a1, b1, c1 = non_zero_indices[idx1]
            a2, b2, c2 = non_zero_indices[idx2]
            
            # We need u, v, w such that:
            # u[a1] v[b1] w[c1] = 1
            # u[a2] v[b2] w[c2] = 1
            # And zero elsewhere (guaranteed by construction of the box)
            
            # Simple solution:
            # If a1 == a2:
            #   u[a1] = 1
            #   v[b1] w[c1] = 1
            #   v[b2] w[c2] = 1
            #   We can set v[b1]=1, w[c1]=1, v[b2]=1, w[c2]=1?
            #   Then u[a1]=1.
            #   Check if this creates non-zeros in the box.
            #   The box is {a1} x {b1,b2} x {c1,c2}.
            #   Indices: (a1,b1,c1)=1, (a1,b1,c2)=1, (a1,b2,c1)=1, (a1,b2,c2)=1.
            #   We need (a1,b1,c2) and (a1,b2,c1) to be 0.
            #   But with u[a1]=1, v[b1]=1, v[b2]=1, w[c1]=1, w[c2]=1, they are 1.
            #   So we need to adjust.
            
            # We need to solve:
            # u1 v1 w1 = 1
            # u2 v2 w2 = 1
            # u1 v1 w2 = 0 (if (a1,b1,c2) is in box and not target)
            # u1 v2 w1 = 0 (if (a1,b2,c1) is in box and not target)
            # u2 v1 w1 = 0 (if (a2,b1,c1) is in box and not target)
            # u2 v2 w2 = 1 (target)
            # u2 v1 w2 = 0 (if (a2,b1,c2) is in box and not target)
            # u2 v2 w1 = 0 (if (a2,b2,c1) is in box and not target)
            
            # This requires careful selection.
            # Given the complexity, and the risk of incorrect factor construction,
            # it might be safer to just return the schoolbook if we can't easily construct the factors.
            # However, the score improves with lower rank.
            # Let's try a specific simple assignment:
            # Set u[a1]=1, u[a2]=1 (if a1!=a2, else 1)
            # Then v[b1] w[c1] = 1, v[b2] w[c2] = 1.
            # And we need to zero out the cross terms.
            # If we set v[b1]=1, w[c1]=1, v[b2]=1, w[c2]=1, cross terms are 1.
            # To zero them, we can use signs? No, must be exact.
            # Maybe we can't always do it with 1s.
            
            # Let's stick to schoolbook if factor construction is non-trivial.
            # The matching logic is a good start, but implementation of factors is the bottleneck.
            pass
            
    # Fallback to schoolbook for now to ensure correctness
    U = [[0] * total_rank for _ in range(n*m)]
    V = [[0] * total_rank for _ in range(m*p)]
    W = [[0] * total_rank for _ in range(p*n)]
    for col, (a, b, c) in enumerate(non_zero_indices):
        U[a][col] = 1
        V[b][col] = 1
        W[c][col] = 1
    return {"U": U, "V": V, "W": W}
