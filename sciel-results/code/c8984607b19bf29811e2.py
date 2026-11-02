"""Rewritten program using MIS with Tomita BB-MCS to find a large kissing configuration in R^11."""
import itertools
import numpy as np

D = 11

# EVOLVE-BLOCK-START
def construct():
    # 1. Generate Candidate Vectors
    # Weight 1: +/- e_i
    # Weight 2: +/- e_i +/- e_j
    candidates = []
    indices = range(D)
    
    # Weight 1
    for i in indices:
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            candidates.append(tuple(v))
            
    # Weight 2
    for i, j in itertools.combinations(indices, 2):
        for s1 in (1, -1):
            for s2 in (1, -1):
                v = [0] * D
                v[i] = s1
                v[j] = s2
                candidates.append(tuple(v))
                
    # Ensure uniqueness
    candidates = list(set(candidates))
    n = len(candidates)
    
    # 2. Build Compatibility Graph
    # Compat if dist_sq >= max(norm_sq_u, norm_sq_v)
    norms_sq = np.array([sum(x*x for x in v) for v in candidates])
    V = np.array(candidates, dtype=np.int64)
    
    dots = V @ V.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    max_norm_sq = np.maximum(norms_sq[:, None], norms_sq[None, :])
    
    compat = dist_sq >= max_norm_sq
    np.fill_diagonal(compat, False)
    
    # Adjacency lists for Compatibility Graph
    adj_compat = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if compat[i, j]:
                adj_compat[i].add(j)
                adj_compat[j].add(i)
                
    # 3. Tomita BB-MCS Algorithm for Maximum Clique (in Compatibility Graph)
    
    def greedy_coloring(vertices):
        """
        Returns a list `colors` where `colors[i]` is the maximum color used in the subgraph induced by `vertices[:i+1]`.
        """
        if not vertices:
            return []
            
        n_v = len(vertices)
        color_of = [-1] * n_v
        running_max_colors = [0] * n_v
        
        for i in range(n_v):
            v = vertices[i]
            used_colors = set()
            for j in range(i):
                u = vertices[j]
                if u in adj_compat[v]:
                    if color_of[j] != -1:
                        used_colors.add(color_of[j])
                        
            c = 1
            while c in used_colors:
                c += 1
            color_of[i] = c
            
            if i == 0:
                running_max_colors[i] = c
            else:
                running_max_colors[i] = max(running_max_colors[i-1], c)
                
        return running_max_colors

    def mcs_ordering(n, adj):
        """
        Returns an ordering of vertices for Tomita's algorithm.
        """
        order = []
        picked = [False] * n
        start = 0
        picked[start] = True
        order.append(start)
        
        deg_picked = [0] * n
        for u in adj[start]:
            deg_picked[u] += 1
            
        for _ in range(n - 1):
            max_deg = -1
            best_v = -1
            for v in range(n):
                if not picked[v]:
                    if deg_picked[v] > max_deg:
                        max_deg = deg_picked[v]
                        best_v = v
                        
            if best_v == -1:
                break
                
            picked[best_v] = True
            order.append(best_v)
            
            for u in adj[best_v]:
                if not picked[u]:
                    deg_picked[u] += 1
                    
        return order

    order = mcs_ordering(n, adj_compat)
    P = order[::-1]
    colors = greedy_coloring(P)
    
    best_clique_size = [0]
    best_clique = []
    
    def expand(R, P, colors):
        nonlocal best_clique_size, best_clique
        
        if not P:
            if len(R) > best_clique_size[0]:
                best_clique_size[0] = len(R)
                best_clique = R[:]
            return
        
        if len(R) + colors[-1] <= best_clique_size[0]:
            return
            
        for i in range(len(P) - 1, -1, -1):
            if len(R) + colors[i] <= best_clique_size[0]:
                break
                
            v = P[i]
            
            new_P = [u for u in P[:i] if u in adj_compat[v]]
            
            if new_P:
                new_colors = greedy_coloring(new_P)
                expand(R + [v], new_P, new_colors)
            else:
                if len(R) + 1 > best_clique_size[0]:
                    best_clique_size[0] = len(R) + 1
                    best_clique = R + [v]
            
    expand([], P, colors)
    
    # 4. Return the vectors corresponding to best_clique
    result_vectors = [candidates[v] for v in best_clique]
    
    return np.array(result_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
