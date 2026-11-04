import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    time_limit = 100  # leave buffer for overhead
    
    # Baseline: 220 support-size-2 vectors
    candidates2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates2.append(v)
    candidates2 = np.array(candidates2, dtype=np.int64)
    baseline = len(candidates2)  # 220

    # New Strategy: 330 support-size-4 vectors (0/1)
    # Vectors are 0/1 with exactly 4 ones.
    # Norm squared = 4. Norm = 2.
    # Condition: min pairwise dist >= max norm = 2.
    # dist^2 >= 4.
    # ||v_i - v_j||^2 = 4 + 4 - 2 dot(v_i, v_j) = 8 - 2 dot(v_i, v_j)
    # 8 - 2 dot >= 4 => 4 >= 2 dot => dot <= 2.
    # For 0/1 vectors, dot is intersection size.
    # So we need a family of 4-subsets of {1..11} such that any two intersect in <= 2 elements.
    # This is the Maximum Independent Set in the graph where edges connect sets with intersection >= 3.
    # Or Maximum Clique in the graph where edges connect sets with intersection <= 2.
    
    supports = list(itertools.combinations(range(D), 4))
    n4 = len(supports) # 330
    
    # Build adjacency matrix for "compatible" graph (intersection <= 2)
    # We will find Max Clique in this graph.
    # adj[i][j] = True if supports[i] and supports[j] intersect in <= 2 elements.
    
    # Precompute intersection sizes or just use sets
    # 330 nodes is small, O(N^2) is fine.
    adj = [[False] * n4 for _ in range(n4)]
    for i in range(n4):
        adj[i][i] = False
        s_i = set(supports[i])
        for j in range(i + 1, n4):
            s_j = set(supports[j])
            inter_size = len(s_i & s_j)
            if inter_size <= 2:
                adj[i][j] = True
                adj[j][i] = True

    # Implement Tomita's Maximum Clique Algorithm
    # We need to find the largest clique in the graph defined by 'adj'.
    
    # Convert to adjacency list for efficiency in search
    adj_list = [set() for _ in range(n4)]
    for i in range(n4):
        for j in range(n4):
            if adj[i][j]:
                adj_list[i].add(j)
                adj_list[j].add(i)
                
    # Remap vertices by degree (descending) to improve pruning
    degrees = [len(adj_list[i]) for i in range(n4)]
    order = sorted(range(n4), key=lambda x: -degrees[x])
    remap = {old: new for new, old in enumerate(order)}
    n_remapped = n4
    adj_remapped = [set() for _ in range(n_remapped)]
    for i in range(n4):
        for j in adj_list[i]:
            adj_remapped[remap[i]].add(remap[j])
            
    # Tomita's Algorithm with Coloring
    # We will use a recursive search with greedy coloring upper bound.
    
    best_clique = []
    best_size = 0
    
    # Coloring function:
    # Given a set of vertices P, assign colors such that adjacent vertices have different colors.
    # The number of colors is an upper bound on the clique size in P.
    # We return a list of (vertex, color) sorted by color.
    
    def color_graph(P):
        # P is a set of vertices
        # Greedy coloring
        colors = {}
        color_sets = [] # List of sets of vertices with that color
        for v in P:
            # Find smallest color not used by neighbors in P
            used_colors = set()
            for u in adj_remapped[v]:
                if u in P:
                    if u in colors:
                        used_colors.add(colors[u])
            # Find smallest positive integer not in used_colors
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
            if c > len(color_sets):
                color_sets.append(set())
            color_sets[c-1].add(v)
        
        # Return list of (v, c) sorted by c
        result = []
        for c, vset in enumerate(color_sets, 1):
            for v in vset:
                result.append((v, c))
        return result

    def expand(R, P):
        nonlocal best_size, best_clique
        if time.time() - start_time > time_limit:
            return
        
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        if len(R) + len(P) <= best_size:
            return
            
        # Get coloring of P
        colored_P = color_graph(P)
        
        # Iterate in reverse order of colors (highest color first)
        # This is a heuristic to find large cliques faster
        for v, c in reversed(colored_P):
            if len(R) + c <= best_size:
                return # Prune
            
            # Branch: Include v
            R.append(v)
            # New P is intersection of P and neighbors of v
            P_new = P.intersection(adj_remapped[v])
            expand(R, P_new)
            R.pop()
            
            # Remove v from P
            P.discard(v)
            
    # Initialize
    R = []
    P = set(range(n_remapped))
    expand(R, P)
    
    # Map back to original indices
    if best_size > 0:
        original_indices = [order[i] for i in best_clique]
        # Construct vectors
        result_vecs = []
        for idx in original_indices:
            v = [0] * D
            for pos in supports[idx]:
                v[pos] = 1
            result_vecs.append(v)
        result = np.array(result_vecs, dtype=np.int64)
        
        if len(result) > baseline:
            return result
        else:
            return candidates2
    else:
        return candidates2
