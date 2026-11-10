import itertools
import random
import time

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Precompute all 8-subsets
    all_subsets = list(itertools.combinations(range(n), 8))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute Q3 edges for isomorphism check
    q3_edges = set()
    for u in range(8):
        for v in range(u+1, 8):
            if bin(u ^ v).count('1') == 1:
                q3_edges.add((u, v))
                
    # Precompute subsets containing each pair (u, v)
    pairs_to_subsets = {}
    for i in range(n):
        for j in range(i+1, n):
            pairs_to_subsets[(i, j)] = []
            
    for subset in all_subsets:
        for idx in range(8):
            for idx2 in range(idx+1, 8):
                u, v = subset[idx], subset[idx2]
                if u > v:
                    u, v = v, u
                pairs_to_subsets[(u, v)].append(subset)

    # Initialize edge counts for subsets to 0
    subset_edge_counts = [0] * len(all_subsets)

    def is_q3(vertices, adj, count):
        """
        Check if the induced subgraph on 'vertices' is isomorphic to Q3.
        vertices: tuple of 8 unique vertex labels
        adj: 16x16 boolean matrix
        count: current edge count of this subset
        """
        if count != 12:
            return False
            
        # Check 3-regularity
        for i in range(8):
            deg = 0
            for j in range(8):
                if i != j and adj[vertices[i]][vertices[j]]:
                    deg += 1
            if deg != 3:
                return False
                
        # Backtracking isomorphism
        def bt(q3_v, mapping, used_subset_indices):
            if q3_v == 8:
                return True
            
            for s_idx in range(8):
                if s_idx in used_subset_indices:
                    continue
                
                ok = True
                for prev_q3_v in range(q3_v):
                    if (prev_q3_v, q3_v) in q3_edges:
                        prev_s_idx = mapping[prev_q3_v]
                        if not adj[vertices[prev_s_idx]][vertices[s_idx]]:
                            ok = False
                            break
                    else:
                        prev_s_idx = mapping[prev_q3_v]
                        if adj[vertices[prev_s_idx]][vertices[s_idx]]:
                            ok = False
                            break
                            
                if ok:
                    mapping[q3_v] = s_idx
                    if bt(q3_v + 1, mapping, used_subset_indices | {s_idx}):
                        return True
                    del mapping[q3_v]
            return False

        for start_idx in range(8):
            mapping = {0: start_idx}
            if bt(1, mapping, {start_idx}):
                return True
        return False

    # Initialize empty graph
    adj = [[False]*n for _ in range(n)]
    
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.seed(42)
    random.shuffle(all_edges)
    
    # Greedy addition
    for u, v in all_edges:
        if time.time() > deadline:
            break
            
        # Tentatively add edge
        adj[u][v] = True
        adj[v][u] = True
        
        # Update edge counts for affected subsets
        affected_subsets = pairs_to_subsets[(u, v)]
        for subset in affected_subsets:
            idx = subset_to_idx[subset]
            subset_edge_counts[idx] += 1
            
        # Check for Q3
        valid = True
        for subset in affected_subsets:
            idx = subset_to_idx[subset]
            if is_q3(subset, adj, subset_edge_counts[idx]):
                valid = False
                break
                
        if not valid:
            # Revert
            adj[u][v] = False
            adj[v][u] = False
            for subset in affected_subsets:
                idx = subset_to_idx[subset]
                subset_edge_counts[idx] -= 1
            
    # Count edges
    res = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                res.append([i, j])
                
    return res
