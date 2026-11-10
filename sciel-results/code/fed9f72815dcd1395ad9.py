import random
import time

def construct():
    d = 7
    n = 1 << d
    # Precompute all edges
    edges = []
    edge_to_idx = {}
    idx_to_edge = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v, i))
                edge_to_idx[(u, v)] = idx
                idx_to_edge[idx] = (u, v)
    
    # Precompute all C4s
    # A C4 is defined by (u, i, j) where u has 0 in i, j
    c4s = []
    for u in range(n):
        for i in range(d):
            for j in range(i+1, d):
                if (u & (1 << i)) == 0 and (u & (1 << j)) == 0:
                    v1 = u ^ (1 << i)
                    v2 = u ^ (1 << j)
                    v3 = u ^ (1 << i) ^ (1 << j)
                    e1 = (min(u, v1), max(u, v1))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v3, v2), max(v3, v2))
                    e4 = (min(v2, u), max(v2, u))
                    c4s.append([edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4]])
    
    # Map edge index to list of C4 indices containing it
    edge_c4s = [[] for _ in range(len(edges))]
    for c4_idx, c4 in enumerate(c4s):
        for e_idx in c4:
            edge_c4s[e_idx].append(c4_idx)
    
    best_size = 0
    best_edges = []
    
    start_time = time.time()
    max_time = 110
    
    # Generate a random rank-3 linear map for tie-breaking
    # A is 3x7
    A = [[random.randint(0, 1) for _ in range(d)] for _ in range(3)]
    # Ensure rank 3? Random 3x7 is likely rank 3.
    # Compute L(u) for all u
    L = [0] * n
    for u in range(n):
        val = 0
        for k in range(3):
            bit = 0
            for i in range(d):
                if (u >> i) & 1:
                    bit ^= A[k][i]
            if bit:
                val |= (1 << k)
        L[u] = val
    
    for _ in range(500): # Fewer iterations, but smarter
        if time.time() - start_time > max_time:
            break
        
        selected = [True] * len(edges)
        c4_count = [0] * len(edges)
        
        # Initialize c4_count
        for e_idx in range(len(edges)):
            c4_count[e_idx] = len(edge_c4s[e_idx])
        
        # Active C4s
        active_c4s = list(range(len(c4s)))
        
        # Greedy removal
        # To make it deterministic-ish but varied, we can shuffle active_c4s initially?
        # Or just iterate.
        
        # We need to find a C4 with all 4 edges selected.
        # We can maintain a list of "bad" C4s.
        bad_c4s = set()
        for c4_idx in active_c4s:
            c4 = c4s[c4_idx]
            if all(selected[e] for e in c4):
                bad_c4s.add(c4_idx)
        
        while bad_c4s:
            # Pick a random bad C4? Or the one with max sum of c4_counts?
            # Let's pick the one with the edge having the highest c4_count.
            # To avoid O(N) scan every time, we can just pick a random one from bad_c4s
            # and break it.
            c4_idx = random.choice(list(bad_c4s))
            c4 = c4s[c4_idx]
            
            # Find the edge in this C4 with the highest c4_count
            # Tie-break by L(u) of the edge's start vertex
            max_count = -1
            best_e = -1
            best_l = 255
            for e_idx in c4:
                if selected[e_idx]:
                    cnt = c4_count[e_idx]
                    u, v = idx_to_edge[e_idx]
                    l_val = L[u]
                    if cnt > max_count or (cnt == max_count and l_val < best_l):
                        max_count = cnt
                        best_e = e_idx
                        best_l = l_val
            
            if best_e == -1:
                # Should not happen if c4 is in bad_c4s
                bad_c4s.remove(c4_idx)
                continue
            
            # Remove edge best_e
            selected[best_e] = False
            
            # Update c4_count for edges in C4s that contain best_e
            for c4_idx_2 in edge_c4s[best_e]:
                c4_2 = c4s[c4_idx_2]
                # Check if c4_2 is still "bad" (all selected)
                # If we just removed best_e, c4_2 is no longer bad.
                # But other edges in c4_2 have their c4_count reduced?
                # No, c4_count is the number of C4s the edge is in.
                # When we remove an edge, the C4s it was in are no longer "bad".
                # But the c4_count of *other* edges in those C4s doesn't change?
                # Wait, c4_count is static? No, it should reflect the current "danger".
                # Actually, standard greedy:
                # c4_count[e] = number of C4s containing e that are currently "bad" (all edges present).
                # When we remove e, all C4s containing e become "good" (not bad).
                # So for each C4 c4_2 containing e:
                #   For each other edge e2 in c4_2:
                #     c4_count[e2] decreases by 1.
                #   bad_c4s.remove(c4_2)
                
                if c4_idx_2 in bad_c4s:
                    bad_c4s.remove(c4_idx_2)
                    for e2 in c4_2:
                        if e2 != best_e and selected[e2]:
                            c4_count[e2] -= 1
                            
            # Note: best_e's c4_count doesn't matter anymore.
            
        current_size = sum(selected)
        if current_size > best_size:
            best_size = current_size
            best_edges = [[idx_to_edge[i][0], idx_to_edge[i][1]] for i in range(len(edges)) if selected[i]]
            
    return best_edges
