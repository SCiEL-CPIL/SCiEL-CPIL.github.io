import sys

def construct():
    for n in [47, 46]:
        # Generate all C4s
        c4s = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        c4s.append([(i,j), (j,k), (k,l), (l,i)])
                        c4s.append([(i,j), (j,l), (l,k), (k,i)])
                        c4s.append([(i,k), (k,j), (j,l), (l,i)])
        
        num_c4s = len(c4s)
        
        # Precompute edge to C4 indices
        edge_to_c4s = {}
        for i in range(n):
            for j in range(i+1, n):
                edge_to_c4s[(i,j)] = []
                
        for idx, c4 in enumerate(c4s):
            for e in c4:
                edge_to_c4s[e].append(idx)
                
        active = [True] * num_c4s
        active_count = num_c4s
        
        # edge_counts
        edge_counts = {e: len(edge_to_c4s[e]) for e in edge_to_c4s}
        
        deg = [0] * n
        F_edges = set()
        
        while active_count > 0:
            # Find an active C4
            idx = -1
            for i in range(num_c4s):
                if active[i]:
                    idx = i
                    break
            if idx == -1:
                break
                
            c4 = c4s[idx]
            
            best_edge = None
            best_score = -1
            
            for e in c4:
                u, v = e
                if deg[u] >= 39 or deg[v] >= 39:
                    continue
                
                hits = edge_counts[e]
                cost = deg[u] + deg[v] + 1
                score = hits / cost
                
                if score > best_score:
                    best_score = score
                    best_edge = e
            
            if best_edge is None:
                active_count = 0
                break
                
            u, v = best_edge
            F_edges.add(best_edge)
            deg[u] += 1
            deg[v] += 1
            
            # Update
            for c4_idx in edge_to_c4s[best_edge]:
                if active[c4_idx]:
                    active[c4_idx] = False
                    active_count -= 1
                    # Decrement counts for other edges in this C4
                    for e in c4s[c4_idx]:
                        if e != best_edge:
                            edge_counts[e] -= 1
                        
        if active_count == 0:
            edges_G = []
            for u in range(n):
                for v in range(u+1, n):
                    if (u, v) not in F_edges:
                        edges_G.append([u, v])
                        
            return {'n': n, 'edges': edges_G}
            
    return {'n': 39, 'edges': []}
