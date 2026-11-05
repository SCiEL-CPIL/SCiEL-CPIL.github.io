import time
import sys

def construct():
    start_time = time.time()
    
    # We will try to find a coloring for n=18, then 17, etc.
    # The task says "Maximize n".
    # Known best is 17. 18 is open.
    
    for n in [18, 17]:
        if time.time() - start_time > 110:
            break
            
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        
        # adj[c][u] is a bitmask of neighbors of u in color c
        adj = [[0] * n for _ in range(4)]
        colors = [-1] * num_edges
        
        # Symmetry breaking:
        # Fix colors of edges (0,1), (0,2), (0,3) to 0, 1, 2
        # Index 0: (0,1)
        # Index 1: (0,2)
        # Index 2: (0,3)
        
        def set_edge(u, v, c, idx):
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            colors[idx] = c

        def unset_edge(u, v, c, idx):
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            colors[idx] = -1

        # Apply symmetry breaking
        if n >= 4:
            set_edge(0, 1, 0, 0)
            set_edge(0, 2, 1, 1)
            set_edge(0, 3, 2, 2)
            start_idx = 3
        else:
            start_idx = 0

        def has_path_len_5(u, v, c):
            # Check if there is a simple path of length 5 from u to v in G_c
            # We compute the set of nodes reachable in exactly k steps.
            # reach_k is a bitmask.
            # reach_0 = {u}
            # reach_{k+1} = { w | exists x in reach_k such that (x,w) in E_c }
            # We must ensure simple path? 
            # "No colour class may contain a six-cycle (exact search for not necessarily induced cycles)"
            # A cycle of length 6 is a simple cycle.
            # If there is a walk of length 5 from u to v, does it imply a simple path?
            # Not necessarily. But if there is a simple path of length 5, adding (u,v) creates a C6.
            # If there is a non-simple walk, it might contain a shorter cycle.
            # However, if G_c is C6-free, can it have a walk of length 5 between u and v?
            # If it has a walk of length 5, it might have a simple path of length 5, 3, or 1.
            # If it has a simple path of length 1, then (u,v) is already an edge, which is not the case (we are adding it).
            # If it has a simple path of length 3, adding (u,v) creates a C4. The problem only forbids C6.
            # Wait, the constraint is "No colour class may contain a six-cycle". It does NOT forbid C4.
            # So, if there is a path of length 3, adding (u,v) creates a C4, which is allowed.
            # If there is a path of length 5, adding (u,v) creates a C6, which is forbidden.
            # So we need to check if there is a *simple* path of length 5.
            
            # However, checking for simple paths is hard with bitsets.
            # But for N=18, we can use the fact that if a walk of length 5 exists, 
            # and the graph is C6-free, what does that imply?
            # Actually, the standard check for "adding edge (u,v) creates a C6" is:
            # Is there a path of length 5 between u and v in G_c \ {(u,v)}?
            # Since (u,v) is not in G_c yet, we just check G_c.
            
            # Let's use the property: If there is a simple path of length 5, 
            # there is a walk of length 5.
            # If there is a walk of length 5, does it guarantee a simple path of length 5?
            # No. It could be a walk that repeats nodes.
            # However, if the graph has no C6, and we are checking if adding (u,v) creates a C6,
            # we are looking for a cycle of length 6.
            # A cycle of length 6 is a simple cycle.
            # So we need a simple path of length 5.
            
            # For small N, we can do a DFS for the path check, but optimize it.
            # Or, we can observe that for sparse graphs, walks are unlikely to be non-simple without forming smaller cycles.
            # But C4s are allowed.
            
            # Let's stick to the DFS but make it iterative and fast.
            # The previous DFS was slow because it explored all paths.
            # We can prune if we visit a node twice.
            
            # Alternative: 
            # If there is a path of length 5, it uses 6 distinct vertices.
            # We can iterate over all pairs (a,b) at distance 2 from u and (c,d) at distance 2 from v?
            
            # Let's try a faster DFS:
            # stack: (node, depth, visited_mask, prev)
            # We want to reach v at depth 5.
            
            # Actually, for N=18, the number of paths of length 5 is small if the graph is sparse.
            # The current DFS is $O(N^5)$ worst case, but with visited mask it is bounded by $N \times 2^N$ which is huge.
            # But in practice, depth is only 5.
            # Number of paths of length 5 is at most $N^5$. For N=18, $18^5 \approx 1.9 \times 10^6$.
            # This is too slow if done for every edge assignment (153 edges * 4 colors * 1.9M = 1B ops).
            
            # We need a better check.
            # Check if $v$ is in the 5-step neighborhood of $u$ in the *transitive closure* sense? No.
            
            # Let's use the matrix multiplication approach for "exact length 5 paths" but allowing revisits?
            # No, we need simple paths.
            
            # However, there is a theorem: If a graph has no C6, the number of paths of length 5 is limited?
            
            # Let's try a different heuristic: 
            # Instead of checking all colors for all edges, we can use a constraint propagation.
            
            # But for now, let's optimize the DFS.
            # We can stop early if we find a path.
            # We can order neighbors to find paths faster?
            
            # Let's try to use the bitset reachability for *walks* first.
            # If there is NO walk of length 5, then there is no simple path.
            # If there IS a walk, we need to verify if it's simple.
            # For most assignments, there will be no walk, so the bitset check (fast) will reject quickly.
            # Only when a walk exists do we need the slower DFS.
            
            # Compute reachability in exactly 5 steps (walks).
            # reach_1 = adj[c][u]
            # reach_2 = 0
            # for x in bits(reach_1): reach_2 |= adj[c][x]
            # ...
            
            def get_walks_len_5(u, c):
                # Returns bitmask of nodes reachable from u in exactly 5 steps (walks)
                reach = 1 << u
                for _ in range(5):
                    new_reach = 0
                    r = reach
                    while r:
                        lsb = r & -r
                        x = lsb.bit_length() - 1
                        r ^= lsb
                        new_reach |= adj[c][x]
                    # Exclude u? No, walks can return to u.
                    # But we are looking for v.
                    reach = new_reach
                return reach

            # Check if v is in walks_len_5(u)
            if not (get_walks_len_5(u, c) & (1 << v)):
                return False
                
            # If a walk exists, we must check if a simple path exists.
            # This is rare.
            # DFS for simple path of length 5.
            # We can use the walk existence to guide the DFS?
            
            # For now, if a walk exists, we assume it's a simple path (optimistic, might be wrong).
            # If we are wrong, we might accept an invalid coloring.
            # We MUST be correct.
            
            # Let's do the DFS but limit it to the nodes involved in the walk?
            # Or just do the DFS. It should be rare.
            
            # DFS
            stack = [(u, 0, 1 << u)]
            while stack:
                node, depth, visited = stack.pop()
                if depth == 5:
                    if node == v:
                        return True
                    continue
                
                neighbors = adj[c][node]
                while neighbors:
                    lsb = neighbors & -neighbors
                    nxt = lsb.bit_length() - 1
                    neighbors ^= lsb
                    
                    if visited & lsb:
                        continue
                    
                    if depth == 4 and nxt == v:
                        return True
                        
                    stack.append((nxt, depth + 1, visited | lsb))
            return False

        def backtrack(idx):
            if time.time() - start_time > 110:
                return False
            if idx == num_edges:
                return True
            
            u, v = edges[idx]
            
            for c in range(4):
                if has_path_len_5(u, v, c):
                    continue
                
                set_edge(u, v, c, idx)
                
                if backtrack(idx + 1):
                    return True
                
                unset_edge(u, v, c, idx)
            
            return False

        if backtrack(start_idx):
            return {'n': n, 'colors': colors}
            
    # Fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
