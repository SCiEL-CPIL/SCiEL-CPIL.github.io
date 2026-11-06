import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # We want to maximize n. Try 18, 17, 16...
    # Known best is 17. 18 is the target for improvement.
    
    def find_c6(coloring, n, color):
        """Find a 6-cycle in the subgraph of the given color. Returns list of 6 vertices or None."""
        # Build adjacency list for this color
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                idx = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
                if coloring[idx] == color:
                    adj[i].append(j)
                    adj[j].append(i)
        
        # Check for C6
        # A C6 is v0-v1-v2-v3-v4-v5-v0
        # We can fix v0 and v1 to reduce symmetry, but for simplicity and speed in finding ANY, 
        # we can just iterate.
        
        # Optimization: If any node has degree < 2, no cycle.
        for u in range(n):
            if len(adj[u]) < 2:
                return None

        # Try to find a path of length 5 from u to v where (u,v) is an edge?
        # Or just DFS.
        
        # Let's use a simple check: 
        # Iterate over all pairs (u, v) that are connected.
        # Check if there is a path of length 4 from u to v that doesn't touch u or v internally.
        
        # To make it fast, we can precompute paths of length 2, 3, 4?
        # For n=20, this is small.
        
        # Let's just do a randomized check first, then systematic if needed?
        # No, systematic is better for correctness.
        
        # Path of length 4: u-a-b-c-v
        # a != u, v
        # b != u, v, a
        # c != u, v, b
        
        # We can iterate u, v connected.
        for u in range(n):
            for v in adj[u]:
                if u > v: continue # avoid duplicate
                # Check if path u-a-b-c-v exists
                # a in adj[u], a != v
                # c in adj[v], c != u
                # b in adj[a] AND b in adj[c]
                # b != u, v, a, c
                
                found = False
                for a in adj[u]:
                    if a == v: continue
                    for c in adj[v]:
                        if c == u or c == a: continue
                        # Check if a and c share a neighbor b
                        # b must be in adj[a] and adj[c]
                        # and b != u, v, a, c
                        for b in adj[a]:
                            if b == u or b == v or b == a or b == c: continue
                            if b in adj[c]:
                                return [u, a, b, c, v] # This is a path u-a-b-c-v. 
                                # Wait, the cycle is u-a-b-c-v-u. 
                                # Yes, this is a C6.
                                # Vertices: u, a, b, c, v. That's 5 vertices. 
                                # The cycle is u-a-b-c-v-u. 
                                # Length 5? No. 
                                # Edges: (u,a), (a,b), (b,c), (c,v), (v,u). 
                                # That's 5 edges. That's a C5.
                                # I need a path of length 4: u-a-b-c-d-v?
                                # Cycle: u-a-b-c-d-v-u. 6 edges.
                                # So I need path u-a-b-c-d-v of length 5?
                                # No, path of length 4 means 5 vertices: u, a, b, c, v.
                                # Edges: u-a, a-b, b-c, c-v.
                                # Plus edge v-u.
                                # Total 5 edges. C5.
                                
                                # I need C6.
                                # Cycle: v0-v1-v2-v3-v4-v5-v0.
                                # Fix v0=u, v1=v? No, u and v are connected.
                                # Let's fix edge (u, v).
                                # We need path v-a-b-c-d-u of length 4.
                                # Vertices: v, a, b, c, d, u.
                                # Edges: (v,a), (a,b), (b,c), (c,d), (d,u).
                                # Plus (u,v).
                                # Total 6 edges.
                                
                                # So I need to find a,b,c,d such that:
                                # (v,a) in E, (a,b) in E, (b,c) in E, (c,d) in E, (d,u) in E.
                                # All distinct.
                                
                                pass # The previous logic was for C5.
                
                # Let's implement C6 check properly.
                # For each edge (u, v):
                #   Find path of length 4 from u to v: u-a-b-c-v? No, u-a-b-c-d-v.
                #   a in adj[u], a!=v
                #   b in adj[a], b!=u,v
                #   c in adj[b], c!=u,v,a
                #   d in adj[c], d!=u,v,a,b, d in adj[v]
                
                for a in adj[u]:
                    if a == v: continue
                    for b in adj[a]:
                        if b == u or b == v: continue
                        for c in adj[b]:
                            if c == u or c == v or c == a: continue
                            for d in adj[c]:
                                if d == u or d == v or d == a or d == b: continue
                                if d in adj[v]:
                                    return [u, a, b, c, d, v]
        return None

    def count_c6s(coloring, n):
        count = 0
        for c in range(4):
            if find_c6(coloring, n, c) is not None:
                # This only finds ONE. We need count for annealing?
                # For greedy repair, we just need to find one to break.
                # So we can just return 1 if found, 0 if not?
                # No, we want to minimize.
                # Let's just use the "find one" approach for repair.
                pass
        return 0 # Placeholder

    best_n = 9
    best_coloring = None
    # Seed coloring for n=9
    seed_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    best_coloring = seed_colors
    
    for target_n in [18, 17, 16, 15]:
        if time.time() > deadline - 10: break
        
        found = False
        # Try multiple random restarts
        for restart in range(50):
            if time.time() > deadline - 10: break
            
            # Random init
            coloring = [random.randint(0, 3) for _ in range(target_n * (target_n - 1) // 2)]
            
            # Local search / Repair
            for step in range(1000):
                if time.time() > deadline - 10: break
                
                # Find a C6
                c6_found = None
                color_found = None
                for c in range(4):
                    cycle = find_c6(coloring, target_n, c)
                    if cycle:
                        c6_found = cycle
                        color_found = c
                        break
                
                if not c6_found:
                    # Valid coloring found!
                    if target_n > best_n:
                        best_n = target_n
                        best_coloring = coloring
                    found = True
                    break
                
                # Break the cycle
                # Pick a random edge from the cycle
                # Cycle is [v0, v1, v2, v3, v4, v5]
                # Edges are (v0,v1), (v1,v2), ..., (v5,v0)
                # Pick random index
                edge_idx = random.randint(0, 5)
                u = c6_found[edge_idx]
                v = c6_found[(edge_idx + 1) % 6]
                
                # Get current color
                idx = u * (target_n - 1) - u * (u + 1) // 2 + (v - u - 1) if u < v else v * (target_n - 1) - v * (v + 1) // 2 + (u - v - 1)
                # Ensure u < v for index calc
                if u > v: u, v = v, u
                idx = u * (target_n - 1) - u * (u + 1) // 2 + (v - u - 1)
                
                # Change to a different color
                new_color = random.choice([c for c in range(4) if c != coloring[idx]])
                coloring[idx] = new_color
                
            if found:
                break
                
    return {'n': best_n, 'colors': best_coloring}
