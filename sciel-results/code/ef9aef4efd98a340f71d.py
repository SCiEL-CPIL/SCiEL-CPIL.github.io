import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 20
    max_edges = n * (n - 1) // 2
    
    # Pre-generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # We will try multiple restarts with different random edge orderings
    best_edges = []
    best_count = 0
    
    # To speed up the check, we maintain the current graph structure
    # But checking 4-cycles efficiently is key.
    
    # A 4-cycle is formed by adding edge (u,v) if there exists a path of length 3 between u and v.
    # Specifically, if there exist a,b such that u-a, a-b, b-v are in the graph.
    # For a new edge (u,v) with color c, we must ensure that for all such paths u-a-b-v,
    # the colors of (u,a), (a,b), (b,v) are distinct from each other and from c.
    # Actually, the condition is: every 4-cycle must have 4 distinct colors.
    # When adding (u,v), it creates 4-cycles with any path u-a-b-v.
    # The cycle is u-v-b-a-u. The edges are (u,v), (v,b), (b,a), (a,u).
    # We need colors of these 4 edges to be distinct.
    
    # Let's use a greedy approach with randomization.
    # We assign colors greedily as well, or fix a random color permutation?
    # No, we can choose the color for the new edge.
    # If we add (u,v), we can choose a color c that doesn't conflict with existing 4-cycles involving (u,v).
    # But wait, (u,v) is new, so it only forms 4-cycles where (u,v) is one of the edges.
    # The other 3 edges are already in the graph.
    # So for each path u-a-b-v, we have colors c_ua, c_ab, c_bv.
    # We need to pick c_uv such that c_uv is not in {c_ua, c_ab, c_bv} AND c_ua, c_ab, c_bv are all distinct.
    # Wait, if c_ua, c_ab, c_bv are NOT all distinct, then ANY color c_uv will result in a non-rainbow 4-cycle?
    # No, the 4-cycle must have 4 distinct colors. If the existing 3 edges have duplicate colors, then the 4-cycle cannot be rainbow regardless of c_uv.
    # Therefore, we can ONLY add (u,v) if for EVERY path u-a-b-v, the colors of the 3 existing edges are distinct.
    # AND then we must pick a color c_uv that is distinct from those 3.
    # If multiple paths exist, c_uv must be distinct from the union of all colors appearing in any such path's 3 edges.
    
    # This suggests a strong constraint.
    
    # Let's implement a fast check.
    # Data structures:
    # adj[u] = {v: color}
    # We need to check if adding (u,v) is valid.
    # 1. Find all length-2 paths between neighbors of u and neighbors of v?
    #    Actually, path is u-a-b-v.
    #    a is neighbor of u, b is neighbor of v.
    #    And a-b must be an edge.
    #    So iterate a in adj[u], b in adj[v], check if a-b is edge.
    
    # Let's try a few restarts.
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        random.shuffle(all_edges)
        
        adj = [dict() for _ in range(n)]
        edge_count = 0
        valid = True
        
        # We need to track colors used? No, just check constraints on the fly.
        # To be fast, we might need to precompute neighbor lists or use bitmasks?
        # n=20 is small.
        
        for u, v in all_edges:
            if time.time() > deadline - 5:
                valid = False
                break
                
            # Check if adding (u,v) is valid
            # 1. Collect forbidden colors
            forbidden_colors = set()
            possible = True
            
            # Iterate over common "middle" edges
            # Path u-a-b-v
            # a in adj[u], b in adj[v]
            # Check if (a,b) is in graph
            
            # Optimization: Precompute list of edges? Or just check adj[a].get(b)
            
            for a, c_ua in adj[u].items():
                for b, c_bv in adj[v].items():
                    # Check if a-b is an edge
                    if a == b: continue # Triangle u-a-v? No, a-b must be edge. If a==b, path is u-a-v which is length 2, not 3.
                    # We need path u-a-b-v. So a and b are distinct.
                    c_ab = adj[a].get(b)
                    if c_ab is not None:
                        # We found a path u-a-b-v
                        # Colors involved: c_ua, c_ab, c_bv
                        # If any of these are equal, this path creates a "bad" 4-cycle structure?
                        # No, the 4-cycle is u-v-b-a-u.
                        # Edges: (u,v) [new], (v,b) [c_bv], (b,a) [c_ab], (a,u) [c_ua].
                        # For this 4-cycle to be rainbow, c_ua, c_ab, c_bv must be distinct from each other AND from c_uv.
                        # If c_ua == c_ab, then no matter what c_uv is, the cycle has duplicate colors.
                        # So if c_ua == c_ab or c_ua == c_bv or c_ab == c_bv, we CANNOT add (u,v) because this specific 4-cycle can never be rainbow.
                        if c_ua == c_ab or c_ua == c_bv or c_ab == c_bv:
                            possible = False
                            break
                        # Otherwise, c_uv must not be any of these 3 colors
                        forbidden_colors.add(c_ua)
                        forbidden_colors.add(c_ab)
                        forbidden_colors.add(c_bv)
                        # Note: We don't break here, we need to check ALL such paths to collect all forbidden colors.
                        # But if we find a conflict in existing colors, we break.
                    
                    if not possible:
                        break
                if not possible:
                    break
            
            if not possible:
                continue
                
            # If possible, pick a color for (u,v)
            # We have 20 colors. Forbidden set size is at most... well, depends on number of paths.
            # If len(forbidden_colors) == 20, we can't add.
            if len(forbidden_colors) == 20:
                continue
                
            # Pick a random valid color
            available_colors = [c for c in range(20) if c not in forbidden_colors]
            if not available_colors:
                continue
            c_uv = random.choice(available_colors)
            
            # Add edge
            adj[u][v] = c_uv
            adj[v][u] = c_uv
            edge_count += 1
            
        if valid:
            # Reconstruct edge list
            current_edges = []
            for u in range(n):
                for v, c in adj[u].items():
                    if u < v:
                        current_edges.append([u, v, c])
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = current_edges
                # If we reached a high number, maybe stop early?
                # Max possible is 190. Dense graphs likely around 100-150?
                if best_count > 150:
                    # Break restarts? Maybe not, but it's a good sign.
                    pass

    return best_edges
