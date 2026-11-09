"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

import random
import time

def construct():
    n = 20
    max_colors = 20
    
    # Precompute all 4-cycles: each cycle is (u, v, w, x) meaning u-v-w-x-u
    # We need to check that all edges in the cycle have distinct colors
    # Precompute which cycles contain which edge pairs
    all_4cycles = []
    for u in range(n):
        for v in range(u+1, n):
            for w in range(n):
                if w in (u, v):
                    continue
                for x in range(n):
                    if x in (u, v, w):
                        continue
                    # Cycle u-v-w-x-u requires edges: (u,v), (v,w), (w,x), (x,u)
                    # We'll store as tuple of 4 edges
                    edges_in_cycle = ((u,v), (v,w), (w,x), (u,x))
                    all_4cycles.append(edges_in_cycle)
    
    # For each edge, precompute which cycles contain it
    edge_to_cycles = {}
    for i, cycle in enumerate(all_4cycles):
        for edge in cycle:
            e = tuple(sorted(edge))
            if e not in edge_to_cycles:
                edge_to_cycles[e] = []
            edge_to_cycles[e].append(i)
    
    def solve_once(seed):
        random.seed(seed)
        # adj[u][v] = color or -1
        adj = [[-1]*n for _ in range(n)]
        
        # For each edge, track which colors are forbidden by existing 4-cycles
        # forbidden[edge][color] = True if color would create a rainbow violation
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        edge_index = {(i, j): idx for idx, (i, j) in enumerate(all_edges)}
        
        # forbidden_color_mask[edge_idx][color] = 1 if color c is forbidden for edge
        # A color c is forbidden for edge e if there exists a 4-cycle containing e
        # where the other 3 edges already have colors that include c or create duplicates
        # Actually, let's think more carefully:
        # When adding edge e=(u,v) with color c, we need to check all 4-cycles containing e.
        # For each such cycle with other edges e1, e2, e3 with colors c1, c2, c3:
        #   we need c, c1, c2, c3 to all be distinct
        # So c is forbidden if c == c1 or c == c2 or c == c3
        # Also, the existing colors c1, c2, c3 must already be distinct (they are, since we maintain invariant)
        
        # So for each edge e, maintain a set of forbidden colors
        forbidden = [set() for _ in range(len(all_edges))]
        
        # When we add an edge e with color c, we update forbidden sets for other edges
        # in cycles containing e
        
        random.shuffle(all_edges)
        
        for (u, v) in all_edges:
            e_idx = edge_index[(u, v)]
            # Find a valid color
            valid_colors = [c for c in range(max_colors) if c not in forbidden[e_idx]]
            if not valid_colors:
                continue
            
            # Choose a color - use heuristic: pick color that minimizes future conflicts
            # Simple: pick random valid color
            c = random.choice(valid_colors)
            
            adj[u][v] = c
            adj[v][u] = c
            
            # Update forbidden sets for edges in cycles containing (u,v)
            if (u, v) in edge_to_cycles:
                for cycle_idx in edge_to_cycles[(u, v)]:
                    cycle = all_4cycles[cycle_idx]
                    for other_edge in cycle:
                        if other_edge == (u, v) or other_edge == (v, u):
                            continue
                        oe = tuple(sorted(other_edge))
                        if oe not in edge_index:
                            continue
                        oe_idx = edge_index[oe]
                        # If oe already has a color, it's already been placed
                        # We need to check: does adding c to (u,v) conflict with oe's color?
                        # The cycle has 4 edges: (u,v), other_edge, and two more
                        # For the rainbow constraint, all 4 colors must be distinct
                        # So c must differ from the colors of the other 3 edges in the cycle
                        # But we already checked that c is valid for (u,v)
                        # Now we need to update forbidden for oe: c is now used in a cycle with oe
                        # So for oe, c is now forbidden (if oe hasn't been placed yet)
                        # Wait, this is not quite right. Let me reconsider.
                        
                        # Actually, the forbidden set for an edge e should contain colors c such that
                        # if e is assigned color c, there exists a 4-cycle containing e where the
                        # other 3 edges have colors that conflict with c.
                        
                        # When we assign color c to edge (u,v), for any other edge oe in a 4-cycle
                        # with (u,v), the color c is now "used" in that cycle. So if oe is later
                        # assigned color c, it would create a conflict in that cycle.
                        # So we add c to forbidden[oe_idx].
                        
                        # But wait, we also need to consider that the other two edges in the cycle
                        # might have colors that conflict. Let's just add c to forbidden for all
                        # other edges in cycles containing (u,v).
                        
                        # Actually, this is an over-approximation. Let's be more precise:
                        # For a 4-cycle with edges e0=(u,v), e1, e2, e3:
                        # If e0 has color c, then for e1, the forbidden colors are those that
                        # would make the cycle non-rainbow. Since e2 and e3 may or may not be
                        # placed yet, we can't fully determine this.
                        
                        # Let's use a simpler approach: just track which colors are used
                        # in each cycle, and when adding an edge, check all cycles.
                        
                        pass  # We'll use direct checking instead
        
        # This approach is getting complex. Let me use a cleaner direct-check approach
        # with precomputed cycle incidence.
        
        # Restart with cleaner implementation
        adj = [[-1]*n for _ in range(n)]
        random.shuffle(all_edges)
        
        # Precompute for each edge, the list of (other_edge1, other_edge2, other_edge3) 
        # for each 4-cycle containing it
        # Actually, let's just do direct checking with numpy for speed
        
        for (u, v) in all_edges:
            # Try colors in random order
            colors_to_try = list(range(max_colors))
            random.shuffle(colors_to_try)
            
            for c in colors_to_try:
                valid = True
                # Check all 4-cycles containing edge (u,v)
                # A 4-cycle containing (u,v) has form: u-v-w-x-u or u-v-x-w-u
                # i.e., the other two vertices w, x form a path from v to u through 2 edges
                # or equivalently, w and x are such that:
                #   v-w, w-x, x-u are edges (for cycle u-v-w-x-u)
                #   or v-x, x-w, w-u are edges (for cycle u-v-x-w-u)
                
                for w in range(n):
                    if w in (u, v):
                        continue
                    if adj[v][w] == -1:
                        continue
                    c_vw = adj[v][w]
                    if c_vw == c:
                        valid = False
                        break
                    for x in range(n):
                        if x in (u, v, w):
                            continue
                        # Check cycle u-v-w-x-u: need edges (w,x) and (x,u)
                        if adj[w][x] != -1 and adj[x][u] != -1:
                            c_wx = adj[w][x]
                            c_xu = adj[x][u]
                            # Colors: c, c_vw, c_wx, c_xu must be all distinct
                            if c == c_wx or c == c_xu or c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    adj[u][v] = c
                    adj[v][u] = c
                    break
        
        result = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] != -1:
                    result.append([i, j, adj[i][j]])
        return result

    best_edges = []
    # Run multiple restarts to find a better local optimum
    for i in range(100):
        res = solve_once(i)
        if len(res) > len(best_edges):
            best_edges = res
            
    return best_edges
# EVOLVE-BLOCK-END
