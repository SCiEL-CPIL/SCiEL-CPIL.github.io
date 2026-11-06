import random
import time
import sys

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Precompute all 4-cycles in K20 as sets of 4 edge indices
    # Edge index: (u, v) -> u*n + v for u < v
    def edge_id(u, v):
        if u > v:
            u, v = v, u
        return u * n + v

    # Map edge_id to (u, v)
    edge_map = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_map[edge_id(u, v)] = (u, v)
    total_edges = len(edge_map)

    # Find all 4-cycles
    # A 4-cycle is a set of 4 distinct vertices {a, b, c, d} with edges (a,b), (b,c), (c,d), (d,a)
    # We can find them by iterating over all 4-cliques? No, 4-cycles are not necessarily cliques.
    # We need to find all simple cycles of length 4.
    # For n=20, number of 4-cycles is manageable.
    # Let's enumerate them.
    cycles = []
    # Use adjacency list for K20 (complete graph)
    # We can just iterate over all combinations of 4 vertices and check for cycle structures.
    # For 4 vertices {a,b,c,d}, there are 3 possible cycle structures (up to rotation/reflection):
    # 1. a-b-c-d-a (edges: ab, bc, cd, da)
    # 2. a-b-d-c-a (edges: ab, bd, dc, ca)
    # 3. a-c-b-d-a (edges: ac, cb, bd, da)
    # Actually, it's easier to just check if the 4 edges form a cycle.
    # Let's use a standard method: for each pair of disjoint edges, check if they form a 4-cycle with two other edges.
    # Or simply: for each 4-set of vertices, check the 3 possible 4-cycles.
    
    import itertools
    all_vertices = list(range(n))
    for quad in itertools.combinations(all_vertices, 4):
        a, b, c, d = quad
        # Possible edges among these 4 vertices
        e_ab = edge_id(a, b)
        e_ac = edge_id(a, c)
        e_ad = edge_id(a, d)
        e_bc = edge_id(b, c)
        e_bd = edge_id(b, d)
        e_cd = edge_id(c, d)
        
        # Check 3 possible 4-cycles:
        # 1. a-b-c-d-a: edges ab, bc, cd, da
        cycles.append([e_ab, e_bc, e_cd, e_ad])
        # 2. a-b-d-c-a: edges ab, bd, dc, ca
        cycles.append([e_ab, e_bd, e_cd, e_ac])
        # 3. a-c-b-d-a: edges ac, cb, bd, da
        cycles.append([e_ac, e_bc, e_bd, e_ad])
        
    # Remove duplicate cycles (same set of edges)
    unique_cycles = set()
    for cyc in cycles:
        unique_cycles.add(tuple(sorted(cyc)))
    cycles = [list(c) for c in unique_cycles]
    
    # Precompute for each edge, which cycles it belongs to
    edge_to_cycles = [[] for _ in range(total_edges)]
    for i, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(i)
            
    # Precompute for each cycle, the 4 edges
    cycle_edges = cycles
    
    # We will use a branch-and-bound style search, but simplified.
    # Actually, let's use a smarter greedy + local search with backtracking.
    # Given the small size, we can use a randomized local search (Simulated Annealing) 
    # on the space of edge subsets and colorings.
    
    # State: 
    # present[e] = 1 if edge e is present, 0 otherwise
    # color[e] = color assigned to edge e (only meaningful if present[e]=1)
    
    # Objective: Maximize sum(present[e]) subject to rainbow constraint on all cycles.
    
    # Let's try a different approach: 
    # 1. Start with a random valid configuration (maybe sparse).
    # 2. Use local search: try to add an edge (pick a random missing edge, try all colors).
    #    If valid, add it.
    # 3. If stuck, try to remove an edge and re-add it with a different color, 
    #    or swap colors of two edges to allow adding a third.
    
    # Better: Use a "conflict-based" local search.
    # Maintain a count of conflicts for each edge.
    
    # Let's implement a fast validity checker.
    # For a given state (present, color), check if all cycles are rainbow.
    
    def is_valid_state(present, color):
        for cyc in cycle_edges:
            colors_in_cyc = []
            for e in cyc:
                if present[e]:
                    colors_in_cyc.append(color[e])
                else:
                    # If an edge is missing, the cycle is not formed. 
                    # Wait, the problem says "Every 4-cycle (as a subgraph) must receive four distinct colours".
                    # This implies the 4-cycle must exist in the graph. If an edge is missing, 
                    # that specific 4-cycle is not a subgraph, so no constraint?
                    # Yes, "as a subgraph" means all 4 edges must be present.
                    # So if any edge in the cycle is missing, the constraint is vacuously satisfied.
                    return True # Early exit? No, we need to check all cycles that are fully present.
                    # But we can break out of the inner loop for this cycle.
            # If we reach here, all 4 edges are present
            if len(set(colors_in_cyc)) < 4:
                return False
        return True
    
    # This is O(C * 4) per check. C ~ 1000-2000? 
    # Number of 4-cycles in K20: 
    # C(20,4) * 3 = 4845 * 3 = 14535 cycles.
    # 14535 * 4 = ~60k ops per check. 
    # 1000 checks = 60M ops. Might be too slow in Python.
    
    # Let's optimize: 
    # For each cycle, maintain a count of how many edges are present.
    # And a histogram of colors?
    
    # Let's try a simpler heuristic: 
    # 1. Greedy construction with random restarts and randomized edge order.
    # 2. Improve with a "repair" phase: if an edge can't be added, try to recolor existing edges.
    
    best_edges = 0
    best_present = [0] * total_edges
    best_color = [0] * total_edges
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        present = [0] * total_edges
        color = [0] * total_edges
        
        # Randomize edge order
        edge_indices = list(range(total_edges))
        random.shuffle(edge_indices)
        
        for e in edge_indices:
            if time.time() > deadline - 2:
                break
            u, v = edge_map[e]
            # Try to find a valid color
            # We need to check all cycles containing e.
            # For each such cycle, if the other 3 edges are present, 
            # the color must be distinct from theirs.
            forbidden_colors = set()
            for cyc_idx in edge_to_cycles[e]:
                cyc = cycle_edges[cyc_idx]
                # Check if other 3 edges are present
                other_edges = [x for x in cyc if x != e]
                if all(present[x] for x in other_edges):
                    # Get colors of other edges
                    for x in other_edges:
                        forbidden_colors.add(color[x])
            
            # Find a color not in forbidden_colors
            for c in range(num_colors):
                if c not in forbidden_colors:
                    present[e] = 1
                    color[e] = c
                    break
        
        if sum(present) > best_edges:
            best_edges = sum(present)
            best_present = present[:]
            best_color = color[:]
            
            # Local search: try to add more edges
            # Iterate through missing edges and try to add them
            # This is expensive, so do it only if time permits
            if time.time() < deadline - 10:
                for e in edge_indices:
                    if present[e]:
                        continue
                    # Try to add e
                    forbidden_colors = set()
                    for cyc_idx in edge_to_cycles[e]:
                        cyc = cycle_edges[cyc_idx]
                        other_edges = [x for x in cyc if x != e]
                        if all(present[x] for x in other_edges):
                            for x in other_edges:
                                forbidden_colors.add(color[x])
                    
                    # If no forbidden colors, we can add it with any color
                    # But wait, if we add it, it might create conflicts with OTHER cycles?
                    # No, the check above covers ALL cycles containing e.
                    # If we pick a color not in forbidden_colors, 
                    # then for every cycle where the other 3 edges are present, 
                    # the color is distinct.
                    # For cycles where not all other 3 are present, no constraint.
                    # So yes, it's valid.
                    
                    if len(forbidden_colors) < num_colors:
                        # Pick a random valid color
                        valid_colors = [c for c in range(num_colors) if c not in forbidden_colors]
                        c = random.choice(valid_colors)
                        present[e] = 1
                        color[e] = c
                        best_edges = sum(present)
                        best_present = present[:]
                        best_color = color[:]
                        
    # Convert to output format
    result = []
    for e in range(total_edges):
        if best_present[e]:
            u, v = edge_map[e]
            result.append([u, v, best_color[e]])
            
    return result
