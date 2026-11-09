"""
Dense 20-vertex edge-coloured graph with 20 colours, every 4-cycle rainbow.
Strategy: use the complete graph K20 and assign colours via a proper edge-colouring
that guarantees every 4-cycle is rainbow. Then verify and, if needed, remove edges
that create non-rainbow 4-cycles to maximize density.

Key insight: A 1-factorization of K20 gives 19 perfect matchings. If we use 19 colours
for a 1-factorization, every edge gets a unique colour within its matching. However,
this doesn't guarantee 4-cycles are rainbow.

Better approach: Use a greedy construction with backtracking. Start with all 190 edges
of K20, assign colours, and remove edges that violate the rainbow-4-cycle constraint.
The constraint is: for every 4-cycle (a,b,c,d,a), the four edge colours must be distinct.

We'll use a randomised greedy approach:
1. Start with K20 (all 190 edges).
2. Assign colours randomly or via a structured method.
3. Check all 4-cycles; if any has a repeated colour, remove one of its edges.
4. Iterate until no violations.

To make this efficient, we use the fact that a 4-cycle (a,b,c,d) has edges ab, bc, cd, da.
The constraint is that these four colours are all distinct.

We'll try a structured colouring: use a Latin square based approach. Label vertices 0..19.
For edge (u,v) with u<v, assign colour (u+v) mod 19 if u+v < 19, or handle carefully.

Actually, let's use a simpler approach: 
- Use 19 colours in a cyclic pattern. For edge (u,v), colour = (u+v) mod 19.
- This might not work perfectly, so we verify and fix.

Let me think more carefully. We have 20 vertices and 20 colours. K20 has 190 edges.
The maximum number of edges in a C4-free graph on 20 vertices is bounded, but we don't need
C4-free; we need every C4 to be rainbow. This is a weaker condition than C4-free.

A key result: if we can properly edge-colour K20 such that every 4-cycle is rainbow, we get
190 edges. This is possible if we use a "rainbow" edge-colouring.

Let's try: assign to edge (u,v) the colour (u + 2*v) mod 20, but ensure no two edges in any
4-cycle share a colour. This is complex.

Simpler: use a random greedy approach with verification.
"""

import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    n = 20
    num_colors = 20
    
    # Helper: check if a 4-cycle is rainbow
    def is_rainbow_4cycle(colors, a, b, c, d, edge_color):
        """Check if 4-cycle a-b-c-d-a has all distinct colours."""
        try:
            c1 = edge_color[(a, b)] if (a, b) in edge_color else edge_color[(b, a)]
            c2 = edge_color[(b, c)] if (b, c) in edge_color else edge_color[(c, b)]
            c3 = edge_color[(c, d)] if (c, d) in edge_color else edge_color[(d, c)]
            c4 = edge_color[(d, a)] if (d, a) in edge_color else edge_color[(a, d)]
            return len({c1, c2, c3, c4}) == 4
        except:
            return True  # If any edge missing, not a 4-cycle
    
    # Helper: find all 4-cycles in the current graph
    def find_all_4cycles(edge_set):
        """Find all 4-cycles. A 4-cycle is a set of 4 distinct vertices {a,b,c,d}
        where all four edges ab, bc, cd, da exist (in some order)."""
        cycles = []
        vertices = list(range(n))
        for a, b, c, d in combinations(vertices, 4):
            # Check all 3 possible 4-cycle orderings on these 4 vertices
            # Orderings: a-b-c-d, a-b-d-c, a-c-b-d
            for ordering in [(a,b,c,d), (a,b,d,c), (a,c,b,d)]:
                x, y, z, w = ordering
                if (min(x,y), max(x,y)) in edge_set and (min(y,z), max(y,z)) in edge_set and \
                   (min(z,w), max(z,w)) in edge_set and (min(w,x), max(w,x)) in edge_set:
                    cycles.append((x, y, z, w))
                    break
        return cycles
    
    # Helper: check if a colour assignment is valid
    def is_valid(edge_color, edge_set):
        """Check all 4-cycles are rainbow."""
        cycles = find_all_4cycles(edge_set)
        for (a, b, c, d) in cycles:
            if not is_rainbow_4cycle(edge_color, a, b, c, d, edge_color):
                return False
        return True
    
    # Helper: get edges as list
    def edges_from_set(edge_set):
        result = []
        for (u, v) in edge_set:
            c = edge_color[(u, v)] if (u, v) in edge_color else edge_color[(v, u)]
            result.append([u, v, c])
        return result
    
    # Strategy 1: Try a structured colouring on K20
    # Use colour (u+v) mod 19 for u<v, but we have 20 colours so use mod 20 carefully
    
    # Let's try: for edge (u,v) with u<v, colour = (u + v) % 20
    # But this may cause conflicts. Let's test it.
    
    best_edges = None
    best_count = 0
    
    # Try multiple random seeds
    for seed in range(100):
        if time.time() > deadline:
            break
        random.seed(seed)
        
        # Start with all edges of K20
        all_edges = set()
        for u in range(n):
            for v in range(u+1, n):
                all_edges.add((u, v))
        
        # Assign initial colours: (u+v) % 20
        edge_color = {}
        for (u, v) in all_edges:
            edge_color[(u, v)] = (u + v) % 20
        
        # Iteratively remove edges that create non-rainbow 4-cycles
        edge_set = set(all_edges)
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            # Find a violating 4-cycle
            cycles = find_all_4cycles(edge_set)
            for (a, b, c, d) in cycles:
                if not is_rainbow_4cycle(edge_color, a, b, c, d, edge_color):
                    # Remove a random edge from this cycle
                    edges_in_cycle = [(a,b), (b,c), (c,d), (d,a)]
                    # Normalize
                    norm_edges = []
                    for (x, y) in edges_in_cycle:
                        norm_edges.append((min(x,y), max(x,y)))
                    # Pick one to remove
                    rem_edge = random.choice(norm_edges)
                    edge_set.remove(rem_edge)
                    del edge_color[rem_edge]
                    changed = True
                    break
        
        # Verify
        if is_valid(edge_color, edge_set):
            count = len(edge_set)
            if count > best_count:
                best_count = count
                best_edges = edges_from_set(edge_set)
    
    # If we found a good solution, return it
    if best_edges is not None and best_count > 0:
        return best_edges
    
    # Fallback: star
    return [[0, v, 0] for v in range(1, 20)]
