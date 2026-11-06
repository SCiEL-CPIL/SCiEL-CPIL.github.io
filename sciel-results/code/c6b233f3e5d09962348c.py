import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560

# Precompute triple index lookup
_triple_idx = {}
for _i, _t in enumerate(ALL_TRIPLES):
    _triple_idx[_t] = _i

def _tidx(a, b, c):
    """Get index of triple {a,b,c} (all distinct)."""
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return _triple_idx[(a, b, c)]

def _precompute_conflicts():
    """
    Precompute the conflict graph for all triples.
    Two triples conflict if they are part of the same K(2,2,2).
    A K(2,2,2) is defined by 6 distinct points {a1,a2,b1,b2,c1,c2}.
    The 8 transversal triples are all combinations of choosing one from each pair.
    Any two of these 8 triples are in conflict.
    """
    # adj[i] will be a set of indices j such that i and j conflict
    adj = [set() for _ in range(N_TRIPLES)]
    
    # Iterate over all possible K(2,2,2) configurations
    # A K(2,2,2) is determined by 3 disjoint pairs of vertices.
    # Total ways to choose 6 distinct vertices from 16: C(16,6) = 8008
    # For each set of 6, there are 15 ways to partition into 3 pairs.
    # However, we can just iterate over all 6-tuples and check distinctness, 
    # or more efficiently, iterate over combinations of 6 points.
    
    points = list(range(V))
    
    for combo in combinations(points, 6):
        # For each set of 6 points, find all ways to partition into 3 pairs
        # We can do this by fixing the first point's partner, etc.
        # But simpler: just generate the 15 pairings?
        # Actually, we can just iterate over all ways to pick the 3 pairs from the 6 points.
        # There are 15 perfect matchings on 6 vertices.
        
        # Let's just iterate over all ways to assign the 6 points to (a1,a2), (b1,b2), (c1,c2)
        # But we must avoid duplicates.
        # A simpler way:
        # Pick a1 from combo. Pick a2 from remaining.
        # Pick b1 from remaining. Pick b2 from remaining.
        # c1, c2 are the last two.
        # This generates each K(2,2,2) multiple times, but that's okay, we just add edges.
        
        # To be efficient and avoid massive duplication:
        # We can just take the 6 points p0..p5.
        # The 15 matchings are:
        # 0-1, 2-3, 4-5
        # 0-1, 2-4, 3-5
        # ...
        
        # Let's just use a standard list of matchings for 6 items.
        # Or simpler:
        # Iterate over all subsets of 3 pairs? No.
        
        # Let's just iterate over all permutations of the 6 points into 3 pairs?
        # That's 6! / (2!^3 * 3!) = 720 / 48 = 15.
        # We can generate these 15 matchings explicitly.
        
        p = list(combo)
        # Generate all 15 perfect matchings of indices 0..5
        matchings = [
            [(0,1),(2,3),(4,5)],
            [(0,1),(2,4),(3,5)],
            [(0,1),(2,5),(3,4)],
            [(0,2),(1,3),(4,5)],
            [(0,2),(1,4),(3,5)],
            [(0,2),(1,5),(3,4)],
            [(0,3),(1,2),(4,5)],
            [(0,3),(1,4),(2,5)],
            [(0,3),(1,5),(2,4)],
            [(0,4),(1,2),(3,5)],
            [(0,4),(1,3),(2,5)],
            [(0,4),(1,5),(2,3)],
            [(0,5),(1,2),(3,4)],
            [(0,5),(1,3),(2,4)],
            [(0,5),(1,4),(2,3)]
        ]
        
        for m in matchings:
            # m is a list of 3 pairs of indices into p
            # p[m[0][0]], p[m[0][1]] are a1, a2
            # p[m[1][0]], p[m[1][1]] are b1, b2
            # p[m[2][0]], p[m[2][1]] are c1, c2
            
            a1, a2 = p[m[0][0]], p[m[0][1]]
            b1, b2 = p[m[1][0]], p[m[1][1]]
            c1, c2 = p[m[2][0]], p[m[2][1]]
            
            # Generate the 8 transversal triples
            triples_in_k222 = [
                _tidx(a1, b1, c1),
                _tidx(a1, b1, c2),
                _tidx(a1, b2, c1),
                _tidx(a1, b2, c2),
                _tidx(a2, b1, c1),
                _tidx(a2, b1, c2),
                _tidx(a2, b2, c1),
                _tidx(a2, b2, c2)
            ]
            
            # All pairs of these triples conflict
            for i in range(8):
                for j in range(i+1, 8):
                    u = triples_in_k222[i]
                    v = triples_in_k222[j]
                    adj[u].add(v)
                    adj[v].add(u)
                    
    return adj

# Precompute adjacency lists
ADJ_LISTS = _precompute_conflicts()

# Convert to bitsets for fast operations
# N_TRIPLES = 560, so we need 560 bits, which fits in a Python int.
ADJ_BITS = [0] * N_TRIPLES
for i in range(N_TRIPLES):
    bits = 0
    for j in ADJ_LISTS[i]:
        bits |= (1 << j)
    ADJ_BITS[i] = bits

def _max_independent_set(adj_bits, n, deadline):
    """
    Find Maximum Independent Set using Tomita's branch and bound.
    adj_bits[i] is the bitmask of neighbors of i.
    n is the number of vertices.
    Returns the best independent set found (as a bitmask) and its size.
    """
    best_set = 0
    best_size = 0
    
    # Global variables for recursion state
    # We need to be careful with Python's recursion limit and speed.
    # For n=560, recursion depth can be up to 560.
    # sys.setrecursionlimit(10000) might be needed, but iterative or careful recursion is better.
    # Tomita's algorithm is recursive.
    
    import sys
    sys.setrecursionlimit(10000)
    
    # Greedy coloring for upper bound
    # We need to color the subgraph induced by R (candidates).
    # R is represented as a bitmask.
    # We need a function to compute a coloring of the subgraph induced by R.
    # Returns a list of colors for vertices in R, and the number of colors used.
    # Actually, Tomita's algorithm uses a specific ordering and coloring.
    
    # Helper: Get list of vertices from bitmask
    def get_vertices(mask):
        verts = []
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            verts.append(idx)
            mask ^= lsb
        return verts

    # Helper: Greedy coloring of subgraph induced by R
    # Returns a list of vertices ordered by color, and the max color used.
    # This is used to generate the upper bound.
    def color_subgraph(R):
        # R is a bitmask
        # We want to color the graph induced by R.
        # Standard greedy coloring:
        # 1. Order vertices in R.
        # 2. Assign smallest available color.
        
        # For Tomita, we often order vertices by degree or just use natural order.
        # Let's use natural order (increasing index) for simplicity and determinism, 
        # or maybe decreasing degree for better bounds? 
        # Decreasing degree usually gives better upper bounds (fewer colors).
        
        verts = get_vertices(R)
        # Sort by degree in the subgraph? 
        # Degree in subgraph for v is popcount(R & ADJ_BITS[v])
        # This is expensive to compute for all v every time.
        # Let's stick to natural order or precomputed degree.
        
        # Precomputed degree in full graph:
        # deg[v] = popcount(ADJ_BITS[v])
        
        # Let's just use natural order for now.
        colors = {}
        color_used = [False] * 1000 # Max colors won't exceed n
        max_color = 0
        ordered_verts = []
        
        for v in verts:
            # Find smallest color not used by neighbors in R
            # Neighbors in R: R & ADJ_BITS[v]
            neigh_mask = R & ADJ_BITS[v]
            # We need to know which colors are used by neighbors.
            # This requires tracking which vertices have which color.
            
            # Efficient way:
            # Maintain a list of vertices per color.
            # Or just check against already colored neighbors.
            
            # Let's maintain a dict: color -> set of vertices
            # This is getting complicated. 
            # Simpler: 
            # For each v, check all previously colored vertices u in R.
            # If u is neighbor of v, color[u] is forbidden.
            
            # Since we process in order, we can just keep a list of (vertex, color).
            pass
        
        # Alternative Coloring Strategy for Tomita:
        # The algorithm typically works on a sequence of vertices.
        # Let's implement a standard greedy coloring that returns the number of colors.
        
        # We will use a simple greedy coloring.
        # Input: List of vertices V_R
        # Output: List of colors, and max_color
        
        V_R = verts
        n_v = len(V_R)
        if n_v == 0:
            return [], 0
        
        # Map vertex index to position in V_R for fast lookup?
        # Or just use a set of colored vertices.
        
        color_of = {}
        used_colors = [set() for _ in range(n_v)] # color i contains vertices with that color
        max_c = 0
        
        for v in V_R:
            # Determine forbidden colors
            forbidden = set()
            # Check neighbors in V_R that are already colored
            # We can just iterate through all colored vertices
            # This is O(N^2) per coloring step, which is O(N^3) total.
            # N=560, N^3 is 175M. Might be too slow in Python.
            
            # Optimization: 
            # When we add v, we only need to check neighbors.
            # But we don't have a fast way to get "colors of neighbors" without iterating.
            
            # Let's try a different approach: 
            # The upper bound is simply the number of colors.
            # We can use a simpler bound: the size of the maximum clique in the complement? No.
            # The number of colors in a greedy coloring is an upper bound on the independence number?
            # No, the number of colors in a proper coloring of the graph is an upper bound on the *clique* number?
            # Wait. 
            # We are looking for Independent Set in G.
            # This is Clique in G_bar.
            # Coloring G_bar gives an upper bound on the Clique number of G_bar, which is the Independence number of G.
            # So we should color the *complement* graph of the subgraph induced by R.
            
            # Complement of subgraph induced by R:
            # Two vertices are adjacent in G_bar if they are NOT adjacent in G.
            # So for v in R, neighbors in G_bar are R \ ({v} U N_G(v)).
            
            # Let's implement coloring on G_bar.
            
            # Reset
            color_of = {}
            max_c = 0
            # We want to assign colors to V_R such that adjacent in G_bar get different colors.
            # Adjacent in G_bar means NOT adjacent in G.
            
            for v in V_R:
                # Find smallest color c such that no neighbor in G_bar has color c.
                # Neighbor in G_bar of v: u in R such that u != v and u not in N_G(v).
                
                # Let's collect colors used by G_bar neighbors.
                forbidden_colors = set()
                for u, c in color_of.items():
                    # Check if u is a neighbor of v in G_bar
                    # u is in R (by construction of V_R)
                    # u != v
                    # u is NOT in N_G(v)
                    # N_G(v) is ADJ_BITS[v]
                    if u != v and not (ADJ_BITS[v] >> u) & 1:
                        forbidden_colors.add(c)
                
                # Find smallest c not in forbidden_colors
                c = 0
                while c in forbidden_colors:
                    c += 1
                color_of[v] = c
                if c > max_c:
                    max_c = c
            
            return V_R, max_c + 1 # 1-based count of colors

    def expand(R, P, size):
        """
        R: bitmask of vertices in the independent set so far? No, usually R is the set of vertices *available* to add.
        P: bitmask of vertices in the current independent set? 
        Tomita's algorithm:
        expand(R, P):
          if R is empty:
             if size(P) > best: update best
             return
          
          # Color the subgraph induced by R
          # Get an ordering of R based on colors
          # For each color class (from highest to lowest):
          #   For each vertex v in the color class:
          #     if size(P) + (number of colors remaining including this one) <= best: break
          #     expand(R \ (N(v) U {v}), P U {v}, size(P) + 1)
          #     R = R \ {v}
        """
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        if R == 0:
            if size > best_size:
                best_size = size
                best_set = P
            return
        
        # Get vertices in R
        verts = get_vertices(R)
        if not verts:
            return
            
        # Color the subgraph induced by R (in the complement graph G_bar)
        # We need the vertices ordered by color.
        # My color_subgraph returns (V_R, max_color). It doesn't return the ordering.
        # Let's modify it to return the ordering.
        
        # Re-implement coloring to return ordered list
        V_R = verts
        color_of = {}
        max_c = 0
        for v in V_R:
            forbidden_colors = set()
            for u, c in color_of.items():
                if u != v and not (ADJ_BITS[v] >> u) & 1:
                    forbidden_colors.add(c)
            c = 0
            while c in forbidden_colors:
                c += 1
            color_of[v] = c
            if c > max_c:
                max_c = c
        
        # Group vertices by color
        # Colors are 0..max_c
        # We want to process from highest color to lowest.
        # Vertices with color i form an independent set in G_bar, which is a clique in G?
        # No. 
        # In G_bar, color i is an independent set.
        # So in G, color i is a clique.
        # Wait, we want Independent Set in G.
        # So we want Clique in G_bar.
        # A clique in G_bar cannot have two vertices with the same color in a proper coloring.
        # So the size of the clique is at most the number of colors.
        
        # Tomita's branching:
        # We iterate through vertices in R.
        # We use the coloring to bound the search.
        
        # Let's create a list of vertices sorted by color descending.
        # Actually, Tomita's algorithm iterates through the vertices in the order given by the coloring.
        # Specifically, it processes vertices from the highest color class to the lowest.
        
        # Let's build the list:
        # vertices_by_color = [[] for _ in range(max_c + 1)]
        # for v in V_R:
        #    vertices_by_color[color_of[v]].append(v)
        
        # Then loop c from max_c down to 0:
        #   for v in vertices_by_color[c]:
        #      if size + c + 1 <= best_size: return (prune)
        #      # Branch on v
        #      new_P = P | (1 << v)
        #      new_R = R & ~(ADJ_BITS[v] | (1 << v)) # Remove v and its neighbors in G
        #      expand(new_R, new_P, size + 1)
        #      R &= ~(1 << v) # Remove v from R for subsequent iterations
        
        # Note: The pruning condition `size + c + 1 <= best_size` works because
        # any independent set formed from the remaining vertices (colors 0..c) 
        # can have size at most c+1 (since each color class is an independent set in G_bar, 
        # wait. 
        # In G_bar, vertices with the same color are NOT adjacent. 
        # So they CAN be in the same clique in G_bar? 
        # No, a clique requires all vertices to be adjacent. 
        # If they are not adjacent in G_bar, they cannot be in the same clique. 
        # So a clique in G_bar can contain at most one vertex from each color class. 
        # Thus, the size of the clique is at most the number of colors.
        
        vertices_by_color = [[] for _ in range(max_c + 1)]
        for v in V_R:
            vertices_by_color[color_of[v]].append(v)
        
        for c in range(max_c, -1, -1):
            if size + c + 1 <= best_size:
                return # Prune
            
            # Iterate through vertices in this color class
            # We need to iterate over a copy or handle removal carefully
            # Since we modify R, we should be careful.
            # Tomita's algorithm usually removes v from R after processing.
            
            # To avoid modifying the list while iterating, we can just iterate over the list 
            # and check if v is still in R.
            
            for v in vertices_by_color[c]:
                if time.time() > deadline:
                    return
                    
                # Check if v is still in R
                if not (R >> v) & 1:
                    continue
                
                # Branch on v
                # Add v to P
                new_P = P | (1 << v)
                
                # Remove v and its neighbors from R
                # Neighbors in G: ADJ_BITS[v]
                # We want to keep vertices in R that are NOT neighbors of v and NOT v.
                # So new_R = R & ~(ADJ_BITS[v] | (1 << v))
                
                new_R = R & ~(ADJ_BITS[v] | (1 << v))
                
                expand(new_R, new_P, size + 1)
                
                # Remove v from R
                R &= ~(1 << v)
                
                # Optional: If R becomes 0, break
                if R == 0:
                    break

    # Initial call
    # R is all vertices
    # P is empty
    # Size is 0
    initial_R = (1 << N_TRIPLES) - 1
    expand(initial_R, 0, 0)
    
    return best_set, best_size

def _heuristic_greedy(deadline):
    """
    Simple greedy heuristic to get a good initial lower bound.
    """
    remaining = set(range(N_TRIPLES))
    independent_set = set()
    
    # Order by degree (ascending)
    # Lower degree vertices are "less constraining"
    order = sorted(range(N_TRIPLES), key=lambda x: len(ADJ_LISTS[x]))
    
    for v in order:
        if time.time() > deadline:
            break
        if v in remaining:
            independent_set.add(v)
            # Remove neighbors
            for u in ADJ_LISTS[v]:
                remaining.discard(u)
            remaining.discard(v)
            
    return len(independent_set), independent_set

def construct():
    start = time.time()
    deadline = start + 115 # Safety margin
    
    # 1. Run heuristic to get initial bound
    h_size, h_set = _heuristic_greedy(deadline)
    
    # Convert h_set to bitmask
    h_mask = 0
    for v in h_set:
        h_mask |= (1 << v)
        
    # 2. Run Branch and Bound
    # Pass the initial best to B&B? 
    # My B&B implementation doesn't take an initial best. 
    # I should modify it to accept an initial best_set and best_size.
    
    # Let's modify the B&B call to use the heuristic result as the starting point.
    # I will wrap the B&B logic to allow initialization.
    
    best_set = h_mask
    best_size = h_size
    
    # Re-run B&B with initial best
    # To do this cleanly, I'll define the expand function inside construct or pass best_size/best_set as non-locals.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def get_vertices(mask):
        verts = []
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            verts.append(idx)
            mask ^= lsb
        return verts

    def expand(R, P, size):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        if R == 0:
            if size > best_size:
                best_size = size
                best_set = P
            return
        
        verts = get_vertices(R)
        if not verts:
            return
            
        # Coloring
        V_R = verts
        color_of = {}
        max_c = 0
        for v in V_R:
            forbidden_colors = set()
            for u, c in color_of.items():
                if u != v and not (ADJ_BITS[v] >> u) & 1:
                    forbidden_colors.add(c)
            c = 0
            while c in forbidden_colors:
                c += 1
            color_of[v] = c
            if c > max_c:
                max_c = c
        
        vertices_by_color = [[] for _ in range(max_c + 1)]
        for v in V_R:
            vertices_by_color[color_of[v]].append(v)
        
        for c in range(max_c, -1, -1):
            if size + c + 1 <= best_size:
                return
            
            for v in vertices_by_color[c]:
                if time.time() > deadline:
                    return
                    
                if not (R >> v) & 1:
                    continue
                
                new_P = P | (1 << v)
                new_R = R & ~(ADJ_BITS[v] | (1 << v))
                
                expand(new_R, new_P, size + 1)
                
                R &= ~(1 << v)
                
                if R == 0:
                    break

    initial_R = (1 << N_TRIPLES) - 1
    expand(initial_R, 0, 0)
    
    # Convert best_set bitmask to list of triples
    result = []
    for i in range(N_TRIPLES):
        if (best_set >> i) & 1:
            result.append(list(ALL_TRIPLES[i]))
            
    return result

random.seed(42)
