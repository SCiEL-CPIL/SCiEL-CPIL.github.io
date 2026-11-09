"""Maximum Weight Independent Set via Branch-and-Bound for the pair-sum divisibility task."""
import time
import math

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s wall-clock kill
    n = 256
    numbers = list(range(1, n + 1))
    idx_to_num = {i: numbers[i] for i in range(n)}
    num_to_idx = {numbers[i]: i for i in range(n)}

    # Build conflict graph: edge (i,j) if a*b % (a+b) == 0
    # We want maximum independent set: no two selected have an edge.
    # adj[i] = set of j != i such that (a_i * a_j) % (a_i + a_j) == 0
    adj = [set() for _ in range(n)]
    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]
            if (a * b) % (a + b) == 0:
                adj[i].add(j)
                adj[j].add(i)

    # Degree of each vertex
    deg = [len(adj[i]) for i in range(n)]

    # Greedy coloring for upper bound (DSATUR-style ordering for coloring)
    # We color the graph so that each color class is an independent set.
    # Upper bound for MIS = number of colors (since each color class can contribute at most 1 to an independent set... wait no)
    # Actually: in a proper coloring, each color class is an independent set. 
    # For MIS, the upper bound is the number of colors because we can pick at most one vertex per color? No that's wrong.
    # 
    # Correction: A proper coloring partitions vertices into independent sets (color classes).
    # Any independent set can contain at most |color_class| vertices from each color class? No.
    # An independent set can contain multiple vertices from the same color class only if they are not adjacent.
    # But in a proper coloring, all vertices in the same color class are NOT adjacent to each other.
    # So an independent set could potentially contain ALL vertices of a single color class.
    # 
    # The correct upper bound: the chromatic number χ(G) satisfies χ(G) >= α(G) where α(G) is the independence number.
    # Wait no: α(G) >= n/χ(G) is not an upper bound on α(G).
    # Actually: α(G) * χ(G) >= n, so α(G) >= n/χ(G). That's a lower bound.
    # 
    # For upper bound on MIS size: we use the fact that if we have a k-coloring, then any independent set 
    # can have at most... hmm, actually the standard bound used in Tomita's algorithm is:
    # The coloring gives an upper bound: if the graph is k-colored, then α(G) <= k. 
    # NO! That's wrong. α(G) can be much larger than k.
    #
    # Let me reconsider. In Tomita's maximum clique algorithm (which is equivalent to MIS on complement),
    # the coloring bound works because: in a proper coloring of the graph, a clique can contain at most one vertex per color.
    # For MIS: in a proper coloring of the COMPLEMENT graph, an independent set in G is a clique in complement,
    # so it can contain at most one vertex per color of the complement.
    #
    # Alternatively: color the original graph G. An independent set in G can contain ALL vertices of any single color class
    # (since they're mutually non-adjacent). So the coloring of G doesn't directly give an upper bound on α(G).
    #
    # The correct approach for MIS upper bound: color the COMPLEMENT graph. Then α(G) = ω(complement) <= number of colors of complement.
    #
    # But building the complement of a 256-vertex graph is fine. Let me use the standard approach:
    # For maximum clique on graph H (where H = complement of G), use coloring of H for upper bound.
    # MIS(G) = MaxClique(complement(G)).
    #
    # Let me redefine: work with the complement graph for maximum clique.
    # comp_adj[i] = set of j where (i,j) is NOT an edge in G, i.e., (a_i * a_j) % (a_i + a_j) != 0.
    
    # Actually, let me just work directly with MIS on G using a different bound.
    # 
    # Standard approach for MIS with branch and bound:
    # - Use a greedy coloring of the remaining subgraph to get an upper bound.
    # - The bound: if the remaining subgraph on vertex set S can be colored with k colors,
    #   then the maximum independent set within S has size at most... 
    #
    # I think the cleanest is to convert to Max Clique on the complement and use Tomita's algorithm.
    # Let me do that.

    # Complement adjacency: comp_adj[i] contains j if i and j are NOT adjacent in G (and i != j)
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and j not in adj[i]:
                comp_adj[i].add(j)

    # Now find Max Clique in comp_adj.
    # Tomita's algorithm with branch and bound.
    
    best_clique = []
    best_size = 0

    # Greedy coloring for upper bound on clique size
    def greedy_color(vertices):
        """
        Color the subgraph induced by vertices (in the complement graph).
        Returns (colors, color_of) where colors[v] is the color index for vertex v.
        The number of colors used is an upper bound on the clique size in this subgraph.
        We use a simple sequential coloring with DSATUR-like ordering.
        """
        # Order vertices by degree in descending order for better coloring
        vlist = sorted(vertices, key=lambda v: -len(comp_adj[v] & set(vertices)))
        color_of = {}
        next_color = 0
        for v in vlist:
            # Find the smallest color not used by any neighbor of v in vlist
            used = set()
            for u in comp_adj[v]:
                if u in color_of:
                    used.add(color_of[u])
            c = 0
            while c in used:
                c += 1
            color_of[v] = c
            next_color = max(next_color, c + 1)
        return next_color, color_of

    # Branch and bound for maximum clique
    def branch_and_bound(candidates, current_clique, deadline_check):
        nonlocal best_clique, best_size
        
        if time.time() > deadline_check:
            return
        
        if not candidates:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = list(current_clique)
            return
        
        # Get upper bound via greedy coloring
        num_colors, color_of = greedy_color(candidates)
        
        if len(current_clique) + num_colors <= best_size:
            return
        
        # Order candidates by color (process highest color first for better pruning)
        # Group by color
        max_c = max(color_of.values()) if color_of else 0
        # Process vertices in order of decreasing color
        ordered = sorted(candidates, key=lambda v: -color_of[v])
        
        # Branch: try adding each vertex
        for v in ordered:
            if time.time() > deadline_check:
                return
            
            # Upper bound: current + color_of[v] (since v and all remaining vertices 
            # with color >= color_of[v] can form at most color_of[v] clique vertices)
            if len(current_clique) + color_of[v] <= best_size:
                break  # All subsequent vertices have <= color, so we can stop
            
            # Add v to clique
            current_clique.append(v)
            
            # New candidates: those in candidates that are adjacent to v in comp_adj
            # and come after v in our ordering (to avoid duplicates)
            new_candidates = [u for u in ordered if u != v and u in comp_adj[v]]
            # Actually we need to be more careful. Let's use the standard approach.
            
            if len(current_clique) + len(new_candidates) > best_size:
                # Filter new_candidates by color bound
                if new_candidates:
                    nc_num_colors, nc_color_of = greedy_color(new_candidates)
                    if len(current_clique) + nc_num_colors > best_size:
                        branch_and_bound(new_candidates, current_clique, deadline_check)
                else:
                    if len(current_clique) > best_size:
                        best_size = len(current_clique)
                        best_clique = list(current_clique)
            else:
                if len(current_clique) > best_size:
                    best_size = len(current_clique)
                    best_clique = list(current_clique)
            
            current_clique.pop()
    
    # Initialize with a greedy clique
    def greedy_clique():
        """Find a greedy clique for initial lower bound."""
        clique = []
        remaining = set(range(n))
        while remaining:
            # Pick vertex with max degree in remaining
            v = max(remaining, key=lambda x: len(comp_adj[x] & remaining))
            clique.append(v)
            remaining = remaining & comp_adj[v]
        return clique
    
    initial = greedy_clique()
    best_size = len(initial)
    best_clique = list(initial)
    
    # Now run branch and bound
    all_vertices = list(range(n))
    branch_and_bound(all_vertices, [], deadline)
    
    # Convert clique indices back to numbers
    result = sorted([idx_to_num[i] for i in best_clique])
    
    # Verify: for every distinct a,b in result, (a*b) % (a+b) != 0
    # (This is guaranteed by construction since we found a clique in the complement,
    #  which means no two are adjacent in G, i.e., no pair sum divides pair product)
    
    if not result:
        result = [1]
    
    return result

# EVOLVE-BLOCK-END
