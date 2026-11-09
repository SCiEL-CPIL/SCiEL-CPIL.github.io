import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    N = 512
    
    # 1. Build adjacency bitmasks
    # adj[i] is a bitmask of nodes compatible with i (0-indexed i corresponds to value i+1)
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) // gcd(i, j) <= N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 2. Greedy Initial Solution
    # Sort nodes by degree (popcount of adj mask)
    nodes = list(range(1, N + 1))
    nodes.sort(key=lambda x: adj[x].bit_count(), reverse=True)
    
    best_clique = []
    candidates_mask = (1 << (N + 1)) - 2 # All bits 1..N set
    for u in nodes:
        if candidates_mask & (1 << u):
            best_clique.append(u)
            candidates_mask &= adj[u]
            
    best_size = len(best_clique)
    
    # 3. Tomita B&B with Coloring
    # We use 0-indexed internal logic for bit operations if possible, 
    # but the values are 1..512.
    # Let's stick to 1..512 for clarity in the clique result, 
    # but bit operations work on indices.
    
    def color_sort(candidates_mask):
        """
        Greedy coloring of the subgraph induced by candidates_mask.
        Returns a list of (vertex, color) pairs.
        The list is ordered such that vertices with higher colors are processed first?
        Actually, Tomita algorithm usually processes vertices in the order of the coloring,
        where the color of a vertex is an upper bound on the clique size in its neighborhood.
        We want to prune if current_size + color <= best_size.
        """
        if candidates_mask == 0:
            return []
        
        colors = []
        remaining = candidates_mask
        color_num = 0
        
        while remaining:
            color_num += 1
            # Select a maximal independent set for this color
            # Greedy: pick vertex with highest degree in remaining?
            # Or just iterate?
            # To make the bound tight, we should order vertices by degree.
            
            # Let's pick vertices one by one for this color class
            # A vertex v can take color c if it is not adjacent to any other vertex in color class c.
            # This is equivalent to finding a maximal independent set.
            
            # For simplicity and speed in Python, we can just assign colors sequentially
            # based on a sorted order.
            # Standard greedy coloring:
            # 1. Sort vertices in remaining by degree (descending).
            # 2. Iterate through them. If a vertex is not adjacent to any previously colored vertex in the current color class, assign it this color.
            
            # Getting degrees of remaining vertices:
            # We can extract vertices from mask.
            verts = []
            m = remaining
            while m:
                # Get lowest set bit
                lsb = m & -m
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            
            # Sort by degree in the induced subgraph
            # Degree of v in remaining is popcount(adj[v] & remaining)
            verts.sort(key=lambda v: (adj[v] & remaining).bit_count(), reverse=True)
            
            color_class = []
            new_remaining = remaining
            
            for v in verts:
                # Check if v is compatible with all in color_class (i.e., not adjacent)
                # Wait, in Clique problem, we color the *complement*? No.
                # In Max Clique, we color the graph itself.
                # The color number of a vertex v in a proper coloring of G is an upper bound on the size of a clique in G containing v?
                # No, the color number is an upper bound on the size of a clique in the *neighborhood* of v?
                # Actually, the theorem is: If we color the graph G, the size of any clique in G is at most the number of colors.
                # More specifically, in the recursive step, we have a set of candidates P.
                # We color P. If we pick a vertex v, the max clique in N(v) is bounded by the color of v in the coloring of P?
                # Yes. The color assigned to v is an upper bound on the clique size in the subgraph induced by P \ {v} that is adjacent to v?
                # Actually, simpler: The color of v in the greedy coloring of P is an upper bound on the clique size in the subgraph induced by the vertices in P that are adjacent to v and have color <= color(v)?
                
                # Let's use the standard Tomita `color_sort` which returns vertices in order of increasing color,
                # and we process them in reverse.
                
                # Standard Greedy Coloring for Tomita:
                # We want to assign colors to vertices in P.
                # A vertex v gets color c if it is not adjacent to any vertex already assigned color c.
                # This ensures that vertices with the same color form an independent set.
                # Therefore, a clique can contain at most one vertex of each color.
                # So the number of colors is an upper bound on the clique size in P.
                
                # We iterate through vertices.
                # To get a tight bound, we should process vertices with high degree first?
                # Actually, the order of coloring affects the number of colors.
                # Smaller number of colors = tighter bound.
                # Greedy coloring with vertex ordering by degree (descending) usually yields good bounds.
                
                pass # Logic below

        # Re-implementing color_sort properly
        def color_sort_impl(mask):
            if mask == 0:
                return []
            
            verts = []
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            
            # Order by degree in the subgraph induced by mask
            verts.sort(key=lambda v: (adj[v] & mask).bit_count(), reverse=True)
            
            colors = [] # list of (v, color)
            used_colors = [0] * (N + 1) # used_colors[c] is mask of vertices assigned color c
            
            for v in verts:
                # Find the smallest color c such that v is not adjacent to any vertex in used_colors[c]
                # i.e., adj[v] & used_colors[c] == 0
                c = 1
                while c <= len(used_colors) - 1:
                    if (adj[v] & used_colors[c]) == 0:
                        break
                    c += 1
                
                # Assign color c
                if c == len(used_colors):
                    used_colors.append(0)
                used_colors[c] |= (1 << v)
                colors.append((v, c))
            
            # Tomita processes vertices in reverse order of color?
            # Actually, the bound for vertex v is its color c.
            # If current_clique_size + c <= best_size, we can prune.
            # We should process vertices with *highest* potential first?
            # No, we iterate through the list.
            # The list `colors` is ordered by the order we assigned colors (which was by degree).
            # We need to return the list such that we can iterate and prune.
            # Standard Tomita:
            # The vertices are ordered v1, v2, ... vk.
            # color(vi) is the color.
            # We process vi. If len(clique) + color(vi) <= best_size, return.
            # This requires that color(vi) is a valid upper bound for the clique in the subgraph induced by {v1, ..., vi} adjacent to vi?
            # Actually, the standard implementation returns the list of vertices sorted by color, and we iterate backwards?
            # Or forwards?
            # Let's check: If we have colors 1, 1, 2, 3.
            # Max clique size <= 3.
            # If we pick the vertex with color 3, the max clique in its neighborhood is <= 2?
            # No, the bound is: The max clique in the subgraph induced by the remaining candidates (including v) is <= color(v).
            # So if we process v, and current_size + color(v) <= best, we prune.
            # To maximize pruning, we should process vertices with *small* color first?
            # If we process a vertex with small color, and it doesn't prune, we add it.
            # If we process a vertex with large color, the bound is loose.
            # However, Tomita's algorithm typically sorts the candidates by color and processes them in that order (often reverse to find large cliques faster, or forward for pruning?).
            # Actually, the pruning condition `len(clique) + color(v) <= best_size` is valid for ANY v in the set.
            # If we process v, and the condition holds, we can skip v entirely.
            # So we should process vertices with *smallest* color first to trigger pruning?
            # Or does the order of the list affect the *validity* of the bound?
            # The bound `color(v)` is valid for the subgraph induced by the vertices in the list *up to and including v* (in the specific ordering used for coloring).
            # So we must respect the order.
            
            # Let's return the list of (v, color) in the order they were added (sorted by degree).
            # Then in the B&B, we iterate through this list.
            # If we pick v, the new candidates are neighbors of v that come *after* v in this list?
            # No, the candidates for the next step are `candidates & adj[v]`.
            # But we must ensure we don't double count or miss vertices.
            # Standard Tomita:
            # `expand(clique, candidates)`:
            #   `color_sort(candidates)` -> `P` (list of vertices) and `colors` (list of ints).
            #   For i from len(P)-1 down to 0:
            #      v = P[i]
            #      if len(clique) + colors[i] <= best_size: return
            #      clique.append(v)
            #      expand(clique, candidates & adj[v])
            #      clique.pop()
            #      candidates &= ~(1 << v)
            
            return colors

        return color_sort_impl(mask)

    # Correcting the color_sort logic to match Tomita
    def get_color_sorted_list(mask):
        if mask == 0:
            return [], []
        
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        
        # Sort by degree in induced subgraph (descending)
        verts.sort(key=lambda v: (adj[v] & mask).bit_count(), reverse=True)
        
        P = []
        colors = []
        # used_colors[c] = mask of vertices assigned color c
        used_colors = [] 
        
        for v in verts:
            # Find min color c such that v is independent of used_colors[c]
            c = 0
            while c < len(used_colors):
                if (adj[v] & used_colors[c]) == 0:
                    break
                c += 1
            
            if c == len(used_colors):
                used_colors.append(0)
            used_colors[c] |= (1 << v)
            
            P.append(v)
            colors.append(c + 1) # 1-indexed colors
            
        return P, colors

    def expand(clique, candidates_mask):
        nonlocal best_size, best_clique, deadline
        
        if time.time() > deadline:
            return
            
        if candidates_mask == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
        
        # Pruning
        if len(clique) + candidates_mask.bit_count() <= best_size:
            return
            
        P, colors = get_color_sorted_list(candidates_mask)
        
        # Iterate in reverse order (highest color first? No, P is sorted by degree.
        # The bound colors[i] is valid for the subgraph induced by P[0..i]?
        # Actually, the standard Tomita iterates from the end of the list.
        # Why? Because the vertices at the end of the list (processed last in coloring)
        # have the "tightest" bounds relative to the remaining set?
        # Let's follow the standard: Iterate i from len(P)-1 down to 0.
        
        for i in range(len(P) - 1, -1, -1):
            if time.time() > deadline:
                return
            
            v = P[i]
            # The color of v is an upper bound on the clique size in the subgraph induced by P[0..i] (including v)
            # Wait, the coloring was done on the whole set.
            # The property is: The max clique in the subgraph induced by {v_1, ..., v_i} is at most colors[i-1]?
            # Let's check the specific bound:
            # In Tomita's paper, `color_sort` produces an ordering where `color(v_i)` is an upper bound on the clique size in the subgraph induced by `v_1, ..., v_i`.
            # So if we are at index `i` (0-indexed), the bound is `colors[i]`.
            # If `len(clique) + colors[i] <= best_size`, we can prune the rest of the loop?
            # No, we can prune this specific branch?
            # Actually, if we are iterating backwards, and we encounter a v where the bound is too low,
            # we can break out of the loop entirely because for all j < i, colors[j] <= colors[i]?
            # Is colors non-decreasing?
            # In the greedy coloring algorithm above, colors are assigned 1, 1, 2, 1, 3...
            # They are not necessarily sorted.
            # However, the *list* P is sorted by degree.
            # The standard Tomita implementation sorts the vertices by color, or uses the color order.
            
            # Let's stick to a safer pruning:
            if len(clique) + colors[i] <= best_size:
                # Since we are going backwards, and the bound for previous vertices (lower index)
                # is typically <= current bound? Not necessarily.
                # But if the bound for v is too small, we skip v.
                # Can we break?
                # If the coloring ensures that colors[i] is an upper bound for the clique in P[0..i],
                # and we are processing P[i], and we prune, we should continue to P[i-1].
                # We cannot break unless we know colors[j] <= colors[i] for all j < i.
                # To be safe, we just continue.
                continue
            
            # Branch: Include v
            clique.append(v)
            # New candidates: neighbors of v in the current candidate set
            # But we must only consider vertices that are "before" v in the ordering to avoid duplicates?
            # In Max Clique, if we process vertices in a fixed order, we only add neighbors that come *before* v in the order?
            # Or *after*?
            # If we pick v, the new candidates must be adjacent to v.
            # To ensure we find each clique exactly once, we restrict the new candidates to the subset of P that comes *after* v in the original list?
            # No, usually we restrict to the subset of candidates that are *before* v in the sorted list?
            # Let's use the mask approach:
            # new_candidates = candidates_mask & adj[v]
            # But we should mask out vertices that are "after" v in P to avoid symmetry?
            # Actually, the standard recursive step is:
            # expand(clique + [v], candidates & adj[v])
            # Where candidates is the set of vertices *after* v in the current list?
            
            # Let's refine the loop structure.
            # The list P is the current candidate set.
            # We pick v = P[i].
            # The new candidate set should be { u in P | u is adjacent to v and index(u) < i }?
            # Or index(u) > i?
            # If we iterate i from len-1 down to 0:
            # When we pick P[i], the remaining candidates for the next level are P[0..i-1] that are adjacent to P[i].
            
            # So we need a mask of P[0..i-1].
            # Let's precompute prefix masks? Or just use the fact that P is a list.
            
            # Simpler:
            # new_cand_mask = 0
            # for j in range(i):
            #    if P[j] in adj[v]:
            #        new_cand_mask |= (1 << P[j])
            
            # This is O(N) per step. N=512. 512*512 is fine.
            
            new_cand_mask = 0
            for j in range(i):
                u = P[j]
                if adj[v] & (1 << u):
                    new_cand_mask |= (1 << u)
            
            expand(clique, new_cand_mask)
            clique.pop()
            
            # Remove v from candidates for the next iteration of the loop?
            # The loop variable i handles the "next" vertex.
            # We don't need to update candidates_mask explicitly if we use the P list indices.

    # Start B&B
    # Initial candidates: All nodes
    # But we should start with the greedy solution?
    # The B&B will find the max.
    # To speed up, we can start with the greedy clique as the initial "clique" and
    # restrict candidates to neighbors of all in greedy clique?
    # No, that restricts us to cliques *containing* the greedy clique.
    # We want the global max.
    # So we start with empty clique and all candidates.
    
    # However, we can use the greedy size as `best_size`.
    
    # One optimization: Order the initial candidates by degree.
    # The `get_color_sorted_list` does this internally.
    
    expand([], (1 << (N + 1)) - 2)
    
    return sorted(best_clique)

# EVOLVE-BLOCK-END
