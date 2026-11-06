import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N = 256
    max_sum = 2 * N
    squares = set()
    k = 2
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
    
    # 1. Identify valid nodes (no self-loop)
    valid_nodes = []
    node_to_idx = {}
    for i in range(1, N + 1):
        if 2 * i not in squares:
            valid_nodes.append(i)
            node_to_idx[i] = len(valid_nodes)
            
    M = len(valid_nodes)
    
    # 2. Build complement adjacency (H')
    # H edge: u+v is square.
    # H' edge: u+v is NOT square.
    # We want Max Clique in H'.
    
    # Precompute square sums for valid nodes
    # adj[i] is bitmask of j such that (valid_nodes[i] + valid_nodes[j]) is NOT square
    adj = [0] * M
    
    # First, compute conflict mask (H edges)
    conflict_mask = [0] * M
    for i in range(M):
        ui = valid_nodes[i]
        for j in range(i + 1, M):
            uj = valid_nodes[j]
            if ui + uj in squares:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
                
    # Complement mask
    all_mask = (1 << M) - 1
    for i in range(M):
        # Neighbors in H' are all nodes except i and its neighbors in H
        # adj[i] = all_mask ^ (1 << i) ^ conflict_mask[i]
        adj[i] = all_mask ^ (1 << i) ^ conflict_mask[i]

    # 3. Branch and Bound for Max Clique
    best_clique_size = 0
    best_clique_nodes = []
    
    # Order vertices initially?
    # Greedy coloring helps.
    
    def greedy_coloring(candidates_mask):
        # Returns list of (vertex, color)
        # We want to color the subgraph induced by candidates in H'
        # H' is dense, so greedy coloring might use many colors?
        # Wait, we want upper bound on clique size.
        # Chromatic number of H'[C] is an upper bound on clique number of H'[C].
        # Greedy coloring gives an upper bound on chromatic number? No, it gives a specific coloring, so the number of colors used is an upper bound on the chromatic number?
        # No, chromatic number <= colors used.
        # So colors used is an UPPER bound on chromatic number?
        # Yes. And Clique Number <= Chromatic Number.
        # So colors used is an upper bound on Clique Number.
        
        uncolored = candidates_mask
        colors = {}
        color_count = 0
        
        # Standard greedy: pick vertex, assign smallest color not used by neighbors
        # To get a tight bound, we should order vertices.
        # DSATUR is better but complex. Simple greedy with static ordering might be weak.
        # Let's try ordering by degree in H' descending?
        # Or just iterate through bits.
        
        # To make it faster and potentially tighter, we can sort vertices by degree in the induced subgraph?
        # That's expensive per node.
        # Let's just do a simple greedy pass.
        
        # Actually, for B&B, we often sort the candidates by the color assigned in the *global* or *previous* step?
        # Let's just assign colors 0..k-1.
        
        while uncolored:
            # Pick a vertex
            # To improve bound, we should pick vertices that are "hard" to color?
            # Let's just pick the lowest index for simplicity, or highest degree.
            # Let's pick lowest index.
            v = (uncolored & -uncolored).bit_length() - 1
            uncolored ^= (1 << v)
            
            # Find available colors
            used_colors = set()
            neighbors = adj[v] & candidates_mask
            # Check neighbors already colored
            n_mask = neighbors
            while n_mask:
                u = (n_mask & -n_mask).bit_length() - 1
                n_mask ^= (1 << u)
                if u in colors:
                    used_colors.add(colors[u])
            
            # Assign smallest available color
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            if c + 1 > color_count:
                color_count = c + 1
                
        return colors, color_count

    # B&B state
    # R: current clique (list of vertices)
    # P: candidates (mask)
    
    def expand(R, P):
        nonlocal best_clique_size, best_clique_nodes
        
        if P == 0:
            if len(R) > best_clique_size:
                best_clique_size = len(R)
                best_clique_nodes = R[:]
            return
        
        # Check time
        if time.time() > deadline:
            return
            
        # Color P
        colors, num_colors = greedy_coloring(P)
        
        # If max possible size (len(R) + num_colors) <= best, prune
        if len(R) + num_colors <= best_clique_size:
            return
            
        # Order vertices in P by color descending
        # Vertices with higher color have "more" potential?
        # Actually, in Max Clique B&B, we usually sort by color ascending?
        # Let's check Tomita's algorithm.
        # It sorts by color descending.
        # Why? If a vertex has color k, it can potentially extend the clique by k.
        # If we process high color first, we might find a large clique quickly, raising the lower bound.
        
        p_list = [v for v in range(M) if (P >> v) & 1]
        p_list.sort(key=lambda x: colors[x], reverse=True)
        
        for v in p_list:
            # Prune: if len(R) + colors[v] <= best_clique_size, 
            # then even if we add v and all its "color class" neighbors (which is an upper bound on what we can add from this branch),
            # we won't beat the best.
            # Wait, colors[v] is the color assigned to v.
            # The number of vertices with color >= colors[v] in P is an upper bound on the clique size we can add from P?
            # No. The coloring is a proper coloring.
            # A clique can contain at most one vertex of each color.
            # So the size of the clique in P is at most the number of distinct colors used in P.
            # If we remove vertices with color < colors[v], the remaining graph is colored with colors colors[v]..max.
            # The number of such colors is max_color - colors[v] + 1?
            # Not exactly.
            
            # Standard Tomita:
            # The upper bound for the branch starting with v is len(R) + (number of vertices in P with color >= colors[v]).
            # Let's call this count bound.
            
            # Let's just use the simple bound: len(R) + colors[v] <= best?
            # This is valid if colors are 1-indexed and represent the "level".
            # If v has color c, it is the c-th color.
            # Can we add a clique of size c from the subgraph induced by vertices with color >= c?
            # Yes, because any clique in that subgraph uses distinct colors from c..max.
            # So the size of the clique is at most the number of colors in {c, ..., max}.
            # Which is max_color - c + 1.
            
            # Let's compute the bound properly.
            # We need the count of vertices in P with color >= colors[v].
            # This is expensive to compute for every v if we do it naively.
            # But we can precompute counts?
            # Or just use the global num_colors as a loose bound for all v?
            # That's too loose.
            
            # Let's stick to the simple bound: len(R) + colors[v] <= best?
            # If colors are 0-indexed, and we have colors 0..k-1.
            # If v has color c, the clique can have at most k-c vertices?
            # No, it can have at most one vertex of each color.
            # If we restrict to vertices with color >= c, the number of available colors is k-c.
            # So the clique size added is at most k-c.
            # So bound is len(R) + (num_colors - colors[v]).
            
            bound = len(R) + (num_colors - colors[v])
            if bound <= best_clique_size:
                continue # Since p_list is sorted by color desc, subsequent v will have smaller colors, so larger bound?
                # Wait. If color is 5, bound is len(R) + (k-5).
                # If color is 4, bound is len(R) + (k-4).
                # So bound INCREASES as color DECREASES.
                # So if we sort by color DESC, the bound starts SMALL and gets LARGER.
                # So we CANNOT break. We must continue.
                
            # Add v to R
            R.append(v)
            
            # New P is neighbors of v in H' intersected with current P
            new_P = P & adj[v]
            
            # Remove v from P for next iterations?
            # In the loop, we iterate over p_list.
            # We should remove v from P so it's not considered in subsequent iterations of THIS level?
            # No, P is the set of candidates for the NEXT level.
            # For the current level, we are choosing one v to add to R.
            # The other vertices in P are still candidates for OTHER branches (where we don't pick v).
            # But in the recursion, we pass `new_P`.
            # After recursion, we remove v from P for the next v in the loop?
            # Yes, because if we don't pick v, we can still pick other vertices.
            # But we must ensure we don't pick v again.
            # So we should do P ^= (1 << v) at the end of the loop iteration?
            # Or just use a local variable for P in the loop?
            
            expand(R, new_P)
            
            R.pop()
            P ^= (1 << v) # Remove v from candidates for subsequent branches at this level

    # Initial call
    # Order initial vertices?
    # Let's just start with all valid nodes.
    initial_P = (1 << M) - 1
    expand([], initial_P)
    
    # Map back to original values
    result = [valid_nodes[i] for i in best_clique_nodes]
    
    return sorted(result)
