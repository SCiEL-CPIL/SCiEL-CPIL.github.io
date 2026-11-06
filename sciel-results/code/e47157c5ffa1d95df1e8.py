import sys

def construct():
    N = 256
    # 1. Build conflict graph (edges where divisibility holds)
    # We want to AVOID these edges. So we want Independent Set in this graph.
    # Which is Max Clique in the Complement Graph.
    # Let's build the "Non-Conflict" graph (Complement) directly for Max Clique.
    # non_adj[i] will have bits set for j where (i+1)*(j+1) % (i+1+j+1) != 0
    non_adj = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) != 0:
                non_adj[i] |= (1 << j)
                non_adj[j] |= (1 << i)
                
    # 2. Max Clique using Branch and Bound (Tomita style)
    best_size = 0
    best_set = 0
    
    def popcount(x):
        return bin(x).count('1')
        
    # Greedy coloring of the subgraph induced by 'mask' in the NON_ADJ graph
    # This gives an upper bound on the size of the Max Clique in that subgraph.
    def greedy_color_bound(mask):
        if mask == 0:
            return 0
        
        colors = 0
        remaining = mask
        # We assign colors to vertices in 'remaining'
        # The number of colors is the upper bound.
        while remaining:
            colors += 1
            # Select an independent set in the NON_ADJ graph? 
            # No, coloring assigns same color to non-adjacent vertices in the graph being colored.
            # We are coloring the NON_ADJ graph.
            # So vertices with same color must be non-adjacent in NON_ADJ.
            # Which means they MUST be adjacent in ADJ (conflict graph).
            # This seems counter-intuitive.
            
            # Let's re-verify.
            # We want Max Clique in G (where G is non-conflict graph).
            # Upper bound for Max Clique in G is the Chromatic Number of G? No.
            # Upper bound for Max Clique in G is the Chromatic Number of the Complement of G?
            # Yes. $\chi(\bar{G}) \ge \omega(G)$.
            # So we need to greedily color the **Complement** of the current subgraph of G.
            # The current subgraph is defined by `mask` in `non_adj`.
            # The complement of this subgraph has edges where `non_adj` has NO edge.
            # So for a vertex $v$, its neighbors in the complement are `mask & ~non_adj[v] & ~(1<<v)`.
            
            # Let's implement coloring of the complement.
            # Step 1: Find a set of vertices that are mutually adjacent in the Complement.
            # This is a Clique in the Complement, which is an Independent Set in G (non_adj).
            # So we need to find an Independent Set in the subgraph of G defined by `remaining`.
            # Greedy Independent Set:
            # Pick a vertex $v$. Remove $v$ and its neighbors in G (non_adj).
            # The remaining vertices are not adjacent to $v$ in G, so they can share the color?
            # No.
            # If we color the Complement $\bar{G}$:
            # Vertices with same color in $\bar{G}$ must be non-adjacent in $\bar{G}$.
            # Non-adjacent in $\bar{G}$ means Adjacent in $G$.
            # So a color class in $\bar{G}$ is a Clique in $G$.
            # This doesn't help bound the Clique in $G$ directly by the number of colors?
            # Wait. $\chi(\bar{G})$ is the number of colors.
            # $\omega(G) \le \chi(\bar{G})$.
            # So if we find a valid coloring of $\bar{G}$ with $k$ colors, the max clique in $G$ is at most $k$.
            
            # So, to color $\bar{G}$:
            # We need to assign colors such that adjacent vertices in $\bar{G}$ have different colors.
            # Adjacent in $\bar{G}$ means NOT adjacent in $G$.
            # So, if $u, v$ are NOT connected in $G$, they must have different colors.
            # This is equivalent to: if $u, v$ ARE connected in $G$, they CAN have the same color.
            # So a color class is a set of vertices that are pairwise connected in $G$? No.
            # If $u, v$ are connected in $G$, they are NOT adjacent in $\bar{G}$, so they CAN have the same color.
            # If $u, v$ are NOT connected in $G$, they ARE adjacent in $\bar{G}$, so they MUST have different colors.
            # So, a color class in $\bar{G}$ consists of vertices that are pairwise NOT connected in $G$?
            # No. If $u, v$ are in the same color class, they are NOT adjacent in $\bar{G}$.
            # Not adjacent in $\bar{G}$ means Adjacent in $G$.
            # So a color class in $\bar{G}$ is a **Clique** in $G$.
            
            # So, to greedily color $\bar{G}$:
            # 1. Find a Maximal Clique in $G$ (induced by `remaining`).
            # 2. Assign current color to all vertices in this Clique.
            # 3. Remove these vertices from `remaining`.
            # 4. Repeat.
            
            # Finding a Maximal Clique in G greedily:
            # Pick a vertex $v$. 
            # Add $v$ to the clique.
            # Remove $v$ and all its NON-neighbors in G from `remaining_for_clique`.
            # Why non-neighbors? Because in a clique, everyone must be connected to everyone.
            # If $u$ is not connected to $v$ in G, $u$ cannot be in the clique with $v$.
            # So we keep only neighbors of $v$ in G.
            
            current_clique = 0
            rem_for_clique = remaining
            while rem_for_clique:
                # Pick a vertex, e.g., highest degree in rem_for_clique within G?
                # Or just lowest index.
                v_bit = rem_for_clique & -rem_for_clique
                v_idx = v_bit.bit_length() - 1
                current_clique |= v_bit
                # Keep only neighbors of v in G
                # Neighbors in G are non_adj[v_idx]
                rem_for_clique &= non_adj[v_idx]
                # Remove v itself
                rem_for_clique &= ~v_bit
                
            remaining &= ~current_clique
            
        return colors

    def branch(candidates, selected, size):
        nonlocal best_size, best_set
        
        if candidates == 0:
            if size > best_size:
                best_size = size
                best_set = selected
            return
            
        # Upper bound
        # Simple: size + popcount(candidates)
        if size + popcount(candidates) <= best_size:
            return
            
        # Better bound: size + greedy_color_bound(candidates)
        ub = size + greedy_color_bound(candidates)
        if ub <= best_size:
            return
            
        # Choose a vertex to branch on
        # Heuristic: Pick vertex with highest degree in the induced subgraph of 'candidates' in G (non_adj)
        # This helps expand the clique faster.
        max_deg = -1
        v_idx = -1
        c = candidates
        while c:
            v_bit = c & -c
            idx = v_bit.bit_length() - 1
            deg = popcount(non_adj[idx] & candidates)
            if deg > max_deg:
                max_deg = deg
                v_idx = idx
            c &= c - 1
            
        if v_idx == -1:
            return
            
        v_bit = 1 << v_idx
        
        # Branch 1: Include v_idx
        # If we include v, we can only include its neighbors in G.
        new_candidates_include = candidates & non_adj[v_idx]
        branch(new_candidates_include, selected | v_bit, size + 1)
        
        # Branch 2: Exclude v_idx
        new_candidates_exclude = candidates & ~v_bit
        branch(new_candidates_exclude, selected, size)

    all_nodes = (1 << N) - 1
    branch(all_nodes, 0, 0)
    
    result = []
    for i in range(N):
        if (best_set >> i) & 1:
            result.append(i + 1)
            
    return result
