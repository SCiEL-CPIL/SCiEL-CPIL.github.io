"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def construct():
    n = 20
    colors = 20
    
    # We want to maximize edges in a graph on n vertices with edge colors in 0..colors-1
    # such that every 4-cycle is rainbow (all 4 edges have distinct colors).
    
    # Strategy: 
    # 1. Start with a complete graph K_n (max edges = n*(n-1)/2 = 190)
    # 2. Assign colors such that no 4-cycle has repeated colors.
    # 3. If we can't satisfy constraints, remove edges or recolor.
    
    # Key insight: A 4-cycle in a complete graph is any set of 4 distinct vertices {a,b,c,d} 
    # with edges (a,b), (b,c), (c,d), (d,a) forming a cycle. But in K_n, there are many 4-cycles.
    # Actually, a 4-cycle is a cycle of length 4. In K_n, for any 4 distinct vertices, 
    # there are 3 different 4-cycles (the 3 ways to arrange them in a cycle up to rotation/reflection).
    # Wait, for 4 vertices a,b,c,d, the 4-cycles are:
    # a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
    # These are the 3 distinct 4-cycles on 4 vertices.
    
    # The constraint: for every 4-cycle (u,v,w,x) in order, the colors c(uv), c(vw), c(wx), c(xu) must be all distinct.
    
    # This is a very strong constraint. Let's think about what colorings are possible.
    
    # One approach: use a proper edge coloring of the complete graph with additional constraints.
    # K_n for n=20 has 190 edges. A proper edge coloring uses at least 19 colors (since max degree is 19).
    # We have 20 colors available.
    
    # Let's try a different approach: use a 1-factorization of K_20.
    # K_20 has a 1-factorization into 19 perfect matchings (since n is even).
    # Each matching uses one color. But wait, we have 20 colors and 19 matchings.
    # If we color each edge in matching i with color i, then any 4-cycle... 
    # In a 1-factorization, each color class is a perfect matching.
    # A 4-cycle would use 4 edges. Can a 4-cycle use two edges of the same color?
    # Two edges of the same color are in the same matching, so they are disjoint.
    # A 4-cycle has 4 edges. If two edges have the same color, they must be disjoint (since same matching).
    # In a 4-cycle, opposite edges are disjoint. So a 4-cycle could have two pairs of opposite edges with the same color.
    # E.g., cycle a-b-c-d-a: edges ab, bc, cd, da. If ab and cd have same color, and bc and da have same color, 
    # then we have only 2 colors used, not 4 distinct. This violates the rainbow condition.
    
    # So a simple 1-factorization coloring doesn't work directly.
    
    # Alternative: use a proper edge coloring where no two edges in any 4-cycle share a color.
    # This is stronger than proper edge coloring (which only requires adjacent edges to have different colors).
    
    # Let's think about the structure more carefully.
    # For a 4-cycle u-v-w-x-u, we need c(uv) != c(vw), c(vw) != c(wx), c(wx) != c(xu), c(xu) != c(uv),
    # AND c(uv) != c(wx) (opposite edges), c(vw) != c(xu) (opposite edges).
    # So all 4 colors must be distinct.
    
    # This means: in any 4-cycle, the two pairs of opposite edges must have different colors from each other too.
    
    # Let's try a construction based on modular arithmetic.
    # Vertices 0..19. Color edge (i,j) with (i+j) mod 20? Let's check.
    # 4-cycle: 0-1-2-3-0. Edges: (0,1):1, (1,2):3, (2,3):5, (3,0):3. Colors 1,3,5,3. Not rainbow (3 repeated).
    
    # Try color (i,j) = (i-j) mod 20? 
    # 4-cycle 0-1-2-3-0: (0,1):19, (1,2):19, ... no, (i-j) mod 20 for i<j would be negative.
    # Let's use (i+j) mod 20 but with a twist.
    
    # Actually, let's try: color(i,j) = (i + 2*j) mod 20 for i < j? No, need symmetry.
    # color(i,j) = (i+j) mod 20 is symmetric.
    
    # The problem with (i+j) mod 20: for cycle 0-1-2-3-0, we get 1,3,5,3. The issue is that (0,1) and (3,0) both give 3 mod 20? 
    # (0+1)=1, (1+2)=3, (2+3)=5, (3+0)=3. Yes, 3 repeats.
    
    # What if we use a different modulus or a different function?
    # Let's try color(i,j) = (i*3 + j*5) mod 20? Not symmetric.
    # color(i,j) = (i+j) mod 20 fails.
    
    # Let's try a completely different approach: greedy with backtracking, or use a known construction.
    
    # Known result: The maximum number of edges in a graph on n vertices with a rainbow 4-cycle-free edge coloring 
    # is related to anti-Ramsey numbers. For n=20, we need to find a dense graph.
    
    # Let's try: complete bipartite graph K_{10,10} with a proper edge coloring.
    # K_{10,10} has 100 edges. If we can color it so that every 4-cycle is rainbow, that's 100 edges.
    # A 4-cycle in K_{10,10} is of the form a1-b1-a2-b2-a1 where a1,a2 in A, b1,b2 in B.
    # The edges are (a1,b1), (b1,a2), (a2,b2), (b2,a1).
    
    # If we use a proper edge coloring of K_{10,10}, each color class is a matching.
    # K_{10,10} has a 10-edge-coloring (since max degree is 10).
    # Can we arrange it so that no 4-cycle has two edges of the same color?
    # In K_{10,10}, a 4-cycle uses 2 edges incident to a1 and 2 edges incident to a2 (and similarly for b1, b2).
    # If we use a Latin square type coloring: color(a_i, b_j) = (i+j) mod 10? 
    # 4-cycle: a1-b1-a2-b2-a1. Edges: (a1,b1):(1+1)=2, (a2,b1):(2+1)=3, (a2,b2):(2+2)=4, (a1,b2):(1+2)=3.
    # Colors: 2,3,4,3. Not rainbow.
    
    # Try color(a_i, b_j) = (i*j) mod 10? Not necessarily proper.
    
    # Let's go back to the complete graph and try a greedy approach with careful color assignment.
    
    # Actually, let's try a simpler construction: a complete graph with a specific coloring pattern.
    
    # For K_n with n even, use a 1-factorization. There are n-1 matchings.
    # Assign colors 0..n-2 to the matchings. We have n colors available, so one color is unused.
    # The issue is that 4-cycles can have opposite edges with the same color.
    
    # What if we break the symmetry? Use a non-uniform coloring.
    
    # Let's try: for edge (i,j), color = (i+j) mod 20, but if that causes a conflict, adjust.
    # This is getting complex. Let's implement a greedy algorithm with backtracking for small n, 
    # but n=20 might be too large for backtracking.
    
    # Alternative: use a random greedy approach and hope for the best.
    
    # Let's try: start with empty graph, add edges one by one, assign colors greedily.
    # For each new edge, try all colors and check if it creates a non-rainbow 4-cycle.
    
    # This is O(E^2 * C) which for E=190, C=20 is manageable.
    
    # Let's implement this.
    
    import random
    random.seed(42)
    
    n = 20
    max_colors = 20
    
    # adjacency: adj[i][j] = color of edge (i,j), or -1 if no edge
    adj = [[-1]*n for _ in range(n)]
    
    # List of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    # For each edge, try to add it with some color
    def is_valid_4cycle(u, v, w, x, c_uv, c_vw, c_wx, c_xu):
        """Check if the 4-cycle u-v-w-x-u has all distinct colors."""
        return len({c_uv, c_vw, c_wx, c_xu}) == 4
    
    def can_add_edge(u, v, c):
        """Check if adding edge (u,v) with color c creates any non-rainbow 4-cycle."""
        # A 4-cycle containing (u,v) must have the form u-v-w-x-u or u-v-x-w-u etc.
        # The 4th vertex x must be adjacent to both u and v (or will be).
        # Actually, for a 4-cycle u-v-w-x-u, we need edges (u,v), (v,w), (w,x), (x,u).
        # So w must be adjacent to v, and x must be adjacent to w and u.
        
        # For each possible w (adjacent to v), and x (adjacent to w and u):
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w] == -1:
                continue
            for x in range(n):
                if x == u or x == v or x == w:
                    continue
                if adj[w][x] == -1 or adj[u][x] == -1:
                    continue
                # Check 4-cycle u-v-w-x-u
                if not is_valid_4cycle(u, v, w, x, c, adj[v][w], adj[w][x], adj[u][x]):
                    return False
                # Also check the other 4-cycle: u-v-x-w-u? No, that's a different cycle.
                # The 4-cycle is defined by the 4 vertices in order. u-v-w-x-u is one cycle.
                # Is there another 4-cycle using edge (u,v) with vertices u,v,w,x?
                # The other cycle would be u-v-x-w-u, which uses edges (u,v), (v,x), (x,w), (w,u).
                # But (v,x) and (w,u) may not exist. Let's check if this is a valid 4-cycle.
                if adj[v][x] != -1 and adj[w][u] != -1:
                    if not is_valid_4cycle(u, v, x, w, c, adj[v][x], adj[x][w], adj[w][u]):
                        return False
        return True
    
    for (u, v) in all_edges:
        added = False
        # Try colors in a random order
        colors_to_try = list(range(max_colors))
        random.shuffle(colors_to_try)
        for c in colors_to_try:
            if can_add_edge(u, v, c):
                adj[u][v] = c
                adj[v][u] = c
                added = True
                break
        if not added:
            # Can't add this edge with any color
            continue
    
    # Build result
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j] != -1:
                result.append([i, j, adj[i][j]])
    
    return result
# EVOLVE-BLOCK-END
