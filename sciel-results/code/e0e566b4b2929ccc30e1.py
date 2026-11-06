"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n = 20
    colors = 20
    # Total possible edges in K_20
    all_edges = list(itertools.combinations(range(n), 2))
    edge_set = set(all_edges)
    
    # We want to maximize edges such that every 4-cycle is rainbow.
    # A 4-cycle is a set of 4 edges (u,v), (v,w), (w,x), (x,u) with u,v,w,x distinct.
    # The condition is that for every 4-cycle, the 4 colors are distinct.
    
    # Strategy: 
    # Start with a complete graph K_20 and remove edges until valid? 
    # Or start with empty and add edges?
    # Adding edges is likely better for maximizing count if we can check validity incrementally.
    # However, checking all 4-cycles for each new edge is expensive.
    
    # Let's try a greedy approach with randomization and backtracking if needed, 
    # or just a smart greedy that prioritizes edges that create fewest conflicts.
    
    # Actually, a known construction for "no monochromatic 4-cycle" or "rainbow 4-cycles" 
    # often involves structured colorings.
    # The task says "every 4-cycle is rainbow". This is a very strong condition.
    # In a complete graph K_n, the number of 4-cycles is huge.
    
    # Let's analyze the constraint:
    # If we have a 4-cycle u-v-w-x-u, colors c1, c2, c3, c4 must be distinct.
    # This implies no two adjacent edges in any 4-cycle can have the same color.
    # Wait, "rainbow" means all 4 are distinct.
    
    # Consider a triangle u-v-w-u. It's not a 4-cycle.
    # But consider a 4-cycle involving a triangle? No, simple 4-cycle has 4 distinct vertices.
    
    # Let's try to construct a dense graph.
    # If we use a 1-factorization of K_20, we have 19 perfect matchings.
    # Each matching has 10 edges.
    # If we color each matching with a unique color, does every 4-cycle become rainbow?
    # A 4-cycle in a complete graph uses 4 edges.
    # In a 1-factorization, any two edges of different matchings are disjoint? No.
    # Edges in the same matching are disjoint.
    # So a 4-cycle cannot use two edges from the same matching because those edges would be disjoint (no shared vertex), but a 4-cycle is a connected cycle.
    # Wait, if a 4-cycle uses two edges from the same matching, those edges must be disjoint.
    # In a cycle, edges are connected. So a 4-cycle cannot contain two disjoint edges.
    # Therefore, a 4-cycle must use edges from 4 different matchings.
    # If each matching has a unique color, then the 4 edges in the 4-cycle have 4 distinct colors.
    # Thus, the 4-cycle is rainbow!
    
    # So, if we take the complete graph K_20 and color its edges using a 1-factorization where each factor (perfect matching) gets a unique color, the condition is satisfied.
    # K_20 has 19 perfect matchings in a 1-factorization (since n is even, K_n has n-1 disjoint perfect matchings).
    # Total edges = 19 * 10 = 190.
    # This is the complete graph K_20!
    # So the maximum number of edges is 190.
    
    # Let's verify:
    # 1. K_20 has 190 edges.
    # 2. We need to color them with 20 colors (we have 20 available, using 19 is fine, or we can use 19 distinct colors).
    # 3. The coloring is based on a 1-factorization.
    # 4. Any 4-cycle in K_20 consists of 4 edges.
    # 5. Since the edges in a 1-factor are disjoint, no two edges in a 4-cycle can belong to the same factor.
    # 6. Therefore, the 4 edges belong to 4 distinct factors.
    # 7. Since each factor has a unique color, the 4 edges have distinct colors.
    # 8. Thus, every 4-cycle is rainbow.
    
    # We just need to generate a 1-factorization of K_20.
    # Standard construction for K_{2m} (here m=10):
    # Vertices: 0, 1, ..., 2m-2, and infinity (or 2m-1).
    # Actually, standard round-robin tournament scheduling.
    # Vertices 0..18 and 19 (infinity).
    # Fix vertex 19. Rotate 0..18.
    # For round i = 0..18:
    #   Match 19 with (i) mod 19? No, standard algorithm:
    #   Place vertex 19 in the center.
    #   The other 19 vertices are arranged in a circle.
    #   In round i, vertex 19 matches with vertex i.
    #   The other 9 pairs are formed by reflecting across the axis through 19 and i.
    
    # Let's implement the 1-factorization.
    # n = 20.
    # Vertices: 0..19.
    # Let v_inf = 19.
    # The other vertices are 0..18.
    # For round r in 0..18:
    #   Pair (v_inf, r)
    #   For k in 1..9:
    #       a = (r + k) % 19
    #       b = (r - k) % 19
    #       Pair (a, b)
    #   Assign color r to all pairs in this round.
    
    # Let's double check if this covers all edges exactly once.
    # Yes, this is a standard 1-factorization of K_{2m} for n=2m.
    
    # Let's generate the edges.
    edges = []
    for r in range(19):
        color = r
        # Pair with infinity
        u = 19
        v = r
        edges.append([u, v, color])
        
        for k in range(1, 10):
            a = (r + k) % 19
            b = (r - k) % 19
            # Ensure a < b for consistency, though not strictly required for the list format
            if a > b:
                a, b = b, a
            edges.append([a, b, color])
            
    # Check if we have 190 edges
    # 19 rounds * 10 pairs = 190 edges.
    
    # Verify no duplicate edges
    edge_set_check = set()
    for e in edges:
        u, v, c = e
        if u > v:
            u, v = v, u
        edge_set_check.add((u, v))
    if len(edge_set_check) != 190:
        # Fallback to star if something went wrong
        return [[0, v, 0] for v in range(1, 20)]
        
    return edges

# EVOLVE-BLOCK-END
