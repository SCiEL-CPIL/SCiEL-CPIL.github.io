"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _has_c6(color, adj):
    """Check if color class contains a C6 using adjacency matrix."""
    n = len(adj)
    # For each pair of non-adjacent vertices (in color graph), check if they share 2 common neighbors
    # Actually C6 means 6 vertices v0-v1-v2-v3-v4-v5-v0 all with same color edges
    # More efficient: for each 3-clique-free check... 
    # Let's use: C6 exists iff there exist 6 distinct vertices forming a cycle
    # Efficient check: for each edge (u,v) in color, and each edge (w,x) in color disjoint from {u,v},
    # check if there exist y,z completing the cycle
    # Simpler: enumerate all 6-subsets? Too slow for n>15.
    # Use: C6 exists iff there are two non-adjacent vertices with >= 2 common neighbors that form a path of length 4 between them
    # Actually simplest correct check for small n:
    # A C6 is a cycle of length 6. Check all 6-tuples (v0,...,v5) distinct where 
    # (v0,v1),(v1,v2),...,(v5,v0) are all in the color class.
    # For n up to ~17, number of 6-tuples is n*(n-1)*...*(n-5) which for n=17 is 17*16*15*14*13*12 = 8.2M
    # That's borderline. Let's be smarter.
    
    # Better: for each pair of vertices (a,b) NOT connected by color edge,
    # count paths of length 2 from a to b through color edges. If >= 2, and the intermediate
    # vertices are distinct and not a or b, we have a C4 not C6.
    
    # For C6: vertices v0-v1-v2-v3-v4-v5-v0
    # Pick v0, v3 (opposite). They are not adjacent (in C6, v0 and v3 are distance 3).
    # Actually in C6, v0 and v3 ARE non-adjacent. v0-v1-v2-v3 is path of length 3, v0-v5-v4-v3 is path of length 3.
    # So: for each non-edge (v0, v3) in the color graph, check if there exist v1, v2, v4, v5 such that
    # v0-v1, v1-v2, v2-v3, v3-v4, v4-v5, v5-v0 are all color edges, and all 6 vertices distinct.
    
    # This is still complex. Let's just do the direct check for n <= 18.
    # Number of 6-subsets of 18: C(18,6) = 18564. For each, check all 6!/2 = 360 orderings? No.
    # For a 6-set, check if it contains a C6 as a subgraph. There are (6-1)!/2 = 60 Hamiltonian cycles on 6 labeled vertices.
    # So total: 18564 * 60 = 1.1M checks. Each check is 6 edge lookups. That's ~7M operations. OK.
    
    # Actually even simpler: iterate over all 6-element subsets, and for each check if the induced subgraph contains a C6.
    # A C6 in the induced subgraph on 6 vertices means the induced subgraph has a Hamiltonian cycle.
    
    from itertools import combinations, permutations
    
    # Precompute for speed: just check all 6-tuples (v0,...,v5) with v0 < v1 < ... < v5 won't work because cycle order matters.
    # Let's enumerate 6-subsets and check if there's a cyclic ordering.
    
    # For efficiency with n up to 18:
    # For each 6-subset S, check if the color-induced subgraph on S has a Hamiltonian cycle.
    # Use DP or just check the 60 possible cycles.
    
    # Actually, let me just use a direct approach that's fast enough:
    # For each ordered 6-tuple (v0,...,v5) of distinct vertices, check if all 6 edges are in color.
    # But that's nP6 = n!/(n-6)! which for n=18 is 18! / 12! = 18*17*16*15*14*13 = 10,281,600. Too many.
    
    # Better: fix v0. For each v0, for each v1 != v0 with edge(v0,v1), for each v2 != v0,v1 with edge(v1,v2),
    # for each v3 != v0,v1,v2 with edge(v2,v3), for each v4 != v0..v3 with edge(v3,v4),
    # for each v5 != v0..v4 with edge(v4,v5) and edge(v5,v0).
    # That's bounded by n * d^5 where d is avg degree in color class. For n=17, d~8, that's 17*8^5 = 700K. OK!
    
    n = len(adj)
    for v0 in range(n):
        for v1 in range(n):
            if v1 == v0 or not adj[v0][v1]:
                continue
            for v2 in range(n):
                if v2 in (v0, v1) or not adj[v1][v2]:
                    continue
                for v3 in range(n):
                    if v3 in (v0, v1, v2) or not adj[v2][v3]:
                        continue
                    for v4 in range(n):
                        if v4 in (v0, v1, v2, v3) or not adj[v3][v4]:
                            continue
                        for v5 in range(n):
                            if v5 in (v0, v1, v2, v3, v4) or not adj[v4][v5] or not adj[v5][v0]:
                                continue
                            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Start with known n=9 coloring
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Try to extend to larger n using backtracking
    for target_n in range(10, 18):
        if time.time() > deadline:
            break
        
        # Use backtracking to assign colors to edges
        n = target_n
        num_edges = n * (n - 1) // 2
        colors = [-1] * num_edges
        adj = [[False] * n for _ in range(n)]
        
        # Precompute edge index
        def edge_idx(i, j):
            if i > j:
                i, j = j, i
            return i * (n - 1) - i * (i + 1) // 2 + j - i - 1
        
        def set_color(c, i, j, col):
            idx = edge_idx(i, j)
            c[idx] = col
            adj[i][j] = (col == 0)  # We'll track per-color adjacencies differently
            # Actually let's track all 4 color adjacencies
            return True
        
        # Better approach: track adjacencies per color
        adj_c = [[[[False] * n for _ in range(n)] for _ in range(4)]]  # This is getting complex
        
        # Simpler: just use a 2D array per color
        adj_by_color = [[[False] * n for _ in range(n)] for _ in range(4)]
        
        def assign(i, j, col):
            colors[edge_idx(i, j)] = col
            adj_by_color[col][i][j] = True
            adj_by_color[col][j][i] = True
        
        def unassign(i, j):
            col = colors[edge_idx(i, j)]
            colors[edge_idx(i, j)] = -1
            adj_by_color[col][i][j] = False
            adj_by_color[col][j][i] = False
        
        # Backtracking: assign colors to edges in order
        # Edge (i,j) for i<j in lex order
        # For efficiency, process edges in a smart order
        
        found = False
        # Use iterative deepening or just try to build up
        
        # Simpler approach: use a known construction or random restarts
        # Let's try a greedy + local search approach
        
        # Actually, let me just try to verify if a random coloring works, then use hill climbing
        # This is getting complex. Let me use a simpler but effective approach.
        
        # For now, try the circular construction for various n
        # The known result is n=17 exists. Let me try to construct it.
        
        # Try algebraic construction: Z_p where p is prime
        # For 4 colors, we need p such that we can partition Z_p \ {0} into 4 sets, none containing a C6 pattern.
        
        # Known: R_4(C6) > 17 is established. Let me try to find a specific construction.
        
        # Try: vertices are Z_17, color edge (i,j) based on (i-j) mod 17
        # We need to partition {1,...,16} into 4 sets of 4 elements each, such that no set contains a C6.
        
        # A C6 in Z_p with steps from set S: a, a+s1, a+s1+s2, ..., a+s1+...+s5, a
        # where s1+...+s5 = 0 mod p and all partial sums distinct mod p.
        
        # This is complex. Let me just try random search with verification.
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            if found:
                break
            
            random.seed(attempt * 12345 + target_n)
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            
            # Check if valid
            valid = True
            adj_by_color = [[[False] * n for _ in range(n)] for _ in range(4)]
            for i in range(n):
                for j in range(i + 1, n):
                    col = colors[edge_idx(i, j)]
                    adj_by_color[col][i][j] = True
                    adj_by_color[col][j][i] = True
            
            for col in range(4):
                if _has_c6(col, adj_by_color[col]):
                    valid = False
                    break
            
            if valid:
                best_n = n
                best_colors = colors[:]
                found = True
                break
            
            # If not valid, try local search: flip colors of edges involved in C6s
            # This is complex, so just try more random restarts
        
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
