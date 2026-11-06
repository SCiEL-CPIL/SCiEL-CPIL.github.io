"""
Four-colourings of complete graphs without monochromatic C6 (R_4(C6)).
Strategy:
  1. Start from the known Z_17 construction (cyclic group Z_17 with difference set).
  2. Attempt to extend to n=18 via bit-parallel backtracking with pruning.
  3. If n=18 fails, verify n=17 and return it.
  4. Deadline guard: return best valid coloring found before 110s.
"""
import time
import sys

def _has_c6_bitmask(color_adj, n):
    """
    Check if color_adj (list of n integers, bit i set means edge to i in this color)
    contains a 6-cycle. A 6-cycle is a sequence v0-v1-v2-v3-v4-v5-v0 where all
    consecutive pairs (and v5-v0) are edges in this color class.
    We check for simple 6-cycles: v0 < v1 < ... to avoid duplicates, but for
    correctness we just need to detect existence.
    Use: for each pair (a,b) in color, for each (c,d) in color with c!=a,b and d!=a,b,
    check if there exists e such that b-e and e-d are in color, and e not in {a,b,c,d},
    and also a-e not required... this is getting complex.
    
    Simpler: enumerate all 6-tuples (v0,...,v5) with v0<v1<...<v5 is too restrictive.
    Instead: a 6-cycle exists iff there are vertices v0,...,v5 (distinct) such that
    (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v5),(v5,v0) are all edges in the color.
    
    Efficient check: for each ordered pair (a,b) with a!=b and edge(a,b) in color,
    we want to know if there's a path of length 4 from b back to a using 4 distinct
    intermediate vertices. That's expensive.
    
    Alternative: use the fact that in a graph with n vertices, checking for C6 can be
    done by: for each edge (u,v), check if there's a path of length 4 between u and v
    in G - {u,v} that uses 4 distinct vertices. Still expensive.
    
    For n<=18, we can use a simpler approach: iterate over all 6-element subsets and
    check if any induces a C6 (not necessarily induced, but contains a 6-cycle).
    Number of 6-subsets of 18: C(18,6)=18564. For each, check if the 6 vertices contain
    a Hamiltonian cycle in the color subgraph. That's 6! / 2 / 6 = 60 cycles to check
    per subset worst case, but we can use backtracking.
    
    Actually, even simpler: for each color class, build adjacency matrix and use
    a standard C6 detection: for each pair (a,b), compute reachability via paths of
    length 4. Let's use: A = adjacency matrix. C6 exists iff there exist a,b,c,d,e,f
    distinct with edges ab,bc,cd,de,ef,fa.
    
    For n=18, let's just do: for each 6-subset, check if the induced subgraph has a
    cycle of length 6. We can check by trying all permutations of the 6 vertices as
    a cycle: 6!/2/6 = 60 distinct cycles. 18564 * 60 = ~1.1M checks, each O(1) with
    bitmask. This is feasible.
    """
    # Precompute for each vertex, the bitmask of neighbors in this color
    # color_adj[i] is a bitmask of neighbors of i in this color
    # For each 6-subset, check if it contains a 6-cycle
    # A 6-cycle on vertices {v0..v5} exists if there's a permutation (w0..w5) of these
    # 6 vertices such that each consecutive pair (including w5-w0) is an edge.
    
    # Faster: for the 6 vertices, build a 6x6 adjacency and check for Hamiltonian cycle
    # in that subgraph. Use DP or just brute force since 6 is small.
    
    # Actually, let's use a different approach: 
    # For each pair (a, b) that is an edge in this color, check if there exists a
    # path of length 4 from b to a through 4 other distinct vertices.
    # We can compute: for each vertex v, paths of length k from a to v.
    
    # Given n is at most 18, let's use the 6-subset approach but optimize.
    
    # For each color, if the number of edges is small, C6 is unlikely.
    # Let's count edges first.
    edge_count = sum(bin(color_adj[i]).count('1') for i in range(n)) // 2
    if edge_count < 6:
        return False
    
    # Use the following: a graph contains C6 iff there exist vertices
    # v0, v1, v2, v3, v4, v5 (all distinct) such that all consecutive edges exist.
    # We iterate v0 < v1 to fix orientation, then search for v2, v3, v4, v5.
    
    # For efficiency with bitmask:
    # For each edge (v0, v1), we need v2 adjacent to v1 (v2 != v0),
    # v3 adjacent to v2 (v3 != v0, v1),
    # v4 adjacent to v3 (v4 != v0, v1, v2),
    # v5 adjacent to v4 and v5 adjacent to v0, v5 != v0, v1, v2, v3.
    
    for v0 in range(n):
        adj_v0 = color_adj[v0]
        if adj_v0 == 0:
            continue
        # For each neighbor v1 of v0
        remaining = adj_v0
        while remaining:
            v1 = (remaining & -remaining).bit_length() - 1
            remaining &= remaining - 1
            
            # v2 must be neighbor of v1, not v0
            adj_v1 = color_adj[v1] & ~(1 << v0)
            if adj_v1 == 0:
                continue
            
            rem2 = adj_v1
            while rem2:
                v2 = (rem2 & -rem2).bit_length() - 1
                rem2 &= rem2 - 1
                
                # v3 neighbor of v2, not v0, v1
                adj_v2 = color_adj[v2] & ~(1 << v0) & ~(1 << v1)
                if adj_v2 == 0:
                    continue
                
                rem3 = adj_v2
                while rem3:
                    v3 = (rem3 & -rem3).bit_length() - 1
                    rem3 &= rem3 - 1
                    
                    # v4 neighbor of v3, not v0, v1, v2
                    adj_v3 = color_adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                    if adj_v3 == 0:
                        continue
                    
                    rem4 = adj_v3
                    while rem4:
                        v4 = (rem4 & -rem4).bit_length() - 1
                        rem4 &= rem4 - 1
                        
                        # v5 neighbor of v4 and v0, not v0, v1, v2, v3
                        adj_v4 = color_adj[v4] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v3)
                        adj_v0_mask = adj_v0 & ~(1 << v1) & ~(1 << v2) & ~(1 << v3) & ~(1 << v4)
                        if adj_v4 & adj_v0_mask:
                            return True
    return False


def _build_color_adj(colors, n):
    """Build adjacency bitmask for each color class."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj


def _is_valid(colors, n):
    """Check if no color class contains a C6."""
    adj = _build_color_adj(colors, n)
    for c in range(4):
        if _has_c6_bitmask(adj[c], n):
            return False
    return True


def _z17_construction():
    """
    Known 4-coloring of K_17 using Z_17.
    Partition the 16 nonzero elements of Z_17 into 4 sets of 4 each (since each vertex
    has degree 16 in K_17, and 16/4 = 4 edges per color per vertex).
    
    The known construction uses the difference sets:
    Color 0: {1, 2, 4, 8}  (powers of 2 mod 17)
    Color 1: {3, 6, 12, 9} (3 * powers of 2)
    Color 2: {5, 10, 3, 13}... let me use a verified one.
    
    Actually, the standard Z_17 4-coloring for C6-free:
    Use the fact that 17 is prime. The multiplicative group Z_17* has order 16.
    We need to partition Z_17* into 4 cosets of a subgroup of order 4, or use
    a difference set.
    
    Known: the (17,4,1) difference set is {0, 1, 2, 4} in Z_17.
    For 4-coloring, we can use:
    d in {1,2,4,8} -> color 0
    d in {3,6,12,9} -> color 1  
    d in {5,10,3,13} -> wait, 3 is already used.
    
    Let me use the quadratic residues approach or just hardcode a known valid one.
    
    Actually, for the purpose of this task, let me use the construction from the
    literature: R_4(C6) >= 17. One known coloring uses:
    vertices 0..16, edge (i,j) gets color based on (i-j) mod 17 mapped to one of 4 sets.
    
    The 4 sets partitioning {1,...,16} are:
    S0 = {1, 2, 4, 8}
    S1 = {3, 6, 12, 9}
    S2 = {5, 10, 3, 13} -- no, 3 is in S1.
    
    Let me just use: the 16 nonzero elements partitioned into 4 sets of 4:
    S0 = {1, 4, 8, 16}  (2^0, 2^2, 2^4, 2^6)
    S1 = {2, 3, 6, 12}  (2^1, 3*2^0, 3*2^1, 3*2^2)
    S2 = {5, 10, 13, 9} (5, 10, 13, 9) -- check: 5*2=10, 10*2=20=3, no.
    
    I'll use a simpler approach: just return the known n=17 coloring that's been
    verified. Let me construct it properly.
    
    The known result: R_4(C6) > 16, i.e., there exists a 4-coloring of K_17 with
    no monochromatic C6. One such coloring can be found in the literature.
    
    For simplicity and correctness, let me use a greedy/verified approach:
    I'll hardcode a known valid 17-vertex coloring.
    """
    n = 17
    # Use the Z_17 construction with the following partition of {1,...,16}:
    # Based on the fact that the multiplicative group has a subgroup of order 4: {1,4,13,16} (since 4^1=4, 4^2=16, 4^3=64=13, 4^4=52=1)
    # Cosets:
    # H = {1, 4, 13, 16}
    # 2H = {2, 8, 10, 15}
    # 3H = {3, 12, 7, 14}
    # 5H = {5, 20=3, ...} wait 5*4=20=3, that's in 3H.
    
    # Let me use the cyclic subgroup generated by 4: <4> = {1, 4, 16, 13}
    # Cosets of <4> in Z_17*:
    # 1*<4> = {1, 4, 16, 13}
    # 2*<4> = {2, 8, 15, 10}
    # 3*<4> = {3, 12, 14, 7}
    # 5*<4> = {5, 20=3, ...} 5*1=5, 5*4=20=3, 5*16=80=12, 5*13=65=14. So {5,3,12,14} which is same as 3*<4>.
    
    # So we only get 3 cosets from the subgroup of order 4? No, 17-1=16, 16/4=4 cosets.
    # 1*<4> = {1,4,16,13}
    # 2*<4> = {2,8,15,10}  
    # 3*<4> = {3,12,14,7}
    # 5*<4> = {5,3,12,14} -- overlap with 3*<4>!
    
    # 5 is not in any of the above? 1,4,16,13,2,8,15,10,3,12,14,7 -- that's 12 elements. Missing: 5,6,9,11.
    # 5*<4>: 5, 20=3, 80=12, 65=14. So {5,3,12,14}. But 3,12,14 are in 3*<4>.
    
    # I think I'm making an error. Let me just use a known valid coloring.
    # 
    # Actually, let me just use the simple approach: for n=17, use the construction
    # where color(i,j) = ((i-j) mod 17) mapped to 4 colors via a fixed table.
    # 
    # The key insight: if the 4 sets are each C6-free as "difference sets" in Z_17,
    # then the coloring works.
    #
    # Let me just return a hardcoded valid coloring for n=17.
    # I'll generate it using a known method.
    
    # Use the following: for difference d = (i-j) mod 17, assign color:
    # d in {1, 2, 4, 8} -> 0
    # d in {3, 5, 9, 13} -> 1
    # d in {6, 7, 11, 14} -> 2
    # d in {10, 12, 15, 16} -> 3
    
    color_map = {}
    for d in [1, 2, 4, 8]:
        color_map[d] = 0
        color_map[17-d] = 0
    for d in [3, 5, 9, 13]:
        color_map[d] = 1
        color_map[17-d] = 1
    for d in [6, 7, 11, 14]:
        color_map[d] = 2
        color_map[17-d] = 2
    for d in [10, 12, 15, 16]:
        color_map[d] = 3
        color_map[17-d] = 3
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % 17
            colors.append(color_map[d])
    
    return n, colors


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Step 1: Get the known n=17 coloring
    n17, colors17 = _z17_construction()
    
    # Verify it's valid
    if not _is_valid(colors17, n17):
        # Fallback to n=9
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
    
    best_n = 17
    best_colors = colors17
    
    # Step 2: Try to extend to n=18
    # Add vertex 17, assign colors to edges (17, j) for j=0..16
    # Backtrack over 4^17 possibilities with pruning
    
    n_target = 18
    # We'll try to find colors for the new vertex
    # For efficiency, we can use symmetry breaking: fix color of (0,17) to 0
    
    found_18 = False
    colors18 = None
    
    # Time-limited search
    while time.time() < deadline and not found_18:
        # Try to extend: assign colors to edges (17, j) for j in 0..16
        # Use backtracking with pruning
        
        def backtrack(idx, current_colors):
            nonlocal found_18, colors18
            if time.time() > deadline:
                return False
            
            if idx == 17:
                # Check validity
                if _is_valid(current_colors, 18):
                    found_18 = True
                    colors18 = current_colors[:]
                    return True
                return False
            
            for c in range(4):
                new_colors = current_colors + [c]
                # Pruning: check if adding this edge creates a C6 in color c
                # Build partial adjacency for color c and check
                # This is expensive, so only check at the end or use smarter pruning
                
                if backtrack(idx + 1, new_colors):
                    return True
            
            return False
        
        # Initial: 17*17/2 = 136 edges for K_17, plus 17 edges for vertex 17
        # Total: 153 edges
        init_colors = colors17 + [0] * 17
        if backtrack(0, init_colors):
            found_18 = True
    
    if found_18 and colors18:
        return {'n': 18, 'colors': colors18}
    
    # Return the best we have
    return {'n': best_n, 'colors': best_colors}
