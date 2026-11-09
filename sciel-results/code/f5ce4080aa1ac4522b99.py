import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit
    
    N = 128
    full_mask = (1 << (N + 1)) - 2  # bits 1..N set
    
    # Precompute LCMs
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # Precompute bad triples: for each pair (i,j), store bitset of k such that {i,j,k} is bad
    # bad[i][j] is a bitmask of k > j such that {i,j,k} is bad? 
    # To make it symmetric and easy, bad[i][j] will contain all k != i,j such that {i,j,k} is bad.
    bad = [[0] * (N + 1) for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm(a, b)
            for c in range(b + 1, N + 1):
                if lab == lcm(a, c) == lcm(b, c):
                    bad[a][b] |= (1 << c)
                    bad[b][a] |= (1 << c)
                    bad[a][c] |= (1 << b)
                    bad[c][a] |= (1 << b)
                    bad[b][c] |= (1 << a)
                    bad[c][b] |= (1 << a)
    
    # Greedy lower bound
    def greedy(order):
        S = []
        S_mask = 0
        for x in order:
            if not (full_mask >> x) & 1:
                continue
            compatible = True
            for s in S:
                if bad[s][x] & S_mask:
                    compatible = False
                    break
            if compatible:
                S.append(x)
                S_mask |= (1 << x)
        return S
    
    best_set = []
    best_size = 0
    
    # Try several greedy orderings
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: x.bit_count()),
        sorted(range(1, N + 1), key=lambda x: -x.bit_count())
    ]
    
    for order in orders:
        S = greedy(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # B&B Solver
    # We represent S as a list and A as a bitmask.
    # To speed up "is y compatible with x given S", we can precompute for each x a mask of "conflicts with S".
    # But S changes.
    
    def get_conflict_mask(x, S_mask, S_list):
        """Returns mask of elements in A that conflict with x given current S.
        y conflicts with x if there exists z in S such that {x,y,z} is bad.
        So conflict_mask = OR_{z in S} bad[x][z]
        """
        mask = 0
        for z in S_list:
            mask |= bad[x][z]
        return mask
    
    def greedy_color_upper_bound(A_mask, S_list):
        """
        Computes an upper bound on the size of an independent set in G_dynamic.
        G_dynamic has edge (u,v) if exists z in S such that {u,v,z} is bad.
        We want alpha(G_dynamic).
        This is omega(complement of G_dynamic).
        We perform greedy coloring on the complement graph.
        
        Vertices are in A_mask.
        Two vertices u, v are adjacent in G_dynamic if (bad[u][v] & S_mask) != 0?
        No, bad[u][v] contains w such that {u,v,w} is bad.
        So (u,v) is edge in G_dynamic iff (bad[u][v] & S_mask) != 0.
        
        So in the complement graph, (u,v) is NOT an edge if (bad[u][v] & S_mask) == 0.
        We want to color the complement graph.
        Actually, we want to color the graph H where (u,v) is edge if they CAN be together?
        No.
        Let's stick to: Upper Bound = Greedy Color Count of the graph where edges represent "cannot be together".
        Wait, standard MIS upper bound:
        If we color the graph G (where edges are conflicts), the color classes are independent sets.
        This does NOT bound the size of a single independent set.
        
        Correction:
        We want to bound the size of an Independent Set in G_dynamic.
        Let H be the graph where (u,v) is an edge if u and v are NOT compatible (i.e., they are in G_dynamic).
        We want alpha(H).
        alpha(H) = omega(complement(H)).
        So we compute a greedy coloring of complement(H).
        Let C be the number of colors used in greedy coloring of complement(H).
        Then omega(complement(H)) <= C.
        So alpha(H) <= C.
        
        How to color complement(H)?
        Vertices: A.
        Edge in complement(H) between u, v if u, v are compatible (i.e., NOT in G_dynamic).
        So (u,v) in E(complement(H)) iff (bad[u][v] & S_mask) == 0.
        
        Greedy Coloring of complement(H):
        Order vertices arbitrarily (e.g., by index).
        For each vertex u, assign the smallest color c such that no neighbor v (already colored) has color c.
        Neighbor in complement(H) means compatible.
        So if u is compatible with v, they must have different colors.
        
        This partition A into sets where any two elements in the same set are INCOMPATIBLE (edge in H).
        Wait. If u and v have the same color, they are NOT adjacent in complement(H).
        So they ARE adjacent in H (or are the same vertex).
        So each color class is a Clique in H?
        No.
        In Coloring, adjacent vertices have different colors.
        So if u, v have same color, they are NOT adjacent in the graph being colored (complement(H)).
        So they ARE adjacent in H (or not connected, but in simple graph, non-adjacent in complement means adjacent in original? No.
        complement(H) edge = NOT H edge.
        So if NOT (u,v) in E(complement(H)), then (u,v) in E(H).
        So if u, v have same color, they are not adjacent in complement(H), so they ARE adjacent in H.
        So each color class is a Clique in H.
        
        We want Independent Set in H.
        An Independent Set in H can contain at most one vertex from each Clique in H.
        So the size of the Independent Set is at most the number of Cliques in the partition?
        No, the partition covers all vertices.
        If we partition V into k Cliques, any Independent Set can pick at most 1 from each Clique.
        So size <= k.
        
        So, if we can partition A into k sets such that each set is a Clique in H (i.e., any two elements in the set are incompatible), then the max independent set size is <= k.
        
        How to find such a partition?
        We need to color the graph G_compatible where (u,v) is edge if compatible.
        Wait.
        If we color G_compatible, adjacent vertices (compatible) get different colors.
        So same color vertices are NOT adjacent in G_compatible, meaning they are INCOMPATIBLE.
        So same color vertices form a set of pairwise incompatible vertices.
        This is exactly a Clique in H (where H is the incompatibility graph).
        
        So:
        1. Construct G_compatible on A: (u,v) edge if (bad[u][v] & S_mask) == 0.
        2. Greedy Color G_compatible.
        3. Number of colors is the upper bound.
        """
        if not A_mask:
            return 0
        
        # Get list of vertices
        verts = []
        temp = A_mask
        while temp:
            low = temp & (-temp)
            x = low.bit_length() - 1
            verts.append(x)
            temp ^= low
        
        # Sort vertices by degree in G_compatible? Or just natural order.
        # Natural order is fast.
        
        colors = {}
        num_colors = 0
        
        for u in verts:
            # Determine used colors by neighbors in G_compatible
            used_colors = set()
            # Check all v < u in verts? No, check all previously colored vertices.
            # To be efficient, we can just check all v in verts that are already colored.
            # Since N=128, this is fast enough.
            for v in verts:
                if v == u or v not in colors:
                    continue
                # Check if u, v are compatible
                # Compatible if (bad[u][v] & S_mask) == 0
                if (bad[u][v] & S_mask) == 0:
                    used_colors.add(colors[v])
            
            # Assign smallest available color
            c = 0
            while c in used_colors:
                c += 1
            colors[u] = c
            if c + 1 > num_colors:
                num_colors = c + 1
        
        return num_colors

    def solve(S_list, A_mask):
        nonlocal best_size, best_set, start_time
        
        if time.time() - start_time > deadline:
            return
        
        # Update best
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = S_list[:]
        
        if not A_mask:
            return
        
        # Upper Bound
        ub = len(S_list) + greedy_color_upper_bound(A_mask, S_list)
        if ub <= best_size:
            return
        
        # Branching
        # Pick vertex with highest degree in H (incompatibility graph)
        # H edge (u,v) if (bad[u][v] & S_mask) != 0
        # So degree in H is count of v in A such that (bad[u][v] & S_mask) != 0
        # Or pick vertex with highest degree in G_compatible?
        # High degree in G_compatible means many compatible neighbors, so low degree in H.
        # We want to branch on a vertex that is "constraining".
        # Usually, branching on high degree in the graph being solved (Clique in G_compatible) is good.
        # Here we are solving for Independent Set in H, which is Clique in G_compatible.
        # So pick u with highest degree in G_compatible.
        
        verts = []
        temp = A_mask
        while temp:
            low = temp & (-temp)
            x = low.bit_length() - 1
            verts.append(x)
            temp ^= low
        
        if not verts:
            return
        
        # Calculate degree in G_compatible for each v in verts
        # deg[v] = count of u in verts such that (bad[u][v] & S_mask) == 0
        # This is O(k^2). k <= 128. 128^2 = 16384. Very fast.
        
        max_deg = -1
        best_v = -1
        for v in verts:
            deg = 0
            for u in verts:
                if u == v: continue
                if (bad[u][v] & S_mask) == 0:
                    deg += 1
            if deg > max_deg:
                max_deg = deg
                best_v = v
        
        x = best_v
        
        # Branch 1: Include x
        # New S = S + [x]
        # New A = { y in A \ {x} | y is compatible with x given S }
        # y compatible with x given S iff for all z in S, {x,y,z} is not bad.
        # i.e., (bad[x][y] & S_mask) == 0.
        
        new_A = 0
        temp = A_mask & ~(1 << x)
        while temp:
            low = temp & (-temp)
            y = low.bit_length() - 1
            # Check compatibility
            if (bad[x][y] & S_mask) == 0:
                new_A |= (1 << y)
            temp ^= low
        
        solve(S_list + [x], new_A)
        
        if time.time() - start_time > deadline:
            return
        
        # Branch 2: Exclude x
        solve(S_list, A_mask & ~(1 << x))
    
    # Start search
    # Initial S is empty, A is full_mask
    # But we can start with the greedy result to get a tighter bound immediately?
    # If we start with S = greedy_result, we prune more.
    # However, we must ensure we explore all possibilities.
    # B&B explores the space. Starting with a non-empty S restricts the space to extensions of S.
    # We might miss solutions that are not supersets of S.
    # So we should start with S = [] and A = full_mask.
    # The greedy result gives a good initial best_size for pruning.
    
    solve([], full_mask)
    
    return best_set

