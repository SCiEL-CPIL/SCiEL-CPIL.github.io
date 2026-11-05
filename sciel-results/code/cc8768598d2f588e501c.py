import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 58.0

    N = 256
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build adjacency as bitsets
    adj = [0] * N
    forbidden = [False] * N

    for i in range(N):
        v = i + 1
        if 2 * v in squares:
            forbidden[i] = True
        else:
            for sq in squares:
                u = sq - v
                if 1 <= u <= N:
                    j = u - 1
                    if not forbidden[j]:
                        adj[i] |= (1 << j)

    candidates = [i for i in range(N) if not forbidden[i]]
    cand_mask = 0
    for i in candidates:
        cand_mask |= (1 << i)

    # Number of candidates
    M = len(candidates)
    
    # Map candidate index (0..M-1) to original vertex (0..N-1)
    cand_to_orig = candidates
    orig_to_cand = [-1] * N
    for i, v in enumerate(candidates):
        orig_to_cand[v] = i

    # Build adjacency in candidate space (M vertices)
    # adj_cand[i] is a bitmask of neighbors of candidate i in the candidate graph
    adj_cand = [0] * M
    for i in range(M):
        v = cand_to_orig[i]
        # Get all neighbors in original graph that are also candidates
        mask = adj[v]
        # Convert to candidate bitmask
        cmask = 0
        m = mask
        while m:
            lsb = m & -m
            orig_v = lsb.bit_length() - 1
            ci = orig_to_cand[orig_v]
            if ci != -1:
                cmask |= (1 << ci)
            m ^= lsb
        adj_cand[i] = cmask

    # Complement adjacency in candidate space (for Max Clique formulation)
    # comp_adj_cand[i] is a bitmask of non-neighbors of candidate i (excluding self)
    # This is the graph in which we seek a Maximum Clique
    comp_adj_cand = [0] * M
    full_cand_mask = (1 << M) - 1
    for i in range(M):
        # Non-neighbors are those in full_cand_mask, not i, and not in adj_cand[i]
        comp_adj_cand[i] = full_cand_mask & ~adj_cand[i] & ~(1 << i)

    # Greedy Coloring for Upper Bound on Clique Size
    # We color the vertices in P (bitmask) such that adjacent vertices in comp_adj_cand have different colors.
    # The max color assigned is an upper bound on the clique size in P.
    # To make the bound tight, we order vertices by degree in comp_adj_cand (descending).
    
    # Precompute degrees in comp_adj_cand for sorting heuristic
    comp_deg = [0] * M
    for i in range(M):
        comp_deg[i] = comp_adj_cand[i].bit_count()

    def greedy_color_bound(P):
        """
        Returns (max_color, color_order) where color_order is a list of (vertex, color) 
        sorted by color, then by vertex index? 
        Actually, for Tomita, we just need the max color and potentially the order.
        We return the max color.
        """
        if P == 0:
            return 0
        
        # Get list of vertices in P
        verts = []
        m = P
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        
        # Sort by degree in comp_adj_cand (descending)
        # Higher degree means more constraints, might need higher color?
        # Tomita sorts by degree in the *current* subgraph usually, but static degree is a good proxy.
        verts.sort(key=lambda v: -comp_deg[v])
        
        colors = [0] * M
        max_color = 0
        
        for v in verts:
            # Find available colors
            # Neighbors of v in P
            neighbors = comp_adj_cand[v] & P
            # Used colors by neighbors
            used = [False] * (max_color + 2) # Slightly larger buffer
            m_n = neighbors
            while m_n:
                lsb = m_n & -m_n
                u = lsb.bit_length() - 1
                if colors[u] > 0:
                    if colors[u] <= max_color + 1:
                        used[colors[u]] = True
                    else:
                        # This shouldn't happen if max_color is tracked correctly
                        pass
                m_n ^= lsb
            
            # Find smallest available color
            c = 1
            while c <= max_color + 1 and used[c]:
                c += 1
            
            colors[v] = c
            if c > max_color:
                max_color = c
        
        return max_color

    # Tomita-style Maximum Clique Search
    # P: set of candidate vertices to add to the clique
    # X: set of vertices already excluded (not used in standard simple version, but helps in some variants)
    # Here we use a simple recursive expansion.
    
    best_clique_size = 0
    best_clique_mask = 0
    timeout_flag = [False]

    def expand_clique(P, X, current_clique):
        nonlocal best_clique_size, best_clique_mask, timeout_flag
        
        if time.time() - start_time > DEADLINE - 1.0:
            timeout_flag[0] = True
            return
            
        if P == 0:
            return
            
        # Bound
        ub = greedy_color_bound(P)
        if current_clique.bit_count() + ub <= best_clique_size:
            return
            
        # Select vertex v from P
        # Heuristic: pick vertex with max degree in P? Or just first one?
        # Tomita picks the last vertex in the color order (highest color).
        # For simplicity and speed, let's pick the vertex in P with max degree in P.
        
        # To avoid O(M^2) for degree calc every time, we can just pick the first one?
        # No, that's bad. Let's pick the one with max degree in P.
        m = P
        best_v = -1
        max_deg = -1
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = (comp_adj_cand[v] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            m ^= lsb
        
        v = best_v
        lv = 1 << v
        
        # Branch 1: Include v
        new_P = P & comp_adj_cand[v]
        new_X = X | (P & ~comp_adj_cand[v]) # Not strictly needed for simple version, but good for pruning?
        # Actually, standard Tomita:
        # expand(P & N(v), X | (P \ N(v)), C U {v})
        # But we can simplify if we don't use X for pruning yet.
        
        expand_clique(new_P, X, current_clique | lv)
        
        if timeout_flag[0]:
            return
            
        # Branch 2: Exclude v
        P = P & ~lv
        # We can recompute bound or just continue.
        # To be efficient, we should check bound again?
        # The bound on the new P (which is smaller) is <= old bound.
        # But the current_clique size is the same.
        # So if old bound was tight, new bound might be tighter.
        
        expand_clique(P, X, current_clique)

    # Initialize search
    # Start with empty clique, P = all candidates
    # We can sort initial P by degree to help?
    # The expand function handles ordering internally.
    
    # To improve performance, we can run the search multiple times with different initial orderings or restarts?
    # Or just run once with a good heuristic.
    
    # Let's run the B&B.
    # Note: M can be up to ~200. 2^200 is huge.
    # The bound is critical.
    
    # Optimization: If M is small, we can be more aggressive.
    # If M is large, we might need to rely on the heuristic more.
    
    # Let's add a simple local search improvement on the best clique found so far.
    # Actually, the B&B should find the optimal if it completes.
    # If it times out, we return the best found.
    
    # To make the B&B faster, we can use a greedy initialization to set a good lower bound (best_clique_size).
    
    def greedy_clique_init():
        """
        Greedy construction of a clique.
        """
        P = full_cand_mask
        clique = 0
        while P:
            # Pick vertex with max degree in P
            m = P
            best_v = -1
            max_deg = -1
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = (comp_adj_cand[v] & P).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                m ^= lsb
            
            if best_v == -1:
                break
                
            lv = 1 << best_v
            clique |= lv
            P &= comp_adj_cand[best_v]
            
        return clique

    # Get initial lower bound
    init_clique = greedy_clique_init()
    best_clique_size = init_clique.bit_count()
    best_clique_mask = init_clique
    
    # Run B&B
    # We might want to limit the depth or time per node?
    # The timeout_flag handles the global time.
    
    # To prevent stack overflow or excessive recursion depth, Python's recursion limit might be an issue.
    # sys.setrecursionlimit(10000)
    # But 60s limit is more important.
    
    # Let's run the search.
    # We can also try multiple random restarts of the greedy init to get a better lower bound?
    # Yes, a better lower bound prunes more.
    
    for _ in range(5):
        if time.time() - start_time > DEADLINE - 2.0:
            break
        # Randomized greedy
        P = full_cand_mask
        clique = 0
        while P:
            m = P
            best_v = -1
            max_score = -1
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = (comp_adj_cand[v] & P).bit_count()
                score = deg + random.uniform(0, 1.0)
                if score > max_score:
                    max_score = score
                    best_v = v
                m ^= lsb
            
            if best_v == -1:
                break
                
            lv = 1 << best_v
            clique |= lv
            P &= comp_adj_cand[best_v]
            
        if clique.bit_count() > best_clique_size:
            best_clique_size = clique.bit_count()
            best_clique_mask = clique

    # Now run B&B
    # We pass the full set as P.
    # We can also restrict P to vertices that could potentially improve the solution?
    # Not easily.
    
    expand_clique(full_cand_mask, 0, 0)
    
    # If B&B found a better solution, use it.
    # Otherwise, use the best greedy one.
    
    # The B&B updates best_clique_mask globally.
    
    # Convert best_clique_mask (candidate indices) to original values
    result = []
    m = best_clique_mask
    while m:
        lsb = m & -m
        ci = lsb.bit_length() - 1
        orig_v = cand_to_orig[ci]
        result.append(orig_v + 1)
        m ^= lsb
        
    return sorted(result)

# EVOLVE-BLOCK-END
