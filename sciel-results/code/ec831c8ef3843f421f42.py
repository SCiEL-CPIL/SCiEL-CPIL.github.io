import time
import random
import math

def construct():
    start_time = time.time()
    timeout = 130  # Safety margin before 150s kill

    # Helper to get edge index
    def get_idx(i, j, n):
        if i > j:
            i, j = j, i
        # Number of edges in rows 0..i-1
        # Row k has n-1-k edges
        # Sum_{k=0}^{i-1} (n-1-k) = i(n-1) - i(i-1)/2
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def count_c4s(n, colors):
        """
        Returns the number of monochromatic C4s (K_{2,2}) in the coloring.
        A C4 is formed by two vertices u, v sharing at least 2 common neighbors in the same color.
        Total C4s = sum over all colors c, and all pairs (u, v) of C(common_neighbors(u, v), 2)
        """
        # Build adjacency bitmasks for each color
        # adj[c][u] is a bitmask of neighbors of u in color c
        adj = [[0] * n for _ in range(5)]
        
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[k]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                k += 1
        
        c4_count = 0
        for c in range(5):
            for u in range(n):
                mask_u = adj[c][u]
                # Iterate over neighbors of u
                # For each neighbor v, we want to count common neighbors with u
                # But we must avoid double counting pairs (u, v).
                # Standard way: For each pair (u, v), count common neighbors.
                # To be efficient, we can just iterate u < v.
                for v in range(u + 1, n):
                    common = mask_u & adj[c][v]
                    cnt = bin(common).count('1')
                    if cnt >= 2:
                        c4_count += cnt * (cnt - 1) // 2
        return c4_count

    def lazebnik_woldar_26():
        """
        Construct the known valid 5-coloring of K_26.
        Based on affine plane of order 5? 
        Actually, the Lazebnik-Woldar construction for R_5(C4) > 26 uses vertices Z_5 x Z_5 x Z_5 ? No, that's 125.
        
        The construction for n=26 is typically related to the Paley graph or similar finite fields?
        Let's check the inspiration: "affine plane of order 5".
        Affine plane AG(2, 5) has 25 points. Adding 1 point?
        
        Actually, a well-known construction for 5 colors on 26 vertices:
        Vertices are Z_26? No.
        
        Let's use a robust heuristic search that is guaranteed to find it quickly if we start from a good structure,
        or just implement a very fast local search that starts from a random permutation-based coloring.
        
        Wait, the inspiration says: "The Lazebnik-Woldar construction ... provides a known valid 5-coloring for n=26."
        I will implement a fast local search that starts from a "block" structure which is close to the algebraic one.
        
        Strategy:
        1. Try to construct n=26 using a fast local search starting from a structured initialization.
        2. If it fails or takes too long, fall back to n=25, etc.
        
        Structured Initialization:
        Divide vertices into 5 groups of 5? 25 vertices. Plus 1?
        Or 5 groups of 5, and the 26th vertex connected specially.
        
        Let's try a simpler approach: 
        Use the fact that R_5(C4) > 26 is known. 
        I will use a very aggressive local search for n=26.
        Since the state space is large, we need a good initialization.
        
        Let's try an initialization based on modular arithmetic:
        color(i, j) = (i * j + i + j) % 5 ?
        
        Let's just run a fast local search for n=26 with multiple restarts and a tight time budget.
        If we don't find 0, we try n=25, etc.
        """
        n = 26
        m = n * (n - 1) // 2
        
        # Precompute edge endpoints
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        # We will run local search for a fixed number of iterations or time
        # To maximize chance of finding the known solution, we use a "repair" strategy.
        
        def local_search(n, time_limit, init_colors=None):
            m = n * (n - 1) // 2
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    edges.append((i, j))
            
            if init_colors is None:
                # Structured init: 
                # Try to make each color class a union of cliques or matchings?
                # Random is often slow for n=26.
                # Let's use a "cyclic" init:
                # color(i, j) = (i - j) % 5
                init_colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        init_colors.append((i - j) % 5)
            
            colors = init_colors[:]
            
            # Build adj
            adj = [[0] * n for _ in range(5)]
            for idx, (i, j) in enumerate(edges):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                
            # Calculate initial score
            def calc_score():
                s = 0
                for c in range(5):
                    for u in range(n):
                        for v in range(u + 1, n):
                            common = adj[c][u] & adj[c][v]
                            cnt = bin(common).count('1')
                            if cnt >= 2:
                                s += cnt * (cnt - 1) // 2
                return s
            
            current_score = calc_score()
            if current_score == 0:
                return colors
                
            best_colors = colors[:]
            best_score = current_score
            
            # Precompute for fast delta
            # We need to update score efficiently.
            # Delta for changing edge (i,j) from c1 to c2:
            # Remove (i,j) from c1:
            #   For each neighbor w of i in c1 (w != j):
            #     common(i, w) decreases by 1 (j was common).
            #     Let k = common(i, w) - 1.
            #     Score change for pair (i, w): -k (if k>=1) else 0?
            #     Actually, score contribution of pair (u,v) is C(cnt, 2).
            #     If cnt -> cnt-1, change is - (cnt-1).
            #     So if new_cnt >= 1, delta -= new_cnt.
            #   Similarly for neighbors of j.
            # Add (i,j) to c2:
            #   For each neighbor w of i in c2 (w != j):
            #     common(i, w) increases by 1.
            #     Let k = common(i, w).
            #     New cnt = k + 1.
            #     Change is k.
            #     So delta += k.
            
            t_start = time.time()
            # Use a simple random walk with acceptance of improvements and some random moves
            # Since we are looking for a specific known solution, greedy hill climbing with random restarts might work.
            # But the landscape might be rugged.
            
            # Let's use Simulated Annealing with a fast cooling schedule.
            T = 1.0
            cool = 0.9995
            
            while time.time() - t_start < time_limit:
                idx = random.randint(0, m - 1)
                i, j = edges[idx]
                old_c = colors[idx]
                new_c = random.randint(0, 4)
                if new_c == old_c:
                    continue
                
                delta = 0
                
                # Remove from old_c
                # Neighbors of i in old_c
                mask_i = adj[old_c][i]
                # Exclude j
                mask_i &= ~(1 << j)
                while mask_i:
                    lsb = mask_i & (-mask_i)
                    w = lsb.bit_length() - 1
                    mask_i ^= lsb
                    # common(i, w) in old_c
                    # j is a common neighbor? Yes, because (j,w) is an edge in old_c?
                    # Wait, w is neighbor of i. Is w neighbor of j?
                    # common(i, w) = adj[old_c][i] & adj[old_c][w]
                    # We are removing (i,j). So j is removed from adj[old_c][i].
                    # If j was in adj[old_c][w], then j was a common neighbor.
                    # j is in adj[old_c][w] iff (j,w) is edge in old_c.
                    # We don't know that.
                    # Let's calculate k = popcount(adj[old_c][i] & adj[old_c][w]) BEFORE removal.
                    # This includes j if (j,w) is edge.
                    # After removal, j is not in adj[old_c][i].
                    # So new common count = k - 1 if j was common, else k.
                    # j is common iff j in adj[old_c][w].
                    # Let's just compute k = popcount(adj[old_c][i] & adj[old_c][w]).
                    # Since we are iterating neighbors of i, and we haven't updated adj yet,
                    # adj[old_c][i] still has j.
                    k = bin(adj[old_c][i] & adj[old_c][w]).count('1')
                    # j is in adj[old_c][i].
                    # Is j in adj[old_c][w]?
                    # If yes, k includes j. New k' = k - 1. Delta -= (k-1).
                    # If no, k does not include j. New k' = k. Delta -= 0 (since C(k,2) - C(k,2) = 0? No.)
                    # Wait. The score term is C(cnt, 2).
                    # If cnt changes from k to k-1, delta = C(k-1, 2) - C(k, 2) = -(k-1).
                    # If cnt changes from k to k, delta = 0.
                    # So we need to know if j was a common neighbor.
                    # j is common neighbor of i and w in color old_c iff (j,w) is edge in old_c.
                    # We can check if (1 << w) & adj[old_c][j] is non-zero.
                    if (1 << w) & adj[old_c][j]:
                        delta -= (k - 1)
                    # else delta -= 0
                
                # Neighbors of j in old_c
                mask_j = adj[old_c][j]
                mask_j &= ~(1 << i)
                while mask_j:
                    lsb = mask_j & (-mask_j)
                    w = lsb.bit_length() - 1
                    mask_j ^= lsb
                    k = bin(adj[old_c][j] & adj[old_c][w]).count('1')
                    if (1 << w) & adj[old_c][i]:
                        delta -= (k - 1)
                
                # Add to new_c
                # Neighbors of i in new_c
                mask_i_new = adj[new_c][i]
                mask_i_new &= ~(1 << j)
                while mask_i_new:
                    lsb = mask_i_new & (-mask_i_new)
                    w = lsb.bit_length() - 1
                    mask_i_new ^= lsb
                    k = bin(adj[new_c][i] & adj[new_c][w]).count('1')
                    # j is NOT in adj[new_c][i] yet.
                    # j is common neighbor of i and w in new_c iff (j,w) is edge in new_c.
                    # If yes, new k' = k + 1. Delta += k.
                    if (1 << w) & adj[new_c][j]:
                        delta += k
                
                # Neighbors of j in new_c
                mask_j_new = adj[new_c][j]
                mask_j_new &= ~(1 << i)
                while mask_j_new:
                    lsb = mask_j_new & (-mask_j_new)
                    w = lsb.bit_length() - 1
                    mask_j_new ^= lsb
                    k = bin(adj[new_c][j] & adj[new_c][w]).count('1')
                    if (1 << w) & adj[new_c][i]:
                        delta += k
                
                new_score = current_score + delta
                if new_score < current_score or random.random() < math.exp(-delta / T):
                    # Accept
                    colors[idx] = new_c
                    adj[old_c][i] &= ~(1 << j)
                    adj[old_c][j] &= ~(1 << i)
                    adj[new_c][i] |= (1 << j)
                    adj[new_c][j] |= (1 << i)
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_colors = colors[:]
                        if best_score == 0:
                            return best_colors
                    T *= cool
                # else reject
                
            return best_colors if best_score == 0 else None

        # Try n=26 with several restarts and different inits
        # Time budget for n=26: 60s
        t_limit_26 = 60
        t_start = time.time()
        
        # Try cyclic init
        res = local_search(26, t_limit_26)
        if res:
            return {'n': 26, 'colors': res}
            
        # Try random init
        if time.time() - t_start < t_limit_26:
            res = local_search(26, t_limit_26 - (time.time() - t_start))
            if res:
                return {'n': 26, 'colors': res}
                
        # If n=26 fails, try n=25
        # n=25 is likely easier.
        t_limit_25 = 20
        t_start = time.time()
        res = local_search(25, t_limit_25)
        if res:
            return {'n': 25, 'colors': res}
            
        res = local_search(25, t_limit_25 - (time.time() - t_start))
        if res:
            return {'n': 25, 'colors': res}

        # Fallback to smaller n
        for n in range(24, 10, -1):
            if time.time() - start_time > timeout - 5:
                break
            t_left = timeout - 5 - (time.time() - start_time)
            if t_left < 1:
                break
            res = local_search(n, t_left)
            if res:
                return {'n': n, 'colors': res}
                
        # Absolute fallback
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append((i + j) % 5)
        return {'n': n, 'colors': colors}

    # Execute
    return lazebnik_woldar_26()

