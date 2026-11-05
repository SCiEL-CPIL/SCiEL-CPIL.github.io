import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # We want to maximize n.
    # Lazebnik and Woldar construction gives n=26.
    # We will try to find a valid 5-coloring for n=26, then 25, etc.
    
    def popcount(x):
        return bin(x).count('1')
    
    def solve_n(n, time_budget):
        """
        Attempts to find a valid 5-coloring of K_n with no monochromatic C4.
        Returns list of colors if found, None otherwise.
        """
        if n < 1:
            return None
        if n < 4:
            # No C4 possible
            return [0] * (n * (n - 1) // 2)
            
        num_edges = n * (n - 1) // 2
        
        # Map (i, j) to edge index
        # We need fast access.
        # For n <= 32, we can use bitmasks for adjacency.
        
        # Precompute edge indices
        edge_idx = {}
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx[(i, j)] = k
                edge_idx[(j, i)] = k
                k += 1
                
        # Initialize random colors
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        
        # Build adjacency bitmasks for each color
        # adj[c][v] is an integer where bit u is 1 if edge (v, u) has color c
        adj = [[0] * n for _ in range(5)]
        
        def build_adj():
            for c in range(5):
                for v in range(n):
                    adj[c][v] = 0
            for k, c in enumerate(colors):
                # Recover i, j from k
                # This is slow if done naively. 
                # Better: iterate i, j and compute k.
                pass
            
            # Rebuild properly
            k = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[k]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    k += 1
        
        build_adj()
        
        def count_c4s():
            count = 0
            for c in range(5):
                for i in range(n):
                    for j in range(i + 1, n):
                        common = popcount(adj[c][i] & adj[c][j])
                        if common >= 2:
                            count += common * (common - 1) // 2
            return count
            
        def update_adj(edge_idx_val, old_c, new_c):
            # Get i, j from edge_idx_val
            # We need a reverse map or compute it.
            # Let's store the pairs.
            pass
            
        # To make update_adj fast, let's store the pairs in a list
        pairs = []
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
                k += 1
                
        def update_adj_fast(idx, old_c, new_c):
            i, j = pairs[idx]
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            if new_c != old_c:
                adj[new_c][i] |= (1 << j)
                adj[new_c][j] |= (1 << i)
                
        def find_c4():
            """Returns (i, j, w, x, c) or None"""
            for c in range(5):
                for i in range(n):
                    for j in range(i + 1, n):
                        common = popcount(adj[c][i] & adj[c][j])
                        if common >= 2:
                            # Find two common neighbors
                            mask = adj[c][i] & adj[c][j]
                            # Get two bits
                            # Simple loop
                            bits = []
                            m = mask
                            while m and len(bits) < 2:
                                # Get lowest set bit
                                lsb = m & -m
                                bits.append(lsb.bit_length() - 1)
                                m ^= lsb
                            if len(bits) >= 2:
                                return (i, j, bits[0], bits[1], c)
            return None

        current_c4_count = count_c4s()
        
        # Local Search
        # We try to reduce C4 count.
        # If we reach 0, we are done.
        
        iterations = 0
        max_iter = 1000000 # Cap to avoid infinite loop if time check fails
        
        while current_c4_count > 0:
            if time.time() > start_time + time_budget:
                return None
            
            iterations += 1
            if iterations > max_iter:
                return None
                
            # Find a C4
            c4 = find_c4()
            if c4 is None:
                # Should not happen if count > 0, but safety
                break
                
            i, j, w, x, c = c4
            # The C4 edges are (i,w), (w,j), (j,x), (x,i)
            # Note: The C4 is i-w-j-x-i.
            # Edges: (i,w), (w,j), (j,x), (x,i)
            
            edges_to_try = [(i, w), (w, j), (j, x), (x, i)]
            random.shuffle(edges_to_try)
            
            improved = False
            for u, v in edges_to_try:
                idx = edge_idx[(u, v)]
                old_c = colors[idx]
                
                # Try changing to other colors
                for new_c in range(5):
                    if new_c == old_c:
                        continue
                    
                    # Apply change
                    update_adj_fast(idx, old_c, new_c)
                    colors[idx] = new_c
                    
                    # Evaluate change
                    # Instead of full count, we can check if this specific C4 is broken
                    # and if new C4s are created.
                    # But full count is safer and n is small.
                    # Optimization: Only count C4s involving the changed edge?
                    # Let's stick to full count for correctness, it's fast enough with bitmasks.
                    
                    new_count = count_c4s()
                    
                    if new_count < current_c4_count:
                        current_c4_count = new_count
                        improved = True
                        break
                    else:
                        # Revert
                        update_adj_fast(idx, new_c, old_c)
                        colors[idx] = old_c
                        
                if improved:
                    break
                    
            if not improved:
                # Stuck. Try a random move that doesn't increase count too much.
                random_idx = random.randint(0, num_edges - 1)
                old_c = colors[random_idx]
                new_c = random.randint(0, 4)
                
                update_adj_fast(random_idx, old_c, new_c)
                colors[random_idx] = new_c
                new_count = count_c4s()
                
                if new_count <= current_c4_count + 5: # Allow small increase
                    current_c4_count = new_count
                else:
                    # Revert
                    update_adj_fast(random_idx, new_c, old_c)
                    colors[random_idx] = old_c
                    # If we can't even do a small increase, maybe restart?
                    # For now, just break and try next n? No, try again with random restart.
                    # Let's just return None if we are stuck for a while?
                    # Simpler: just break and hope the next attempt (if we had restarts) works.
                    # Since we don't have restarts in this simple loop, we break.
                    break

        if current_c4_count == 0:
            return colors
        else:
            return None

    # Try n from 26 down to 11
    # Allocate time budget based on remaining time.
    # Give more time to larger n.
    
    n_values = [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]
    
    # Weighted time budget
    total_weight = sum(100 / (27 - n) for n in n_values)
    
    for n in n_values:
        if time.time() > deadline - 5:
            break
            
        # Time budget for this n
        weight = 100 / (27 - n)
        time_budget = (weight / total_weight) * (deadline - time.time() - 2)
        
        if time_budget < 1:
            time_budget = 1
            
        result = solve_n(n, time_budget)
        
        if result is not None:
            return {'n': n, 'colors': result}
            
    # If all failed, return a trivial valid solution for n=3 or something?
    # The task says 1 <= n <= 29.
    # n=3 is always valid (no C4).
    n = 3
    colors = [0, 0, 0]
    return {'n': n, 'colors': colors}
