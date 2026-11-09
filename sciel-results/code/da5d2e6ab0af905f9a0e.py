"""Program for R_5(C4) maximization using Simulated Annealing with Bitmask Optimization."""
import random
import time
import numpy as np

def construct():
    """
    Finds a valid 5-coloring of K_n without monochromatic C4s.
    Uses Simulated Annealing with bitmask-based fast C4 counting.
    Tries n=28, 27, 26 in order.
    """
    
    def count_c4_bitmask(n, colors):
        """
        Counts the number of 'bad' pairs (u, v, c) where u and v share >= 2 neighbors in color c.
        Actually, it sums binom(common_neighbors, 2) for all pairs.
        """
        # adj[c][u] is a bitmask of neighbors of u in color c
        adj = [[0] * n for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        total_bad = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj[c][u] & adj[c][v]
                    k = common.bit_count()
                    if k >= 2:
                        total_bad += k * (k - 1) // 2
        return total_bad

    def simulated_annealing(n, time_limit=10.0):
        """
        Runs Simulated Annealing to find a valid coloring.
        Returns (n, colors) if valid, else None.
        """
        random.seed(42) # Deterministic start for reproducibility in early stages, or random for exploration?
        # Let's use a fixed seed for the first attempt to ensure we don't waste time on bad luck, 
        # but SA itself uses randomness.
        
        m = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(m)]
        
        # Precompute edge indices for (i, j)
        # We need to map (i, j) to index in colors list
        # And vice versa
        edge_to_idx = {}
        idx_to_edge = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_to_idx[(i, j)] = idx
                idx_to_edge[idx] = (i, j)
                idx += 1
        
        current_energy = count_c4_bitmask(n, colors)
        
        if current_energy == 0:
            return n, colors
        
        T_start = 1.0
        T_end = 0.001
        alpha = 0.999 # Cooling rate
        start_time = time.time()
        
        T = T_start
        while time.time() - start_time < time_limit:
            if current_energy == 0:
                return n, colors
            
            # Pick random edge
            e_idx = random.randint(0, m - 1)
            i, j = idx_to_edge[e_idx]
            old_c = colors[e_idx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
            
            # Update colors
            colors[e_idx] = new_c
            
            # Calculate new energy
            # For efficiency, we could update incrementally, but bitmask count is fast enough
            # Let's check how fast it is. 
            # n=28, 5 colors. 5 * 28*27/2 = 1890 pairs.
            # Each pair does AND and bit_count.
            # 1890 ops is very fast in Python? Maybe ~0.1ms?
            # If so, we can do 10,000 iterations in 1s.
            
            new_energy = count_c4_bitmask(n, colors)
            
            delta = new_energy - current_energy
            
            if delta < 0:
                current_energy = new_energy
            else:
                # Accept with probability exp(-delta / T)
                if T > 0:
                    p = np.exp(-delta / T)
                    if random.random() < p:
                        current_energy = new_energy
                    else:
                        # Revert
                        colors[e_idx] = old_c
                # else T is 0, only accept improvements
            
            T *= alpha
            if T < T_end:
                T = T_end
        
        if current_energy == 0:
            return n, colors
        
        return None

    # Try n from 28 down to 26
    # We allocate time budget
    # 28: 10s, 27: 10s, 26: 5s (should be easy)
    for n in [28, 27, 26]:
        t_limit = 10.0 if n > 26 else 5.0
        result = simulated_annealing(n, t_limit)
        if result:
            return {'n': result[0], 'colors': result[1]}
            
    # Fallback: Known valid n=11 or just return n=26 if we assume it's always found?
    # If SA fails for 26 (unlikely), we should have a hardcoded 26 or 11.
    # Let's try a simple greedy for 26 as a last resort?
    # Or just return n=11 cyclic if all else fails.
    
    # Cyclic 11 fallback
    n_val = 11
    colors_val = []
    for i in range(n_val):
        for j in range(i + 1, n_val):
            d = (j - i) % n_val
            d = min(d, n_val - d)
            colors_val.append(d - 1)
            
    return {'n': n_val, 'colors': colors_val}
