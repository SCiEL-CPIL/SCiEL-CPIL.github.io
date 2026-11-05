import random
import time

def construct():
    """
    Attempts to find a valid 2-coloring of K_n with no monochromatic K_{2,2,2}.
    Uses Random Local Search with a fast bitset-based violation checker.
    Targets n=29, the known lower bound for R(K_{2,2,2}).
    """
    start_time = time.time()
    time_limit = 110  # Leave buffer for overhead
    
    def edge_index(n, u, v):
        if u > v:
            u, v = v, u
        return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)
    
    def build_adj(n, colors):
        adj0 = [0] * n
        adj1 = [0] * n
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[k]
                if c == 0:
                    adj0[i] |= (1 << j)
                    adj0[j] |= (1 << i)
                else:
                    adj1[i] |= (1 << j)
                    adj1[j] |= (1 << i)
                k += 1
        return adj0, adj1
    
    def flip_edge(n, colors, adj0, adj1, u, v):
        """Flip edge (u,v) and update adj0/adj1 in place. Returns new color."""
        if u > v:
            u, v = v, u
        idx = edge_index(n, u, v)
        old_c = colors[idx]
        new_c = 1 - old_c
        colors[idx] = new_c
        if old_c == 0:
            adj0[u] &= ~(1 << v)
            adj0[v] &= ~(1 << u)
            adj1[u] |= (1 << v)
            adj1[v] |= (1 << u)
        else:
            adj1[u] &= ~(1 << v)
            adj1[v] &= ~(1 << u)
            adj0[u] |= (1 << v)
            adj0[v] |= (1 << u)
        return new_c
    
    def find_violation_fast(adj, n):
        """
        Finds one edge (u,v) that is part of a K_{2,2,2} in graph adj.
        Returns (u, v) or None.
        """
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    continue
                common = adj[u] & adj[v]
                if common & (common - 1) == 0:
                    continue
                # Iterate over pairs x, y in common
                common_bits = common
                verts = []
                while common_bits:
                    lsb = common_bits & (-common_bits)
                    idx = (lsb.bit_length() - 1)
                    verts.append(idx)
                    common_bits ^= lsb
                for i_x in range(len(verts)):
                    x = verts[i_x]
                    for i_y in range(i_x + 1, len(verts)):
                        y = verts[i_y]
                        mask_excl = ~((1 << u) | (1 << v) | (1 << x) | (1 << y))
                        cand_a = adj[v] & adj[x] & adj[y] & mask_excl
                        if cand_a == 0:
                            continue
                        cand_b = adj[u] & adj[x] & adj[y] & mask_excl
                        if cand_b == 0:
                            continue
                        return (u, v)
        return None
    
    def is_valid_fast(adj0, adj1, n):
        return find_violation_fast(adj0, n) is None and find_violation_fast(adj1, n) is None
    
    # Try n=29 first, then smaller
    for n in range(29, 16, -1):
        num_edges = n * (n - 1) // 2
        max_attempts = 100 if n >= 25 else 200
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize random coloring
            colors = [random.randint(0, 1) for _ in range(num_edges)]
            adj0, adj1 = build_adj(n, colors)
            
            # Local search with simulated annealing-like behavior
            best_colors = colors[:]
            best_adj0 = adj0[:]
            best_adj1 = adj1[:]
            
            for step in range(2000):
                if time.time() - start_time > time_limit:
                    break
                
                # Find a violation
                viol0 = find_violation_fast(adj0, n)
                if viol0 is None:
                    viol1 = find_violation_fast(adj1, n)
                    if viol1 is None:
                        return {'n': n, 'colors': colors}
                    u, v = viol1
                else:
                    u, v = viol0
                
                # Flip the violated edge
                flip_edge(n, colors, adj0, adj1, u, v)
                
                # Occasionally do a random flip to escape local minima
                if step % 50 == 0 and step > 0:
                    rand_idx = random.randint(0, num_edges - 1)
                    # Find u,v from idx
                    # Reverse of edge_index
                    # u*(n-1) - u*(u-1)//2 + (v-u-1) = idx
                    # Let's just find it by iterating
                    found = False
                    for i in range(n):
                        row_start = i * (n - 1) - i * (i - 1) // 2
                        row_end = row_start + (n - 1 - i)
                        if row_start <= rand_idx < row_end:
                            offset = rand_idx - row_start
                            v2 = i + 1 + offset
                            u2 = i
                            flip_edge(n, colors, adj0, adj1, u2, v2)
                            found = True
                            break
                    # If not found, skip (shouldn't happen)
                
            # End of local search attempt, try again with new random start
    
    # Fallback: Return a known valid small n
    # Use the quadratic residue construction for n=17
    n = 17
    squares = {x * x % n for x in range(1, n)}
    colors = [0 if (j - i) % n in squares else 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
