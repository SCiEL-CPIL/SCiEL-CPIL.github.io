# EVOLVE-BLOCK-START
import random
import time

def construct():
    """
    Finds a 4-coloring of K_n with no monochromatic C6.
    Uses a Repair Heuristic with Bitset-based fast C6 detection.
    """
    start_time = time.time()
    time_limit = 110  # Leave 10s margin
    
    # We want to maximize n. Best known is 17.
    for n in [17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() - start_time > time_limit:
            break
            
        m = n * (n - 1) // 2
        edge_indices = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_indices[(i, j)] = idx
                edge_indices[(j, i)] = idx
                idx += 1
        
        # Bitset adjacency lists for each color
        # adj[c][i] is an integer bitmask of neighbors of i in color c
        
        def find_one_c6(adj):
            """
            Returns a list [v0, v1, v2, v3, v4, v5, c] forming a C6 in color c, or None.
            """
            for c in range(4):
                adj_c = adj[c]
                for v0 in range(n):
                    mask_v0 = adj_c[v0]
                    temp = mask_v0
                    while temp:
                        lsb = temp & -temp
                        v1 = lsb.bit_length() - 1
                        temp -= lsb
                        
                        mask_v1 = adj_c[v1]
                        candidates_v2 = mask_v1 & ~(1 << v0)
                        temp2 = candidates_v2
                        while temp2:
                            lsb2 = temp2 & -temp2
                            v2 = lsb2.bit_length() - 1
                            temp2 -= lsb2
                            
                            mask_v2 = adj_c[v2]
                            forbidden_v3 = (1 << v0) | (1 << v1)
                            candidates_v3 = mask_v2 & ~forbidden_v3
                            temp3 = candidates_v3
                            while temp3:
                                lsb3 = temp3 & -temp3
                                v3 = lsb3.bit_length() - 1
                                temp3 -= lsb3
                                
                                mask_v3 = adj_c[v3]
                                forbidden_v4 = (1 << v0) | (1 << v1) | (1 << v2)
                                candidates_v4 = mask_v3 & ~forbidden_v4
                                temp4 = candidates_v4
                                while temp4:
                                    lsb4 = temp4 & -temp4
                                    v4 = lsb4.bit_length() - 1
                                    temp4 -= lsb4
                                    
                                    mask_v4 = adj_c[v4]
                                    common = mask_v4 & mask_v0
                                    forbidden_v5 = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                                    valid_v5 = common & ~forbidden_v5
                                    
                                    if valid_v5:
                                        lsb5 = valid_v5 & -valid_v5
                                        v5 = lsb5.bit_length() - 1
                                        return [v0, v1, v2, v3, v4, v5, c]
            return None

        # Run repair for n
        for attempt in range(20): # Try multiple random inits
            if time.time() - start_time > time_limit:
                break
                
            colors = [random.randint(0, 3) for _ in range(m)]
            adj = [[0] * n for _ in range(4)]
            for (i, j), idx in edge_indices.items():
                if i > j: continue
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
            
            # Repair loop
            for step in range(2000): # Max repair steps
                if time.time() - start_time > time_limit:
                    return {'n': 1, 'colors': [0]} # Timeout fallback
                
                c6_info = find_one_c6(adj)
                if c6_info is None:
                    # Valid coloring found
                    return {'n': n, 'colors': colors}
                
                # Break a random edge in the C6
                v0, v1, v2, v3, v4, v5, c = c6_info
                cycle_edges = [(v0, v1), (v1, v2), (v2, v3), (v3, v4), (v4, v5), (v5, v0)]
                u, v = random.choice(cycle_edges)
                idx = edge_indices[(u, v)]
                old_c = colors[idx]
                
                # Try a random different color
                new_c = random.randint(0, 3)
                if new_c == old_c:
                    new_c = (new_c + 1) % 4
                    
                colors[idx] = new_c
                adj[old_c][u] &= ~(1 << v)
                adj[old_c][v] &= ~(1 << u)
                adj[new_c][u] |= (1 << v)
                adj[new_c][v] |= (1 << u)
            
            # If we didn't find a solution in 2000 steps, try next random init
    
    # Fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
# EVOLVE-BLOCK-END
