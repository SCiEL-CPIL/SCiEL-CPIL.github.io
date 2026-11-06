import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def is_prime(k):
        if k < 2:
            return False
        for i in range(2, int(k**0.5) + 1):
            if k % i == 0:
                return False
        return True

    def paley_coloring(n):
        if not is_prime(n):
            return None
        squares = set()
        for x in range(1, n):
            squares.add((x*x) % n)
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                diff = (j - i) % n
                if diff in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    def is_valid_clean(n, colors):
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        def has_k222(color):
            adj_c = [0]*n
            for i in range(n):
                for j in range(n):
                    if adj[i][j] == color:
                        adj_c[i] |= (1 << j)
            
            for u in range(n):
                mask = adj_c[u]
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    mask ^= lsb
                    if v <= u: continue
                    
                    S_mask = adj_c[u] & adj_c[v]
                    s_list = []
                    m = S_mask
                    while m:
                        lsb = m & -m
                        w = lsb.bit_length() - 1
                        s_list.append(w)
                        m ^= lsb
                    
                    for i in range(len(s_list)):
                        x = s_list[i]
                        for j in range(i+1, len(s_list)):
                            y = s_list[j]
                            common_mask = adj_c[x] & adj_c[y] & S_mask
                            if common_mask.bit_count() >= 2:
                                return True
            return False
        
        return not has_k222(0) and not has_k222(1)

    # Try to find n=30
    p31 = paley_coloring(31)
    if p31:
        # Build adjacency matrix for 31
        adj_31 = [[0]*31 for _ in range(31)]
        idx = 0
        for u in range(31):
            for v in range(u+1, 31):
                c = p31[idx]
                adj_31[u][v] = c
                adj_31[v][u] = c
                idx += 1
        
        # Try removing each vertex
        for remove_v in range(31):
            if time.time() > deadline:
                break
            
            # Create new vertex mapping
            # new_v = old_v if old_v < remove_v else old_v - 1
            # We need to extract colors for the 30-vertex subgraph
            
            colors_30 = []
            for i in range(30):
                for j in range(i+1, 30):
                    # Map new indices to old indices
                    # If i < remove_v, old_i = i. Else old_i = i + 1
                    # Same for j
                    old_i = i if i < remove_v else i + 1
                    old_j = j if j < remove_v else j + 1
                    colors_30.append(adj_31[old_i][old_j])
            
            if is_valid_clean(30, colors_30):
                return {'n': 30, 'colors': colors_30}

    # Fallback to n=29
    p29 = paley_coloring(29)
    if p29 and is_valid_clean(29, p29):
        return {'n': 29, 'colors': p29}

    # Fallback to n=17
    p17 = paley_coloring(17)
    if p17 and is_valid_clean(17, p17):
        return {'n': 17, 'colors': p17}

    return {'n': 1, 'colors': [0]}
