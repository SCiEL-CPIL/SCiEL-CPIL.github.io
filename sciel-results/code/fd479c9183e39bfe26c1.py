import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # We try n=17 first, as it is the best known.
    # If we find it, we return it.
    # We could try 18, but it's likely to time out without a solution.
    # Let's focus on finding a valid 17-vertex coloring.
    
    n = 17
    
    # Edge ordering: Star order
    # (0,1), (0,2), ..., (0, n-1), (1,2), ..., (1, n-1), ...
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            
    # Symmetry breaking
    # Fix color of (0,1) to 0
    # Branch (0,2) in {0, 1}
    
    # N[c][i] is bitmask of neighbors of i in color c
    N = [[0] * n for _ in range(4)]
    colors = [-1] * (n * (n - 1) // 2)
    
    # Precompute edge index
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = idx
            idx += 1
            
    def has_c6(u, v, c):
        # Check if adding (u,v) to color c creates a C6
        # Need path of length 5 from u to v in existing G_c
        # Path: u - v1 - v2 - v3 - v4 - v
        Nu = N[c][u]
        Nv = N[c][v]
        
        # v1 in Nu
        v1_mask = Nu
        while v1_mask:
            v1 = (v1_mask & -v1_mask).bit_length() - 1
            v1_mask &= v1_mask - 1
            if v1 == v:
                continue
            Nv1 = N[c][v1]
            
            # v4 in Nv
            v4_mask = Nv
            while v4_mask:
                v4 = (v4_mask & -v4_mask).bit_length() - 1
                v4_mask &= v4_mask - 1
                if v4 == u or v4 == v1:
                    continue
                Nv4 = N[c][v4]
                
                # v2 in N(v1), v3 in N(v4), v2 ~ v3
                # Exclude u, v, v1, v4
                exclude = (1 << u) | (1 << v) | (1 << v1) | (1 << v4)
                
                v2_mask = Nv1 & ~exclude
                while v2_mask:
                    v2 = (v2_mask & -v2_mask).bit_length() - 1
                    v2_mask &= v2_mask - 1
                    Nv2 = N[c][v2]
                    
                    # v3 in N(v4) and N(v2)
                    v3_mask = Nv4 & Nv2 & ~exclude & ~(1 << v2)
                    if v3_mask:
                        return True
        return False

    def backtrack(edge_pos):
        nonlocal deadline
        if time.time() > deadline:
            return False
            
        if edge_pos == len(edges):
            return True
            
        u, v = edges[edge_pos]
        
        # Symmetry breaking for first few edges
        if edge_pos == 0:
            # (0,1) must be 0
            c = 0
            N[c][u] |= (1 << v)
            N[c][v] |= (1 << u)
            colors[edge_idx[(u, v)]] = c
            if backtrack(edge_pos + 1):
                return True
            N[c][u] &= ~(1 << v)
            N[c][v] &= ~(1 << u)
            colors[edge_idx[(u, v)]] = -1
            return False
            
        if edge_pos == 1:
            # (0,2) can be 0 or 1
            for c in [0, 1]:
                if not has_c6(u, v, c):
                    N[c][u] |= (1 << v)
                    N[c][v] |= (1 << u)
                    colors[edge_idx[(u, v)]] = c
                    if backtrack(edge_pos + 1):
                        return True
                    N[c][u] &= ~(1 << v)
                    N[c][v] &= ~(1 << u)
                    colors[edge_idx[(u, v)]] = -1
            return False
            
        # General case
        for c in range(4):
            if not has_c6(u, v, c):
                N[c][u] |= (1 << v)
                N[c][v] |= (1 << u)
                colors[edge_idx[(u, v)]] = c
                if backtrack(edge_pos + 1):
                    return True
                N[c][u] &= ~(1 << v)
                N[c][v] &= ~(1 << u)
                colors[edge_idx[(u, v)]] = -1
        return False

    # Start search
    if backtrack(0):
        # Found a solution
        return {'n': n, 'colors': colors}
    else:
        # Fallback to smaller n if 17 fails or times out
        # We should try 16, 15, etc.
        # But 17 is the known max. If we fail, we might have a bug or it's too slow.
        # Let's return the seed result if we fail?
        # Or try n=16?
        
        # Let's try n=16 quickly
        n = 16
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        edge_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx[(i, j)] = idx
                idx += 1
        N = [[0] * n for _ in range(4)]
        colors = [-1] * (n * (n - 1) // 2)
        
        def has_c6_16(u, v, c):
            Nu = N[c][u]
            Nv = N[c][v]
            v1_mask = Nu
            while v1_mask:
                v1 = (v1_mask & -v1_mask).bit_length() - 1
                v1_mask &= v1_mask - 1
                if v1 == v:
                    continue
                Nv1 = N[c][v1]
                v4_mask = Nv
                while v4_mask:
                    v4 = (v4_mask & -v4_mask).bit_length() - 1
                    v4_mask &= v4_mask - 1
                    if v4 == u or v4 == v1:
                        continue
                    Nv4 = N[c][v4]
                    exclude = (1 << u) | (1 << v) | (1 << v1) | (1 << v4)
                    v2_mask = Nv1 & ~exclude
                    while v2_mask:
                        v2 = (v2_mask & -v2_mask).bit_length() - 1
                        v2_mask &= v2_mask - 1
                        Nv2 = N[c][v2]
                        v3_mask = Nv4 & Nv2 & ~exclude & ~(1 << v2)
                        if v3_mask:
                            return True
            return False

        def backtrack_16(edge_pos):
            nonlocal deadline
            if time.time() > deadline:
                return False
            if edge_pos == len(edges):
                return True
            u, v = edges[edge_pos]
            if edge_pos == 0:
                c = 0
                N[c][u] |= (1 << v)
                N[c][v] |= (1 << u)
                colors[edge_idx[(u, v)]] = c
                if backtrack_16(edge_pos + 1):
                    return True
                N[c][u] &= ~(1 << v)
                N[c][v] &= ~(1 << u)
                colors[edge_idx[(u, v)]] = -1
                return False
            if edge_pos == 1:
                for c in [0, 1]:
                    if not has_c6_16(u, v, c):
                        N[c][u] |= (1 << v)
                        N[c][v] |= (1 << u)
                        colors[edge_idx[(u, v)]] = c
                        if backtrack_16(edge_pos + 1):
                            return True
                        N[c][u] &= ~(1 << v)
                        N[c][v] &= ~(1 << u)
                        colors[edge_idx[(u, v)]] = -1
                return False
            for c in range(4):
                if not has_c6_16(u, v, c):
                    N[c][u] |= (1 << v)
                    N[c][v] |= (1 << u)
                    colors[edge_idx[(u, v)]] = c
                    if backtrack_16(edge_pos + 1):
                        return True
                    N[c][u] &= ~(1 << v)
                    N[c][v] &= ~(1 << u)
                    colors[edge_idx[(u, v)]] = -1
            return False

        if backtrack_16(0):
            return {'n': n, 'colors': colors}
        else:
            # Return seed
            n = 9
            return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
