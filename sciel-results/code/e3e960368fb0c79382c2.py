import time
import numpy as np
from itertools import combinations

def construct():
    N = 40
    M = 401
    deadline = time.time() + 110
    
    all_offsets = list(range(1, 20))
    
    best_edges = None
    best_score = float('inf')
    
    # Precompute bitmasks for vertices? No, we construct the graph.
    # For a circulant graph defined by S, the adjacency of vertex i is determined by S.
    # adj[i] = union of bits at (i+s)%N and (i-s)%N for s in S.
    
    for S in combinations(all_offsets, 10):
        if time.time() > deadline - 5:
            break
            
        S_set = set(S)
        
        # Calculate max book size for the circulant part
        # and store book sizes for edges (i, i+d) for d in S
        # Due to symmetry, all edges with offset d have the same book size.
        # Book size for offset d is |S \cap (S-d)|.
        
        max_circ_book = 0
        book_by_offset = {}
        
        for d in S:
            count = 0
            for s in S:
                if (s - d) % 40 in S_set:
                    count += 1
            book_by_offset[d] = count
            if count > max_circ_book:
                max_circ_book = count
                
        if max_circ_book > best_score:
            continue
            
        # Build adjacency as bitsets (Python integers)
        # adj[i] is an integer where bit j is 1 if edge (i,j) exists.
        adj = [0] * N
        for i in range(N):
            mask = 0
            for s in S:
                j1 = (i + s) % N
                j2 = (i - s) % N
                mask |= (1 << j1)
                mask |= (1 << j2)
            adj[i] = mask
            
        # Get list of edges
        edges = []
        for i in range(N):
            # Iterate over set bits in adj[i] that are > i
            mask = adj[i]
            # Clear bits <= i
            # We can just check all j > i
            for j in range(i+1, N):
                if (mask >> j) & 1:
                    edges.append((i, j))
                    
        edge_set = set(edges)
        all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
        non_edges = [p for p in all_pairs if p not in edge_set]
        
        # Precompute book sizes for all edges in base graph
        # b(i,j) = popcount(adj[i] & adj[j])
        base_books = {}
        max_base_book = 0
        for i, j in edges:
            b = (adj[i] & adj[j]).bit_count()
            base_books[(i,j)] = b
            if b > max_base_book:
                max_base_book = b
                
        # For each non-edge, evaluate the new max book
        for u, v in non_edges:
            # New edge (u,v)
            common_mask = adj[u] & adj[v]
            b_new = common_mask.bit_count()
            
            # Edges (u, a) and (v, a) for a in common_mask increase by 1
            # We need to check if any of these increased books exceeds current max.
            # Also, the new edge (u,v) has book size b_new.
            
            current_max = max_base_book
            
            if b_new > current_max:
                current_max = b_new
                
            # Check increased edges
            # Iterate over set bits in common_mask
            cm = common_mask
            while cm:
                # Get lowest set bit
                lsb = cm & -cm
                a = lsb.bit_length() - 1
                cm ^= lsb
                
                # Edge (u, a)
                # Is (u,a) an edge? Yes, a is in adj[u].
                # We need book size of (u,a) in base graph.
                # Note: (u,a) might be the same as (v,a) if u==v (impossible) or if we count undirected.
                # We stored base_books with (min, max).
                
                ua = (u, a) if u < a else (a, u)
                va = (v, a) if v < a else (a, v)
                
                bua = base_books.get(ua, 0)
                bva = base_books.get(va, 0)
                
                if bua + 1 > current_max:
                    current_max = bua + 1
                if bva + 1 > current_max:
                    current_max = bva + 1
                    
            if current_max < best_score:
                best_score = current_max
                final_edges = [list(e) for e in edges]
                final_edges.append([u, v])
                best_edges = final_edges
                
                if best_score == 7:
                    return best_edges

    if best_edges:
        return best_edges
    
    # Fallback: Random graph
    import random
    random_edges = []
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(pairs)
    for u, v in pairs[:M]:
        random_edges.append([u, v])
    return random_edges
