import time
import random
from itertools import combinations

def _is_not_3_colorable(adj, n):
    """Check if graph is NOT 3-colorable using fast bitmask backtracking."""
    # Build neighbor masks
    neigh = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i][j]:
                mask |= (1 << j)
        neigh[i] = mask
    
    # Order by degree descending, then by vertex index for determinism
    deg = [bin(neigh[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: (-deg[x], x))
    
    # Map vertex -> position in order
    pos_of = [0] * n
    for idx, v in enumerate(order):
        pos_of[v] = idx
    
    # For each vertex v in order, compute bitmask of neighbors that come before v
    earlier_neigh = [0] * n
    for idx, v in enumerate(order):
        mask = 0
        nv = neigh[v]
        for u in range(n):
            if (nv >> u) & 1:
                if pos_of[u] < idx:
                    mask |= (1 << pos_of[u])
        earlier_neigh[v] = mask
    
    # color_masks[c] has bit idx set if order[idx] has color c
    color_masks = [0, 0, 0]
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        em = earlier_neigh[v]
        forbidden = 0
        if em & color_masks[0]:
            forbidden |= 1
        if em & color_masks[1]:
            forbidden |= 2
        if em & color_masks[2]:
            forbidden |= 4
        available = 7 & ~forbidden
        if available == 0:
            return False
        while available:
            c_bit = available & (-available)
            c = c_bit.bit_length() - 1
            color_masks[c] |= (1 << idx)
            if backtrack(idx + 1):
                return True
            color_masks[c] ^= (1 << idx)
            available ^= c_bit
        return False
    
    # Symmetry breaking: first vertex gets color 0
    color_masks[0] = 1
    return not backtrack(1)

def _count_edges(adj, n):
    return sum(sum(adj[i]) for i in range(n)) // 2

def _get_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges

def _groetzsch_adj(n=16):
    """Create adjacency matrix for Grötzsch graph on vertices 0-10."""
    adj = [[0]*n for _ in range(n)]
    # Cycle on 0-4
    for i in range(5):
        u, v = i, (i+1) % 5
        adj[u][v] = adj[v][u] = 1
    # Vertices 5-9: each i connects to 5+(i+1)%5 and 5+(i-1)%5
    for i in range(5):
        adj[i][5+(i+1)%5] = adj[5+(i+1)%5][i] = 1
        adj[i][5+(i-1)%5] = adj[5+(i-1)%5][i] = 1
    # Vertex 10 connects to all of 5-9
    for u in range(5):
        adj[5+u][10] = adj[10][5+u] = 1
    return adj

def _find_max_independent_sets(adj, n, count, size):
    """Find 'count' independent sets of size 'size' in the graph."""
    sets = []
    vertices = list(range(n))
    
    def is_independent(S):
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if adj[S[i]][S[j]]:
                    return False
        return True

    for comb in combinations(vertices, size):
        if is_independent(comb):
            sets.append(list(comb))
            if len(sets) >= count:
                break
    return sets

def _compute_cn(adj, n):
    """Precompute common neighbor masks for triangle checking."""
    cn = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            mask = 0
            for k in range(n):
                if k != i and k != j and adj[i][k] and adj[j][k]:
                    mask |= (1 << k)
            cn[i][j] = mask
            cn[j][i] = mask
    return cn

def _add_edge_cn(adj, cn, n, i, j):
    """Add edge (i,j) and update common neighbor masks."""
    adj[i][j] = 1
    adj[j][i] = 1
    for k in range(n):
        if k != i and k != j:
            if adj[i][k]:
                cn[i][k] |= (1 << j)
                cn[k][i] |= (1 << j)
            if adj[j][k]:
                cn[j][k] |= (1 << i)
                cn[k][j] |= (1 << i)
    cn[i][j] = 0
    cn[j][i] = 0

def _remove_edge_cn(adj, cn, n, i, j):
    """Remove edge (i,j) and update common neighbor masks."""
    adj[i][j] = 0
    adj[j][i] = 0
    for k in range(n):
        if k != i and k != j:
            if adj[i][k]:
                cn[i][k] &= ~(1 << j)
                cn[k][i] &= ~(1 << j)
            if adj[j][k]:
                cn[j][k] &= ~(1 << i)
                cn[k][j] &= ~(1 << i)
    cn[i][j] = 0
    cn[j][i] = 0

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    adj = _groetzsch_adj(n)
    
    # Find independent sets of size 5 in the Grötzsch part (vertices 0-10)
    iso_sets = _find_max_independent_sets(adj, 11, 5, 5)
    
    if len(iso_sets) < 5:
        while len(iso_sets) < 5:
            iso_sets.append(iso_sets[0] if iso_sets else [])
    
    # Connect new vertices 11-15 to the independent sets
    for i in range(5):
        v_new = 11 + i
        for u in iso_sets[i]:
            adj[v_new][u] = 1
            adj[u][v_new] = 1
    
    best_adj = [row[:] for row in adj]
    best_edges = _count_edges(adj, n)
    
    cn = _compute_cn(adj, n)
    
    # Local Search
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        # Build non-edges list, sorted by potential value (fewer common neighbors = more likely to be triangle-free)
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    # Score: prefer non-edges with fewer common neighbors
                    non_edges.append((cn[i][j].bit_count(), i, j))
        random.shuffle(non_edges)
        non_edges.sort(key=lambda x: x[0])  # Sort by fewest common neighbors first
        
        improved = False
        
        # Phase 1: Try adding edges (prioritized by triangle-freeness likelihood)
        for (cn_count, i, j) in non_edges:
            if time.time() > deadline:
                break
            # Check triangle using precomputed common neighbors
            if cn[i][j] != 0:
                continue
            
            # Tentatively add
            _add_edge_cn(adj, cn, n, i, j)
            
            if _is_not_3_colorable(adj, n):
                new_edges = _count_edges(adj, n)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = [row[:] for row in adj]
                improved = True
            else:
                # Revert
                _remove_edge_cn(adj, cn, n, i, j)
        
        # Phase 2: Try remove+add moves
        if not improved and time.time() < deadline:
            edge_list = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edge_list.append((i, j))
            random.shuffle(edge_list)
            
            for (ri, rj) in edge_list[:30]:
                if time.time() > deadline:
                    break
                
                # Remove edge (ri, rj)
                _remove_edge_cn(adj, cn, n, ri, rj)
                
                # Try adding non-edges
                added = False
                for (ai, aj) in non_edges:
                    if time.time() > deadline:
                        break
                    if adj[ai][aj]:
                        continue
                    if cn[ai][aj] != 0:
                        continue
                    
                    _add_edge_cn(adj, cn, n, ai, aj)
                    
                    if _is_not_3_colorable(adj, n):
                        added = True
                        break
                    else:
                        _remove_edge_cn(adj, cn, n, ai, aj)
                
                # Re-add removed edge
                _add_edge_cn(adj, cn, n, ri, rj)
                
                if added:
                    new_edges = _count_edges(adj, n)
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_adj = [row[:] for row in adj]
                    improved = True
    
    return _get_edges(best_adj, n)
