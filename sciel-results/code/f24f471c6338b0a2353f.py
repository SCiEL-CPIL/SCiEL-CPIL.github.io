import time
import random
from itertools import combinations

def _is_not_3_colorable(adj, n):
    """Check if graph is NOT 3-colorable using bitmask backtracking."""
    # Build neighbor masks
    neigh = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i][j]:
                mask |= (1 << j)
        neigh[i] = mask
    
    # Order by degree descending
    deg = [bin(neigh[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    
    # Precompute for each position in order, which earlier positions are neighbors
    pos_of = [0] * n
    for idx, v in enumerate(order):
        pos_of[v] = idx
    
    # For each vertex v in order, compute bitmask of neighbors that come before v in order
    earlier_neigh = [0] * n
    for idx, v in enumerate(order):
        mask = 0
        for u in range(n):
            if (neigh[v] >> u) & 1:
                if pos_of[u] < idx:
                    mask |= (1 << pos_of[u])
        earlier_neigh[v] = mask
    
    # Bitmask for colors: color_masks[c] has bit idx set if order[idx] has color c
    color_masks = [0, 0, 0]
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        # Colors forbidden: those used by earlier neighbors
        em = earlier_neigh[v]
        forbidden = 0
        for c in range(3):
            if em & color_masks[c]:
                forbidden |= (1 << c)
        available = 7 & ~forbidden
        if available == 0:
            return False
        # Try colors in available
        while available:
            c_bit = available & (-available)
            c = c_bit.bit_length() - 1
            color_masks[c] |= (1 << idx)
            if backtrack(idx + 1):
                return True
            color_masks[c] ^= (1 << idx)
            available ^= c_bit
        return False
    
    return not backtrack(0)

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
    # Vertices 5-9
    for i in range(5):
        u, w = i, (i+1) % 5
        adj[u][5+w] = adj[5+w][u] = 1
        adj[w][5+u] = adj[5+u][w] = 1
    # Vertex 10
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
    """Compute common neighbor masks for all pairs."""
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

def _add_edge(adj, cn, i, j, n):
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

def _remove_edge(adj, cn, i, j, n):
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
    
    # Find 5 independent sets of size 5 in the Grötzsch part (0-10)
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
    
    # Simulated Annealing with fast bitset-based constraint evaluator
    current_temp = 1.0
    cooling_rate = 0.995
    min_temp = 0.01
    
    while time.time() < deadline and current_temp > min_temp:
        if time.time() > deadline:
            break
        
        # Build non-edges list
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    non_edges.append((i, j))
        
        if not non_edges:
            break
        
        random.shuffle(non_edges)
        
        # Try a few random moves
        moves_tried = 0
        max_moves = 50
        
        for (i, j) in non_edges[:max_moves]:
            if time.time() > deadline:
                break
            moves_tried += 1
            
            # Check triangle using precomputed common neighbors
            if cn[i][j] != 0:
                continue
            
            # Tentatively add
            _add_edge(adj, cn, i, j, n)
            
            if _is_not_3_colorable(adj, n):
                new_edges = _count_edges(adj, n)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = [row[:] for row in adj]
                # Accept the move
                current_temp *= cooling_rate
            else:
                # Revert
                _remove_edge(adj, cn, i, j, n)
                # With some probability, accept anyway (simulated annealing)
                if random.random() < current_temp:
                    _add_edge(adj, cn, i, j, n)
                else:
                    _remove_edge(adj, cn, i, j, n)
        
        # Try remove+add moves (swap an edge for another)
        if time.time() < deadline:
            edge_list = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edge_list.append((i, j))
            random.shuffle(edge_list)
            
            for (ri, rj) in edge_list[:10]:  # Limit swaps per iteration
                if time.time() > deadline:
                    break
                
                # Remove edge (ri, rj)
                _remove_edge(adj, cn, ri, rj, n)
                
                # Try adding some non-edges
                added = False
                for (ai, aj) in non_edges:
                    if time.time() > deadline:
                        break
                    if adj[ai][aj]:
                        continue
                    if cn[ai][aj] != 0:
                        continue
                    
                    _add_edge(adj, cn, ai, aj, n)
                    
                    if _is_not_3_colorable(adj, n):
                        added = True
                        new_edges = _count_edges(adj, n)
                        if new_edges > best_edges:
                            best_edges = new_edges
                            best_adj = [row[:] for row in adj]
                        break
                    else:
                        # Revert add
                        _remove_edge(adj, cn, ai, aj, n)
                
                # Re-add removed edge
                _add_edge(adj, cn, ri, rj, n)
                
                if not added:
                    # If we couldn't add anything, keep the original edge
                    pass
        
        current_temp *= cooling_rate
    
    return _get_edges(best_adj, n)
