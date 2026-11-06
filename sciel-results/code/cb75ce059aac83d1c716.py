import time
import random
from itertools import combinations

def _is_not_3_colorable(adj, n):
    """Check if graph is NOT 3-colorable using bitmask backtracking."""
    # Build neighbor masks
    neigh = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                neigh[i] |= (1 << j)
    
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
            if adj[v][u] and pos_of[u] < idx:
                mask |= (1 << pos_of[u])
        earlier_neigh[v] = mask
    
    # Bitmask for colors: color_masks[c] has bit idx set if order[idx] has color c
    color_masks = [0, 0, 0]
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        # Colors forbidden: those used by earlier neighbors
        forbidden = 0
        em = earlier_neigh[v]
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

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n = 16
    # Start with Grötzsch graph on 0-10, isolated 11-15
    adj = _groetzsch_adj(n)
    
    # Precompute common neighbor masks for triangle checking
    # cn[i][j] = bitmask of vertices adjacent to both i and j
    def compute_cn(adj, n):
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
    
    cn = compute_cn(adj, n)
    
    best_adj = [row[:] for row in adj]
    best_edges = _count_edges(adj, n)
    
    # Local Search
    # We will try to add edges and swap edges to maximize count while maintaining constraints.
    
    for iteration in range(20000):
        if time.time() > deadline:
            break
        
        # Build list of valid non-edges (no triangle)
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j] and cn[i][j] == 0:
                    non_edges.append((i, j))
        
        if not non_edges:
            break
            
        random.shuffle(non_edges)
        
        improved = False
        
        # Phase 1: Try adding edges
        for (i, j) in non_edges:
            if time.time() > deadline:
                break
            
            # Add edge
            adj[i][j] = 1
            adj[j][i] = 1
            
            # Update cn
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
            
            if _is_not_3_colorable(adj, n):
                new_edges = _count_edges(adj, n)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = [row[:] for row in adj]
                improved = True
            else:
                # Revert
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
                
        # Phase 2: Try swaps (remove one edge, add another)
        # This helps if we are stuck in a local optimum where no single edge addition works.
        if not improved and time.time() < deadline:
            edge_list = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edge_list.append((i, j))
            random.shuffle(edge_list)
            
            # Try a few swaps
            for (ri, rj) in edge_list[:10]:
                if time.time() > deadline:
                    break
                
                # Remove edge (ri, rj)
                adj[ri][rj] = 0
                adj[rj][ri] = 0
                for k in range(n):
                    if k != ri and k != rj:
                        if adj[ri][k]:
                            cn[ri][k] &= ~(1 << rj)
                            cn[k][ri] &= ~(1 << rj)
                        if adj[rj][k]:
                            cn[rj][k] &= ~(1 << ri)
                            cn[k][rj] &= ~(1 << ri)
                cn[ri][rj] = 0
                cn[rj][ri] = 0
                
                # Try adding a non-edge
                # Re-calculate non-edges quickly or just pick random non-edges from previous list?
                # Better to generate a few candidates
                candidates = []
                for i in range(n):
                    for j in range(i+1, n):
                        if not adj[i][j] and cn[i][j] == 0:
                            candidates.append((i, j))
                if candidates:
                    random.shuffle(candidates)
                    for (ai, aj) in candidates[:5]:
                        if time.time() > deadline:
                            break
                        # Add edge (ai, aj)
                        adj[ai][aj] = 1
                        adj[aj][ai] = 1
                        for k in range(n):
                            if k != ai and k != aj:
                                if adj[ai][k]:
                                    cn[ai][k] |= (1 << aj)
                                    cn[k][ai] |= (1 << aj)
                                if adj[aj][k]:
                                    cn[aj][k] |= (1 << ai)
                                    cn[k][aj] |= (1 << ai)
                        cn[ai][aj] = 0
                        cn[aj][ai] = 0
                        
                        if _is_not_3_colorable(adj, n):
                            # Check if this swap improved edge count (it's a swap, so count is same)
                            # But it might allow future additions.
                            # For now, we just keep it if it's valid.
                            # Actually, we want to maximize edges. A swap keeps edge count same.
                            # So we should only accept a swap if it leads to a state where we can add more.
                            # However, for simplicity, let's just accept valid swaps to explore the space.
                            improved = True
                            break
                        else:
                            # Revert add
                            adj[ai][aj] = 0
                            adj[aj][ai] = 0
                            for k in range(n):
                                if k != ai and k != aj:
                                    if adj[ai][k]:
                                        cn[ai][k] &= ~(1 << aj)
                                        cn[k][ai] &= ~(1 << aj)
                                    if adj[aj][k]:
                                        cn[aj][k] &= ~(1 << ai)
                                        cn[k][aj] &= ~(1 << ai)
                
                # Re-add removed edge (ri, rj)
                adj[ri][rj] = 1
                adj[rj][ri] = 1
                for k in range(n):
                    if k != ri and k != rj:
                        if adj[ri][k]:
                            cn[ri][k] |= (1 << rj)
                            cn[k][ri] |= (1 << rj)
                        if adj[rj][k]:
                            cn[rj][k] |= (1 << ri)
                            cn[k][rj] |= (1 << ri)
                            
    return _get_edges(best_adj, n)
