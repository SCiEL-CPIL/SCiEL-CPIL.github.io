import time
import random
from itertools import combinations

def construct():
    """
    Find a triangle-free graph on 16 vertices with chromatic number >= 4,
    maximizing the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    best_edges = None
    best_score = -1
    
    def edge_count_from_adj(adj):
        return sum(adj[i][j] for i in range(n) for j in range(i+1, n))
    
    def is_triangle_free_adj(adj):
        """Fast triangle-free check using adjacency lists."""
        for i in range(n):
            neighbors_i = [j for j in range(n) if adj[i][j]]
            for j in neighbors_i:
                if j <= i:
                    continue
                # Check if i and j share any common neighbor
                for k in neighbors_i:
                    if k > j and adj[j][k]:
                        return False
        return True
    
    def is_triangle_free_mask(adj_mask):
        """Triangle-free check using bitmasks."""
        for i in range(n):
            mask_i = adj_mask[i]
            # For each neighbor j of i (j > i), check if j's neighbors overlap with i's neighbors
            temp = mask_i
            while temp:
                bit = temp & (-temp)
                j = bit.bit_length() - 1
                if j > i:
                    if adj_mask[j] & mask_i:
                        return False
                temp ^= bit
        return True
    
    def is_3_colorable(adj):
        """
        Fast backtracking 3-coloring with bitmasks.
        Returns True if 3-colorable, False otherwise.
        """
        adj_mask = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    adj_mask[i] |= (1 << j)
        
        color = [-1] * n
        uncolored = list(range(n))
        
        def can_color():
            if not uncolored:
                return True
            
            # Pick uncolored vertex with fewest available colors (MRV heuristic)
            best_idx = -1
            best_count = 100
            best_avail = 0
            for idx, v in enumerate(uncolored):
                # Compute forbidden colors from colored neighbors
                forbidden = 0
                mask = adj_mask[v]
                for u in range(n):
                    if color[u] != -1 and (mask >> u) & 1:
                        forbidden |= (1 << color[u])
                avail = 7 & ~forbidden  # 7 = 0b111, available colors
                count = bin(avail).count('1')
                if count == 0:
                    return False
                if count < best_count:
                    best_count = count
                    best_idx = idx
                    best_avail = avail
                    if count == 1:
                        break
            
            v = uncolored[best_idx]
            uncolored.pop(best_idx)
            
            # Try each available color
            avail = best_avail
            while avail:
                c_bit = avail & (-avail)  # lowest set bit
                avail ^= c_bit
                c = c_bit.bit_length() - 1
                color[v] = c
                if can_color():
                    return True
                color[v] = -1
            
            uncolored.insert(best_idx, v)
            return False
        
        return can_color()
    
    def try_add_edge(adj, u, v):
        """Try adding edge (u,v) while maintaining triangle-freeness and 4-chromaticity."""
        if adj[u][v]:
            return adj
        
        # Check triangle-freeness using common neighbors
        for k in range(n):
            if k != u and k != v and adj[u][k] and adj[v][k]:
                return adj
        
        # Try adding
        adj[u][v] = 1
        adj[v][u] = 1
        
        # Check 4-chromaticity
        if not is_3_colorable(adj):
            return adj
        else:
            # Revert
            adj[u][v] = 0
            adj[v][u] = 0
            return adj
    
    def get_non_edges(adj):
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    non_edges.append((i, j))
        return non_edges
    
    # Grotzsch graph: 11 vertices, 20 edges, triangle-free, 4-chromatic
    grotzsch_edges = []
    for i in range(5):
        grotzsch_edges.append([i, (i+1)%5])
    grotzsch_edges.append([5, 1])
    grotzsch_edges.append([5, 4])
    grotzsch_edges.append([6, 0])
    grotzsch_edges.append([6, 2])
    grotzsch_edges.append([7, 1])
    grotzsch_edges.append([7, 3])
    grotzsch_edges.append([8, 2])
    grotzsch_edges.append([8, 4])
    grotzsch_edges.append([9, 3])
    grotzsch_edges.append([9, 0])
    for i in range(5):
        grotzsch_edges.append([10, 5+i])
    
    base_adj = [[0]*n for _ in range(n)]
    for u, v in grotzsch_edges:
        base_adj[u][v] = 1
        base_adj[v][u] = 1
    
    # Find maximum independent set in Grotzsch
    def find_max_independent_set(adj, vertices):
        best_set = []
        
        def backtrack(remaining, current_set):
            nonlocal best_set
            if len(current_set) > len(best_set):
                best_set = current_set[:]
            
            if not remaining:
                return
            
            v = remaining[0]
            rest = remaining[1:]
            
            # Option 1: Include v
            new_remaining = [u for u in rest if not adj[v][u]]
            backtrack(new_remaining, current_set + [v])
            
            # Option 2: Exclude v
            backtrack(rest, current_set)
        
        backtrack(list(vertices), [])
        return best_set
    
    grotzsch_vertices = list(range(11))
    max_iset = find_max_independent_set(base_adj, grotzsch_vertices)
    new_vertices = list(range(11, 16))
    
    adj = [row[:] for row in base_adj]
    for nv in new_vertices:
        for iv in max_iset:
            adj[nv][iv] = 1
            adj[iv][nv] = 1
    
    if is_triangle_free_adj(adj) and not is_3_colorable(adj):
        m = edge_count_from_adj(adj)
        if m > best_score:
            best_score = m
            best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
    
    # Local search to add edges
    improved = True
    while improved and time.time() < deadline:
        improved = False
        non_edges = get_non_edges(adj)
        random.shuffle(non_edges)
        for i, j in non_edges:
            if time.time() > deadline:
                break
            if adj[i][j]:
                continue
            new_adj = try_add_edge(adj, i, j)
            if new_adj is not adj:
                adj = new_adj
                improved = True
        
        m = edge_count_from_adj(adj)
        if m > best_score:
            best_score = m
            best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
    
    # K8,8-based strategy: start with complete bipartite, delete edges to make it not 3-colorable
    if time.time() < deadline:
        # K8,8 has 64 edges and is 2-colorable (hence 3-colorable)
        # We need to delete edges to break 3-colorability while keeping triangle-free
        # Strategy: partition into two sets A and B of 8 vertices each
        # Start with all cross edges, then selectively remove some
        
        A = list(range(8))
        B = list(range(8, 16))
        
        k88_adj = [[0]*n for _ in range(n)]
        for a in A:
            for b in B:
                k88_adj[a][b] = 1
                k88_adj[b][a] = 1
        
        # K8,8 is 3-colorable, so we need to modify it
        # One approach: create a structure where the bipartition is "forced"
        # but add constraints that make 3-coloring impossible
        
        # Alternative: use the Grotzsch construction but try to add more edges
        # Try connecting new vertices to more Grotzsch vertices
        
        # Try different independent sets and vertex assignments
        all_grotzsch_isets = []
        
        def find_all_independent_sets(adj, vertices, min_size=4):
            results = []
            
            def backtrack(remaining, current_set):
                if len(current_set) >= min_size:
                    results.append(current_set[:])
                if not remaining:
                    return
                v = remaining[0]
                rest = remaining[1:]
                new_remaining = [u for u in rest if not adj[v][u]]
                backtrack(new_remaining, current_set + [v])
                backtrack(rest, current_set)
            
            backtrack(list(vertices), [])
            return results
        
        # Try multiple independent sets
        isets = find_all_independent_sets(base_adj, grotzsch_vertices, min_size=4)
        
        for iset in isets:
            if time.time() > deadline:
                break
            if len(iset) < 4:
                continue
            
            test_adj = [row[:] for row in base_adj]
            # Connect new vertices to this independent set
            for nv in new_vertices:
                for iv in iset:
                    test_adj[nv][iv] = 1
                    test_adj[iv][nv] = 1
            
            if not is_triangle_free_adj(test_adj):
                continue
            
            if is_3_colorable(test_adj):
                continue
            
            # Local search from this starting point
            improved = True
            while improved and time.time() < deadline:
                improved = False
                non_edges = get_non_edges(test_adj)
                random.shuffle(non_edges)
                for i, j in non_edges:
                    if time.time() > deadline:
                        break
                    if test_adj[i][j]:
                        continue
                    new_adj = try_add_edge(test_adj, i, j)
                    if new_adj is not test_adj:
                        test_adj = new_adj
                        improved = True
                
                m = edge_count_from_adj(test_adj)
                if m > best_score:
                    best_score = m
                    best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if test_adj[i][j]]
    
    # Random search as fallback
    if time.time() < deadline:
        random.seed(42)
        
        def random_triangle_free(density):
            adj = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < density:
                        ok = True
                        for k in range(n):
                            if k != i and k != j:
                                if adj[i][k] and adj[j][k]:
                                    ok = False
                                    break
                        if ok:
                            adj[i][j] = 1
                            adj[j][i] = 1
            return adj
        
        for restart in range(50):
            if time.time() > deadline:
                break
            
            density = 0.3 + 0.4 * (restart % 5) / 4
            adj = random_triangle_free(density)
            
            if not is_triangle_free_adj(adj):
                continue
            
            if is_3_colorable(adj):
                continue
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                non_edges = get_non_edges(adj)
                random.shuffle(non_edges)
                for i, j in non_edges:
                    if time.time() > deadline:
                        break
                    if adj[i][j]:
                        continue
                    new_adj = try_add_edge(adj, i, j)
                    if new_adj is not adj:
                        adj = new_adj
                        improved = True
                m = edge_count_from_adj(adj)
                if m > best_score:
                    best_score = m
                    best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
    
    if best_edges is None:
        best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if base_adj[i][j]]
    
    return best_edges
