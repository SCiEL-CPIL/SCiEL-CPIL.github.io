import time
import sys

def _compute_neighbors(n=256):
    """
    Compute neighbor bitsets for the conflict graph.
    Vertex v (1-indexed) corresponds to bit (v-1).
    neighbors[v] is an integer where bit j is set if v and j+1 conflict.
    """
    neighbors = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) == 0:
                # i conflicts with j
                neighbors[i] |= (1 << (j - 1))
                neighbors[j] |= (1 << (i - 1))
    return neighbors

def _greedy_init(neighbors, n=256):
    """
    Find a good initial independent set using a greedy heuristic.
    We try multiple random orderings and pick the best.
    """
    best_set = set()
    best_size = 0
    
    # Strategy 1: Sort by degree ascending (pick least connected first)
    order = sorted(range(1, n + 1), key=lambda x: (neighbors[x].bit_count(), x))
    current = set()
    for v in order:
        # Check if v conflicts with any in current
        # v conflicts with u if bit (u-1) is set in neighbors[v]
        # We need to check if (neighbors[v] & bitmask_of_current) != 0
        # Constructing bitmask of current every time is slow.
        # Instead, maintain a bitmask of current.
        pass
        
    # Better greedy:
    def greedy(order):
        selected_mask = 0
        count = 0
        for v in order:
            v_bit = 1 << (v - 1)
            # Check if v has any neighbor in selected
            if (neighbors[v] & selected_mask) == 0:
                selected_mask |= v_bit
                count += 1
        return count, selected_mask

    import random
    rng = random.Random(42)
    
    # Try degree-sorted
    c, m = greedy(order)
    if c > best_size:
        best_size = c
        best_set = {i+1 for i in range(n) if (m >> i) & 1}
        
    # Try random restarts
    for _ in range(100):
        rand_order = list(range(1, n + 1))
        rng.shuffle(rand_order)
        c, m = greedy(rand_order)
        if c > best_size:
            best_size = c
            best_set = {i+1 for i in range(n) if (m >> i) & 1}
            
    return best_set, best_size

def _color_bound(vertices, neighbors, n=256):
    """
    Greedy coloring of the subgraph induced by 'vertices' (list of 1-indexed vertices).
    Returns the number of colors used, which is an upper bound on the MIS size.
    """
    if not vertices:
        return 0
    
    # Sort vertices by degree in the subgraph (descending) for better coloring
    # Degree of v in subgraph is count of neighbors in vertices
    # To speed this up, we can just use the global degree or a simple heuristic.
    # Tomita's algorithm usually sorts by degree in the *original* graph or dynamic.
    # Let's sort by degree in the subgraph for tighter bound.
    # O(|V|^2) to compute degrees.
    
    degs = {}
    for v in vertices:
        count = 0
        for u in vertices:
            if v != u and (neighbors[v] >> (u-1)) & 1:
                count += 1
        degs[v] = count
        
    sorted_v = sorted(vertices, key=lambda x: degs[x], reverse=True)
    
    colors = {}
    color_idx = 0
    for v in sorted_v:
        # Find the smallest color not used by neighbors of v in the subgraph
        used_colors = set()
        for u in vertices:
            if u != v and (neighbors[v] >> (u-1)) & 1:
                if u in colors:
                    used_colors.add(colors[u])
        
        # Find smallest available color
        c = 0
        while c in used_colors:
            c += 1
        colors[v] = c
        if c > color_idx:
            color_idx = c
            
    return color_idx + 1 # 0-indexed colors, so count is max+1

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    n = 256
    neighbors = _compute_neighbors(n)
    
    # Initial solution
    init_set, best_size = _greedy_init(neighbors, n)
    best_mask = 0
    for v in init_set:
        best_mask |= (1 << (v - 1))
        
    # Branch and Bound
    # State: candidates_mask, current_size
    # We use a list to hold the current path if we need to reconstruct, but we only need the size.
    # However, to return the set, we need to store the best set found.
    
    def dfs(candidates_mask, current_size, current_mask):
        nonlocal best_size, best_mask
        nonlocal deadline
        
        if time.time() > deadline:
            return
            
        # Pruning: Simple bound
        if current_size + candidates_mask.bit_count() <= best_size:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        # Extract vertices for coloring bound
        # This is the expensive part.
        # Optimization: Only compute coloring bound if the simple bound is not enough?
        # Or compute it always? For 256 nodes, it's fine.
        
        # Get list of vertices
        verts = []
        m = candidates_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx + 1)
            m ^= lsb
            
        # Color bound
        # To make it faster, we can skip the sorting by dynamic degree and just use the list order
        # or pre-sorted order. Let's try sorting by global degree first, it's cheaper.
        # Actually, the sorting by subgraph degree is O(V^2). 
        # Let's try a simpler bound first: just the count.
        # If that fails to prune enough, we add coloring.
        
        # Let's use the coloring bound to be safe.
        # But computing subgraph degrees is O(V^2).
        # V is at most 256. 256^2 = 65k ops. 
        # If we have 1000 nodes in the tree, that's 65M ops. Might be slow in Python.
        
        # Alternative: Use a pre-computed order for coloring?
        # Tomita's algorithm sorts by degree at each step.
        
        # Let's try a hybrid:
        # If candidates_mask.bit_count() is small (< 30), use exact DP or simple search?
        # No, let's stick to B&B with a cheaper bound first.
        
        # Cheaper bound: Greedy coloring without sorting by dynamic degree, just by index or global degree.
        # This is less tight but faster.
        
        # Let's implement a fast greedy coloring bound.
        # 1. Get vertices.
        # 2. Sort by global degree (precomputed).
        # 3. Greedy assign colors.
        
        # Precomputed global degrees
        # We didn't precompute them in _compute_neighbors, let's do it here or pass them.
        
        # Actually, let's just use the simple bound `current_size + count <= best_size`.
        # If the graph is sparse, this might work.
        # If not, we need the coloring bound.
        
        # Let's add the coloring bound but optimize it.
        # We can skip the O(V^2) degree calc and just sort by global degree.
        
        # Global degrees
        if not hasattr(dfs, 'global_degs'):
            dfs.global_degs = [neighbors[i].bit_count() for i in range(n+1)]
            
        sorted_verts = sorted(verts, key=lambda x: dfs.global_degs[x], reverse=True)
        
        # Greedy coloring
        colors = {}
        max_color = -1
        for v in sorted_verts:
            used = set()
            # Check neighbors in sorted_verts that are already colored
            # We need to check if u is in verts and colored
            # To make this O(V^2) but with small constant:
            for u in sorted_verts:
                if u == v: continue
                if (neighbors[v] >> (u-1)) & 1:
                    if u in colors:
                        used.add(colors[u])
            
            c = 0
            while c in used:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
                
        bound = current_size + max_color + 1
        
        if bound <= best_size:
            return
            
        # Pick a vertex to branch on.
        # Heuristic: Pick the vertex with the highest degree in the subgraph.
        # We already have `verts`.
        # Let's pick the last one in `sorted_verts` (highest global degree) or the one with highest subgraph degree.
        # Using global degree is cheaper.
        v = sorted_verts[-1]
        
        v_bit = 1 << (v - 1)
        
        # Branch 1: Include v
        # New candidates: remove v and its neighbors
        new_cands_inc = candidates_mask & ~neighbors[v] & ~v_bit
        dfs(new_cands_inc, current_size + 1, current_mask | v_bit)
        
        # Check time again
        if time.time() > deadline:
            return
            
        # Branch 2: Exclude v
        new_cands_exc = candidates_mask & ~v_bit
        dfs(new_cands_exc, current_size, current_mask)

    # Start DFS
    # Initial candidates: all vertices
    initial_mask = (1 << n) - 1
    dfs(initial_mask, 0, 0)
    
    # Reconstruct set from best_mask
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return sorted(result)

# Run a quick test to ensure it doesn't crash
if __name__ == "__main__":
    res = construct()
    print(len(res))
