import time
import random

def _is_3_colorable(adj, n):
    """Check if graph with adjacency bitmasks is 3-colorable using backtracking with vertex ordering."""
    colors = [-1] * n
    
    # Order vertices by degree (descending) for better pruning
    order = sorted(range(n), key=lambda v: bin(adj[v]).count('1'), reverse=True)
    
    def backtrack(pos):
        if pos == n:
            return True
        
        i = order[pos]
        
        forbidden = 0
        neighbors = adj[i]
        while neighbors:
            lsb = neighbors & -neighbors
            j = lsb.bit_length() - 1
            if colors[j] != -1:
                forbidden |= (1 << colors[j])
            neighbors ^= lsb
        
        available = 7 ^ forbidden
        
        if available == 0:
            return False
        
        while available:
            lsb = available & -available
            c = lsb.bit_length() - 1
            colors[i] = c
            if backtrack(pos + 1):
                return True
            colors[i] = -1
            available ^= lsb
            
        return False
    
    return backtrack(0)

def _count_triangles_fast(adj, n):
    """Count triangles using bitmask operations."""
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v]
                count += bin(common).count('1')
    return count // 3

def _greedy_triangle_free(adj, n, all_edges):
    """Add edges greedily in random order, maintaining triangle-freeness."""
    shuffled = all_edges[:]
    random.shuffle(shuffled)
    edge_count = 0
    for u, v in shuffled:
        if (adj[u] & adj[v]) == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_count += 1
    return edge_count

def _local_search(adj, n, all_edges, deadline, best_edges, best_count):
    """
    Local search to maximize edges while maintaining triangle-freeness
    and ensuring non-3-colorability.
    """
    current_adj = list(adj)
    current_edge_set = set()
    for u in range(n):
        for v in range(u + 1, n):
            if current_adj[u] & (1 << v):
                current_edge_set.add((u, v))
    
    current_triangles = _count_triangles_fast(current_adj, n)
    
    # Precompute all edge pairs for fast access
    no_improve = 0
    step = 0
    temp = 1.0
    
    while time.time() < deadline and temp > 0.005:
        step += 1
        
        # Randomly pick an edge to consider flipping
        u, v = random.choice(all_edges)
        
        if (u, v) in current_edge_set:
            # Try removing edge (u,v)
            common = current_adj[u] & current_adj[v]
            tri_contrib = bin(common).count('1')
            
            # Remove edge
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            current_edge_set.remove((u, v))
            new_tri = current_triangles - tri_contrib
            
            # Accept if it reduces triangles, or with probability temp
            if new_tri < current_triangles or random.random() < temp * 0.3:
                current_triangles = new_tri
                if new_tri < current_triangles:
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Revert
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edge_set.add((u, v))
                no_improve += 1
        else:
            # Try adding edge (u,v)
            common = current_adj[u] & current_adj[v]
            tri_contrib = bin(common).count('1')
            
            if tri_contrib == 0:
                # Safe to add
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edge_set.add((u, v))
                no_improve = 0
            else:
                # Would create triangles - accept with low probability
                if random.random() < temp * 0.02:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edge_set.add((u, v))
                    current_triangles += tri_contrib
                    no_improve += 1
                else:
                    no_improve += 1
        
        # Check for valid solution periodically
        if step % 10 == 0 and current_triangles == 0:
            if not _is_3_colorable(current_adj, n):
                edge_count = len(current_edge_set)
                if edge_count > best_count[0]:
                    edges = [[u, v] for u, v in sorted(current_edge_set)]
                    best_edges[0] = edges
                    best_count[0] = edge_count
        
        # Decay temperature
        if no_improve > 30:
            temp *= 0.97
            no_improve = 0
    
    return best_edges, best_count

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    best_edges = [None]
    best_count = [0]
    
    # Pre-compute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Start with Clebsch graph as baseline (40 edges, triangle-free, 4-chromatic)
    clebsch_adj = [0] * n
    clebsch_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            diff = bin(u ^ v).count('1')
            if diff >= 3:
                clebsch_adj[u] |= (1 << v)
                clebsch_adj[v] |= (1 << u)
                clebsch_edges.append([u, v])
    
    # Verify Clebsch is valid
    if _count_triangles_fast(clebsch_adj, n) == 0 and not _is_3_colorable(clebsch_adj, n):
        best_edges[0] = clebsch_edges
        best_count[0] = len(clebsch_edges)
    
    # Multiple restarts with local search
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        
        # Initialize with random triangle-free graph
        adj = [0] * n
        _greedy_triangle_free(adj, n, all_edges)
        
        # Run local search from this starting point
        best_edges, best_count = _local_search(
            adj, n, all_edges, deadline, best_edges, best_count
        )
        
        # Also try local search from Clebsch graph
        if restarts % 3 == 0:
            clebsch_adj_copy = list(clebsch_adj)
            best_edges, best_count = _local_search(
                clebsch_adj_copy, n, all_edges, deadline, best_edges, best_count
            )
        
        # Early exit if we've found a very good solution
        if best_count[0] >= 42:
            break
    
    if best_edges[0] is None:
        # Fallback: Use the Clebsch graph
        best_edges[0] = clebsch_edges
    
    return best_edges[0]
