import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # 150s limit, leave 5s margin
    
    n = 16
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    pair_index = {}
    for i, (u, v) in enumerate(all_pairs):
        pair_index[(u, v)] = i
    
    # Precompute for each pair (u,v) the list of pairs that would form a triangle
    # If we add edge (u,v), we cannot have any edge (u,w) or (v,w) for w != u,v
    # Actually, if we have edges (u,w) and (v,w), then adding (u,v) creates triangle u-v-w
    # So for each potential edge (u,v), the conflicting edges are (u,w) and (v,w) for all w
    
    # Let's represent the graph as a set of active edges (indices into all_pairs)
    # Triangle-free constraint: for any triple (a,b,c), we can't have all three edges (a,b), (b,c), (c,a)
    
    # For Max-SAT / Local Search approach:
    # Variables: x_e for each edge e (1 = edge present, 0 = edge absent)
    # Hard constraints: triangle-freeness (for each triangle (a,b,c), not (x_ab AND x_bc AND x_ca))
    # Soft objective: maximize sum of x_e
    # Additional constraint: graph must NOT be 3-colorable
    
    # Strategy: Local search over edge subsets
    # State: set of edges (bitmask over 120 possible edges)
    # Move: add or remove an edge
    # Score: number of edges, but only if triangle-free AND not 3-colorable
    
    # For efficiency, let's use a different approach:
    # 1. Start with known good graphs (Clebsch has 40 edges)
    # 2. Try to add edges while maintaining triangle-freeness
    # 3. Check 3-colorability periodically
    
    # Fast 3-colorability check using bitmask backtracking
    def is_3_colorable(adj_bitmasks):
        """
        adj_bitmasks: list of 16 ints, each a 16-bit mask of neighbors
        Returns True if 3-colorable
        """
        # Use ordering by degree for better pruning
        order = sorted(range(n), key=lambda i: bin(adj_bitmasks[i]).count('1'), reverse=True)
        pos_in_order = [0] * n
        for idx, v in enumerate(order):
            pos_in_order[v] = idx
        
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Determine available colors
            used_colors = 0
            for w in range(n):
                if (adj_bitmasks[v] >> w) & 1 and pos_in_order[w] < idx:
                    if colors[w] >= 0:
                        used_colors |= (1 << colors[w])
            
            avail = 7 ^ used_colors  # 0b111 minus used
            if avail == 0:
                return False
            
            # Try colors in order
            while avail:
                c_bit = avail & (-avail)
                c = c_bit.bit_length() - 1
                avail ^= c_bit
                
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
            
            return False
        
        return backtrack(0)
    
    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges
    
    def adj_to_edge_count(adj):
        return sum(bin(x).count('1') for x in adj) // 2
    
    # Clebsch graph construction
    def make_clebsch():
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                diff = u ^ v
                dist = bin(diff).count('1')
                if dist == 1 or dist == 4:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj
    
    # Local search: try to add edges to a maximal triangle-free graph
    def try_add_edges(adj, max_iters=1000):
        """Try to add edges to adj while keeping it triangle-free. Return new adj if improved."""
        adj_copy = adj[:]
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try all non-edges
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj_copy[u] >> v) & 1):
                        # Check if adding (u,v) creates a triangle
                        if adj_copy[u] & adj_copy[v]:
                            continue
                        # Safe to add
                        adj_copy[u] |= (1 << v)
                        adj_copy[v] |= (1 << u)
                        improved = True
        return adj_copy
    
    # Better local search: hill climbing with random restarts
    def local_search(adj_init, time_limit):
        """Local search to maximize edges while maintaining triangle-free and non-3-colorable."""
        best_adj = adj_init[:]
        best_count = adj_to_edge_count(best_adj)
        
        # Verify initial graph
        if is_3_colorable(best_adj):
            return None, 0
        
        current_adj = best_adj[:]
        current_count = best_count
        
        while time.time() < time_limit:
            # Try to add a random non-edge
            non_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((current_adj[u] >> v) & 1) and not (current_adj[u] & current_adj[v]):
                        non_edges.append((u, v))
            
            if non_edges:
                u, v = random.choice(non_edges)
                new_adj = current_adj[:]
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                new_count = adj_to_edge_count(new_adj)
                
                if new_count > current_count:
                    # Check 3-colorability
                    if not is_3_colorable(new_adj):
                        current_adj = new_adj
                        current_count = new_count
                        if current_count > best_count:
                            best_adj = current_adj[:]
                            best_count = current_count
                    else:
                        # 3-colorable, reject
                        pass
                else:
                    # Same or fewer edges, but maybe we can swap
                    pass
            
            # Try edge swaps: remove one edge, add another
            if random.random() < 0.3:
                edges_list = get_edges_from_adj(current_adj)
                if edges_list:
                    rem_u, rem_v = random.choice(edges_list)
                    # Remove edge
                    trial_adj = current_adj[:]
                    trial_adj[rem_u] &= ~(1 << rem_v)
                    trial_adj[rem_v] &= ~(1 << rem_u)
                    
                    # Try to add a different edge
                    non_edges2 = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if not ((trial_adj[u] >> v) & 1) and not (trial_adj[u] & trial_adj[v]):
                                non_edges2.append((u, v))
                    
                    if non_edges2:
                        add_u, add_v = random.choice(non_edges2)
                        trial_adj[add_u] |= (1 << add_v)
                        trial_adj[add_v] |= (1 << add_u)
                        trial_count = adj_to_edge_count(trial_adj)
                        
                        if trial_count >= current_count:
                            if not is_3_colorable(trial_adj):
                                current_adj = trial_adj
                                current_count = trial_count
                                if current_count > best_count:
                                    best_adj = current_adj[:]
                                    best_count = current_count
        
        return best_adj, best_count
    
    # Main search
    best_adj = None
    best_count = 0
    
    # Start with Clebsch
    clebsch = make_clebsch()
    clebsch_count = adj_to_edge_count(clebsch)
    
    if not is_3_colorable(clebsch):
        best_adj = clebsch[:]
        best_count = clebsch_count
    
    # Try to expand Clebsch
    expanded = try_add_edges(clebsch)
    exp_count = adj_to_edge_count(expanded)
    if exp_count > best_count and not is_3_colorable(expanded):
        best_adj = expanded[:]
        best_count = exp_count
    
    # Run local search from best
    time_left = deadline - time.time()
    if time_left > 5:
        result_adj, result_count = local_search(best_adj, time_left - 2)
        if result_count > best_count:
            best_adj = result_adj
            best_count = result_count
    
    # Random restarts: generate random maximal triangle-free graphs
    while time.time() < deadline - 5:
        # Random greedy maximal triangle-free
        adj = [0] * n
        random_pairs = all_pairs[:]
        random.shuffle(random_pairs)
        for u, v in random_pairs:
            if not ((adj[u] >> v) & 1) and not (adj[u] & adj[v]):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        count = adj_to_edge_count(adj)
        if count > best_count:
            if not is_3_colorable(adj):
                best_adj = adj[:]
                best_count = count
                # Try to improve with local search
                time_left = deadline - time.time()
                if time_left > 10:
                    result_adj, result_count = local_search(best_adj, time_left - 5)
                    if result_count > best_count:
                        best_adj = result_adj
                        best_count = result_count
    
    if best_adj is None:
        return get_edges_from_adj(clebsch)
    
    return get_edges_from_adj(best_adj)

if __name__ == "__main__":
    construct()
