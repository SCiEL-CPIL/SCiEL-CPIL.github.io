import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def edge_key(u, v):
        return (min(u, v), max(u, v))
    
    def _find_one_hc(adj_lists, used_edges, n, deadline, start=0, path=None, visited_mask=None):
        """Find a single Hamiltonian cycle using available edges with bitmask visited."""
        if time.time() > deadline:
            return None
        
        if path is None:
            path = [start]
            visited_mask = 1 << start
        
        if len(path) == n:
            # Check if we can close the cycle
            last = path[-1]
            ek = edge_key(last, start)
            if ek not in used_edges and start in adj_lists[last]:
                return list(path)
            return None
        
        last = path[-1]
        
        # Get neighbors not yet visited and with available edges
        candidates = []
        for nb in adj_lists[last]:
            if not (visited_mask & (1 << nb)):
                ek = edge_key(last, nb)
                if ek not in used_edges:
                    candidates.append(nb)
        
        if not candidates:
            return None
        
        # Warnsdorff-like heuristic: prefer vertices with fewer forward options
        remaining_unvisited = 0
        for i in range(n):
            if not (visited_mask & (1 << i)):
                remaining_unvisited |= (1 << i)
        
        def count_available(v):
            cnt = 0
            for x in adj_lists[v]:
                if (remaining_unvisited & (1 << x)) and edge_key(v, x) not in used_edges:
                    cnt += 1
            return cnt
        
        candidates.sort(key=count_available)
        
        for nb in candidates:
            new_mask = visited_mask | (1 << nb)
            result = _find_one_hc(adj_lists, used_edges, n, deadline, start, path + [nb], new_mask)
            if result is not None:
                return result
        
        return None
    
    def _find_hc_with_order(adj_lists, used_edges, n, deadline, vertex_order):
        """Try to find HC starting from each vertex in vertex_order."""
        for start_v in vertex_order:
            if time.time() > deadline:
                return None
            cycle = _find_one_hc(adj_lists, used_edges, n, deadline, start=start_v)
            if cycle is not None:
                return cycle
        return None
    
    # Main search
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
    best_result = []
    
    # Strategy: Try structured orderings first, then random
    random.seed(12345)
    
    # Pre-compute a few good starting orders
    natural_order = list(range(n))
    reverse_order = list(range(n-1, -1, -1))
    
    all_orders = [natural_order, reverse_order]
    
    # Generate more random orders for diversity
    for _ in range(40):
        random_order = list(range(n))
        random.shuffle(random_order)
        all_orders.append(random_order)
    
    # Also try some specific heuristic orders based on degree
    degrees = [len(adj[i]) for i in range(n)]
    degree_order = sorted(range(n), key=lambda x: degrees[x])
    degree_order_rev = sorted(range(n), key=lambda x: -degrees[x])
    all_orders.append(degree_order)
    all_orders.append(degree_order_rev)
    
    for order_idx, vertex_order in enumerate(all_orders):
        if time.time() > deadline:
            break
        
        used_edges = set()
        cycles = []
        
        # Try to find up to 5 Hamiltonian cycles (we have 70 edges, each HC uses 21, so max 3 by edge count,
        # but let's try a bit more in case of different cycle structures)
        # Actually with 70 edges and 21 per cycle, max is 3 edge-disjoint HCs.
        # But let's try to be safe and try for more if possible.
        for cycle_idx in range(5):
            if time.time() > deadline:
                break
            
            # Create shuffled adjacency for this attempt
            shuffled_adj = [list(s) for s in adj]
            # Shuffle with a different seed for each cycle
            for lst in shuffled_adj:
                random.shuffle(lst)
            
            cycle = _find_hc_with_order(shuffled_adj, used_edges, n, deadline, vertex_order)
            if cycle is None:
                break
            
            cycles.append(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                used_edges.add(edge_key(u, v))
        
        if len(cycles) > len(best_result):
            best_result = cycles
            if len(best_result) >= 3:
                # We found the theoretical maximum (70/21 = 3.33, so max 3)
                break
    
    # Ensure types are correct
    result = []
    for cycle in best_result:
        result.append([int(v) for v in cycle])
    
    # If we couldn't find any, return a valid single cycle
    if not result:
        used_edges = set()
        cycle = _find_one_hc([list(s) for s in adj], used_edges, n, time.time() + 5)
        if cycle:
            result = [[int(v) for v in cycle]]
    
    return result
# EVOLVE-BLOCK-END
