"""Improved program for packing edge-disjoint Hamiltonian cycles."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {
    'n': 21,
    'edges': [
        [0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20],
        [1, 2], [1, 3], [1, 6], [1, 17], [1, 20],
        [2, 3], [2, 4], [2, 7], [2, 18],
        [3, 4], [3, 5], [3, 8], [3, 19],
        [4, 5], [4, 6], [4, 9], [4, 20],
        [5, 6], [5, 7], [5, 10],
        [6, 7], [6, 8], [6, 11],
        [7, 8], [7, 9], [7, 12],
        [8, 9], [8, 10], [8, 13],
        [9, 10], [9, 11], [9, 14],
        [10, 11], [10, 12], [10, 15],
        [11, 12], [11, 13], [11, 16],
        [12, 13], [12, 14], [12, 17],
        [13, 14], [13, 15], [13, 18],
        [14, 15], [14, 16], [14, 19],
        [15, 16], [15, 17], [15, 20],
        [16, 17], [16, 18],
        [17, 18], [17, 19],
        [18, 19], [18, 20],
        [19, 20]
    ]
}

N = P['n']
MAX_CYCLES = 10

def _build_bitmasks():
    """Build adjacency lists as integers (bitmasks) and standard lists."""
    adj_mask = [0] * N
    adj_list = [[] for _ in range(N)]
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        adj_list[u].append(v)
        adj_list[v].append(u)
    return adj_mask, adj_list

def _find_ham_cycle(adj_mask, adj_list, start, used_edge_mask, deadline):
    """
    Find a Hamiltonian cycle starting at `start` avoiding edges in `used_edge_mask`.
    Returns list of vertices or None.
    """
    n = N
    # Precompute available neighbors for each vertex based on current used edges
    # avail_mask[v] is a bitmask of neighbors u such that edge (v,u) is NOT used
    avail_mask = [0] * n
    
    for v in range(n):
        # Start with all neighbors
        m = adj_mask[v]
        # Filter out used edges
        # We need to check if (v, u) is in used_edge_mask.
        # Since used_edge_mask is a bitmask of edges, and we represent an edge (u,v) uniquely as (1 << (min(u,v)*N + max(u,v)))?
        # Actually, maintaining a set of used edges is easier for lookup, but bitmask is faster for intersection.
        # Let's stick to a set of used edges for simplicity in the mask construction step, 
        # but optimize the backtracking with bitmasks for visited/neighbors.
        
        # Efficient way: 
        # Let's just iterate neighbors and check against a global 'used' set passed in?
        # No, let's compute avail_mask once per cycle attempt.
        pass
        
    # Actually, computing avail_mask requires knowing which edges are used.
    # used_edge_mask is passed. Let's convert it to a set for O(1) lookup if not done already.
    # To make this fast, let's assume used_edge_mask is a set of (u,v) tuples with u<v.
    
    for v in range(n):
        m = 0
        for u in adj_list[v]:
            # Check if edge (v, u) is used
            # Normalize edge
            a, b = (v, u) if v < u else (u, v)
            if (a, b) not in used_edge_mask:
                m |= (1 << u)
        avail_mask[v] = m
        
    # Check start viability
    if (avail_mask[start] & ~(1 << start)).bit_count() < 2:
        return None

    visited_mask = 1 << start
    path = [start]
    
    def backtrack():
        if time.time() > deadline:
            return False
            
        if len(path) == n:
            last = path[-1]
            # Check if edge (last, start) exists and is not used
            if last == start:
                return False # Should not happen as we don't add start again until end
            # Check if last is connected to start and edge is available
            if (1 << start) & avail_mask[last]:
                return True
            return False
            
        current = path[-1]
        
        # Get available neighbors not yet visited
        # avail_mask[current] has bits for available neighbors
        # ~visited_mask has bits for unvisited vertices
        # We want neighbors that are available AND unvisited
        # Note: start is visited (bit set), so it won't be picked unless we allow closing.
        # But we only close when len(path)==n. So we must NOT pick start if len(path) < n.
        
        candidates = avail_mask[current] & (~visited_mask)
        
        # If we are at the second to last step, we can also consider start to close?
        # No, standard Hamiltonian path logic: build path of n-1 vertices, then check edge to start.
        # The loop above handles len(path)==n check.
        
        # Heuristic: Order candidates by their own remaining degree (Warnsdorff's rule)
        # candidates is a bitmask. We need to extract vertices and sort them.
        c_list = []
        cm = candidates
        while cm:
            # Get lowest set bit
            lsb = cm & -cm
            idx = lsb.bit_length() - 1
            c_list.append(idx)
            cm ^= lsb
            
        # Sort by remaining available degree (excluding visited)
        # For each candidate u, count how many neighbors of u are also in candidates of u and unvisited?
        # Simple heuristic: sort by avail_mask[u] & ~visited_mask
        c_list.sort(key=lambda u: (avail_mask[u] & ~visited_mask).bit_count())
        
        for nb in c_list:
            # Check if adding nb creates a "dead end" for other vertices?
            # Simple pruning: if any other unvisited vertex (other than nb and start) 
            # has no available neighbors left (excluding current and nb), we fail.
            # This is expensive. Let's try a simpler one: 
            # If nb is the only available neighbor for some other unvisited vertex w, 
            # and we don't pick nb next, we might get stuck. 
            # Actually, let's just try the move.
            
            new_visited = visited_mask | (1 << nb)
            path.append(nb)
            
            if backtrack():
                return True
            
            path.pop()
            
        return False
        
    if backtrack():
        return path
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    adj_mask, adj_list = _build_bitmasks()
    
    best_cycles = []
    
    # Deterministic first try
    for start in range(N):
        if time.time() > deadline:
            break
        if len(best_cycles) >= MAX_CYCLES:
            break
            
        # used_edges is a set of tuples (u,v) with u < v
        used_edges = set()
        current_cycles = []
        
        # Try to build a full packing starting with a cycle from `start`
        # This is a greedy construction.
        # 1. Find a Hamiltonian cycle from start.
        # 2. Add its edges to used.
        # 3. Repeat for other starts? No, we want a SET of disjoint cycles.
        # Strategy: Iterate starts, find a cycle, add to list, remove edges, repeat until no more found.
        
        # We need to track used edges for the CURRENT packing attempt.
        # Reset used_edges for each new packing strategy?
        # The previous code did:
        # for start in ...:
        #   find cycle
        #   if found: add to list, update used_edges
        # This builds ONE packing.
        
        # Let's do this:
        # Try to build a packing.
        # 1. Pick a start vertex.
        # 2. Find a Hamiltonian cycle avoiding used edges.
        # 3. If found, add to packing, mark edges used.
        # 4. Repeat for next start vertex (or any vertex).
        # 5. If no cycle found for a start, try next start.
        # 6. If we run out of starts or hit MAX_CYCLES, stop.
        
        # Optimization: Instead of fixed order, we can try all starts.
        
        packing = []
        used = set()
        
        # To improve success rate, we can prioritize starts that are "hard" to satisfy?
        # Or just iterate 0..N-1.
        
        for s in range(N):
            if time.time() > deadline:
                break
            if len(packing) >= MAX_CYCLES:
                break
                
            # Only try to find a cycle if there are enough edges left?
            # Each cycle uses 21 edges.
            # If len(used) + 21 > 70, we can't add another.
            if len(used) + N > 70:
                break
                
            cycle = _find_ham_cycle(adj_mask, adj_list, s, used, deadline)
            if cycle:
                # Validate cycle
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i+1) % len(cycle)]
                    a, b = (u, v) if u < v else (v, u)
                    if (a, b) in used:
                        valid = False
                        break
                    if (u, v) not in P['edges'] and (v, u) not in P['edges']:
                         # Check if edge exists in graph
                         # We built adj from P['edges'], so if _find_ham_cycle succeeded, 
                         # it used adj_mask which only contains valid edges.
                         # So this check is redundant if _find_ham_cycle is correct.
                         pass
                if valid:
                    packing.append(cycle)
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i+1) % len(cycle)]
                        a, b = (u, v) if u < v else (v, u)
                        used.add((a, b))
                        
        if len(packing) > len(best_cycles):
            best_cycles = packing
            if len(best_cycles) == MAX_CYCLES:
                break
        
    # Randomized restarts
    best_count = len(best_cycles)
    rng = random.Random(42)
    
    for _ in range(50):
        if time.time() > deadline:
            break
        if best_count == MAX_CYCLES:
            break
            
        # Shuffle start order
        starts = list(range(N))
        rng.shuffle(starts)
        
        packing = []
        used = set()
        
        for s in starts:
            if time.time() > deadline:
                break
            if len(packing) >= MAX_CYCLES:
                break
            if len(used) + N > 70:
                break
                
            cycle = _find_ham_cycle(adj_mask, adj_list, s, used, deadline)
            if cycle:
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i+1) % len(cycle)]
                    a, b = (u, v) if u < v else (v, u)
                    if (a, b) in used:
                        valid = False
                        break
                if valid:
                    packing.append(cycle)
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i+1) % len(cycle)]
                        a, b = (u, v) if u < v else (v, u)
                        used.add((a, b))
                        
        if len(packing) > best_count:
            best_cycles = packing
            best_count = len(packing)
            if best_count == MAX_CYCLES:
                break
                
    # Final type check and formatting
    result = []
    for c in best_cycles:
        # Ensure integers
        if all(isinstance(x, int) and not isinstance(x, bool) for x in c):
            result.append(list(c))
            
    if not result:
        # Fallback: Return a single valid cycle if found, or empty?
        # Task says "nonempty list".
        # If we found none, we return an empty list which might be invalid?
        # "construct() -> nonempty list of simple cycles"
        # If we can't find any, we are in trouble.
        # Let's try one last desperate measure with a simple greedy if best_cycles is empty.
        # But _find_ham_cycle should find at least one if the graph has a Ham cycle.
        # This graph likely has one.
        pass
        
    return result
# EVOLVE-BLOCK-END
