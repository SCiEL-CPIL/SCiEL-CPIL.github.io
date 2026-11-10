import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Fast 2-colorability check using bitset-based DPLL with unit propagation.
    Returns True if 2-colorable (has a proper 2-coloring), False otherwise.
    """
    m = len(edges)
    if m == 0:
        return True
    
    # Precompute bitmasks for each edge
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # For each vertex, bitmask of edges containing it
    vertex_edge_mask = [0] * n
    for i, e in enumerate(edges):
        for v in e:
            vertex_edge_mask[v] |= (1 << i)
    
    # color: -1 uncolored, 0, 1
    color = [-1] * n
    # active edges bitmask
    active_edges = (1 << m) - 1
    
    def propagate(vertex, val):
        """
        Assign color `val` to `vertex`. Use unit propagation.
        Returns False if conflict, True otherwise.
        Modifies color and active_edges in-place (via nonlocal).
        """
        nonlocal active_edges
        stack = [(vertex, val)]
        while stack:
            v, c = stack.pop()
            if color[v] != -1:
                if color[v] != c:
                    return False
                else:
                    continue
            
            color[v] = c
            
            # For each edge containing v, check if we can propagate
            # We need to check edges where v is one of the vertices
            v_edges = vertex_edge_mask[v]
            # Only check active edges
            v_edges &= active_edges
            
            while v_edges:
                # Get lowest set bit
                lsb = v_edges & (-v_edges)
                e_idx = lsb.bit_length() - 1
                v_edges ^= lsb
                
                if not (active_edges & lsb):
                    continue
                    
                edge = edges[e_idx]
                counts = [0, 0]
                uncolored = []
                for u in edge:
                    if color[u] == 0:
                        counts[0] += 1
                    elif color[u] == 1:
                        counts[1] += 1
                    else:
                        uncolored.append(u)
                
                # If 4 are same color, 5th must be opposite
                if counts[0] == 4 and len(uncolored) == 1:
                    stack.append((uncolored[0], 1))
                elif counts[1] == 4 and len(uncolored) == 1:
                    stack.append((uncolored[0], 0))
                elif counts[0] == 5 or counts[1] == 5:
                    return False
                # If an edge has both colors, it's satisfied - remove from active
                elif counts[0] > 0 and counts[1] > 0:
                    active_edges &= ~lsb
        
        return True
    
    def find_unassigned():
        """Find the next unassigned vertex."""
        for i in range(n):
            if color[i] == -1:
                return i
        return -1
    
    def solve():
        v = find_unassigned()
        if v == -1:
            return True
        
        # Try both colors
        for c in [0, 1]:
            saved_color = color[:]
            saved_active = active_edges
            
            if propagate(v, c):
                if solve():
                    return True
            
            color[:] = saved_color
            nonlocal active_edges
            active_edges = saved_active
        
        return False
    
    return solve()

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Start with a known construction: complete 5-uniform on 9 vertices (126 edges)
    # Then try to minimize
    best_result = None
    best_count = float('inf')
    
    # Strategy: Start with various initial hypergraphs, then greedily remove edges
    # while maintaining non-2-colorability
    
    # Initial candidates to try
    initial_candidates = []
    
    # 1. Complete 5-uniform on 9 vertices
    initial_candidates.append(('complete_9', 9, [list(e) for e in combinations(range(9), 5)]))
    
    # 2. Random hypergraphs on 10-12 vertices
    import random
    for n in [10, 11, 12]:
        for seed in range(5):
            random.seed(seed * 100 + n)
            all_edges = list(combinations(range(n), 5))
            random.shuffle(all_edges)
            # Take a subset - enough to likely be non-2-colorable
            # For n=10, C(10,5)=252, for n=11, C(11,5)=462, for n=12, C(12,5)=792
            # Take about 30-50% to start
            frac = 0.3
            num_edges = int(len(all_edges) * frac)
            edges = [list(e) for e in all_edges[:num_edges]]
            initial_candidates.append((f'random_{n}_{seed}', n, edges))
    
    # Try each initial candidate
    for name, n, edges in initial_candidates:
        if time.time() > deadline:
            break
        
        # First check if this is already non-2-colorable
        if is_2_colorable(n, edges):
            # Try adding more edges
            all_possible = list(combinations(range(n), 5))
            random.seed(hash(name) % 10000)
            random.shuffle(all_possible)
            edge_set = set(tuple(e) for e in edges)
            for e in all_possible:
                if time.time() > deadline:
                    break
                if e not in edge_set:
                    edges.append(list(e))
                    edge_set.add(e)
                    if not is_2_colorable(n, edges):
                        break
            
            if is_2_colorable(n, edges):
                continue
        
        # Now greedily remove edges
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Try removing each edge (in reverse order for efficiency)
            for i in range(len(edges) - 1, -1, -1):
                if time.time() > deadline:
                    break
                candidate = edges[:i] + edges[i+1:]
                if not is_2_colorable(n, candidate):
                    edges = candidate
                    changed = True
                    break  # Restart loop as indices changed
        
        # Check if we improved
        if len(edges) < best_count:
            best_count = len(edges)
            best_result = {'n': n, 'edges': edges}
    
    # If we didn't find anything good, return the complete 9-vertex hypergraph
    if best_result is None:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    return best_result
