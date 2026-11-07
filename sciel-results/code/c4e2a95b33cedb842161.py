import time

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _verify_packing(cycles, all_edges_set):
    """
    Verify that all cycles are valid, edge-disjoint, and use the same vertex set.
    Returns True if valid, False otherwise.
    """
    if not cycles:
        return False
    
    n = P['n']
    used_edges = set()
    target_vertex_set = None
    
    for cycle in cycles:
        if not isinstance(cycle, (list, tuple)):
            return False
        if len(cycle) < 3:
            return False
        
        for v in cycle:
            if not isinstance(v, int) or isinstance(v, bool):
                return False
            if v < 0 or v >= n:
                return False
        
        if len(set(cycle)) != len(cycle):
            return False
        
        current_vset = set(cycle)
        if target_vertex_set is None:
            target_vertex_set = current_vset
        elif current_vset != target_vertex_set:
            return False
        
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i+1) % len(cycle)]
            ek = _edge_key(u, v)
            if ek not in all_edges_set:
                return False
            if ek in used_edges:
                return False
            used_edges.add(ek)
    
    return True

def construct():
    n = P['n']
    all_edges_set = set()
    for u, v in P['edges']:
        all_edges_set.add(_edge_key(u, v))
    
    # The host graph is the circulant C_21(1,2,5) with 70 edges.
    # Upper bound: floor(70/21) = 3 edge-disjoint Hamiltonian cycles.
    # We achieve this bound using three coprime steps: 1, 5, 19 (mod 21).
    # Step 1 -> offsets {1, 20}, Step 5 -> offsets {5, 16}, Step 19 -> offsets {19, 2}.
    # All six offsets are distinct mod 21, so the three cycles are edge-disjoint.
    
    def make_cycle(step):
        return [(i * step) % n for i in range(n)]
    
    cycles = [
        [int(v) for v in make_cycle(1)],
        [int(v) for v in make_cycle(5)],
        [int(v) for v in make_cycle(19)],
    ]
    
    if _verify_packing(cycles, all_edges_set):
        return cycles
    
    # Safety net: return empty if verification unexpectedly fails
    return []

# EVOLVE-BLOCK-END
