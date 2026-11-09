"""Dense Q3-free graphs on 16 vertices: Optimized local search."""
import time
import random
import itertools
import sys

# Deadline guard
_DEADLINE = 115.0  # seconds wall-clock limit (safety margin)
_START = None

# Pre-computed data
_SUBSETS_8 = []
_SUBSET_EDGE_INDICES = []
_Q3_LOCAL_MASKS = []
_EDGE_TO_GLOBAL_INDEX = {}
_ALL_EDGES = []

def _init_precompute():
    """Pre-compute all 8-subsets and Q3 masks for fast checking."""
    global _SUBSETS_8, _SUBSET_EDGE_INDICES, _Q3_LOCAL_MASKS, _EDGE_TO_GLOBAL_INDEX, _ALL_EDGES
    
    n = 16
    # 1. Generate all edges and map to indices
    _ALL_EDGES = [(i, j) for i in range(n) for j in range(i+1, n)]
    for idx, (u, v) in enumerate(_ALL_EDGES):
        _EDGE_TO_GLOBAL_INDEX[(u, v)] = idx
        _EDGE_TO_GLOBAL_INDEX[(v, u)] = idx
        
    # 2. Generate all 8-subsets
    _SUBSETS_8 = list(itertools.combinations(range(n), 8))
    
    # 3. For each subset, pre-compute local edge indices and Q3 masks
    # Q3 vertices: 0..7. Edges between i,j if Hamming dist 1.
    # Bipartition: even parity in A, odd in B.
    # A = {0, 3, 5, 6}, B = {1, 2, 4, 7} (depending on labeling)
    # Let's use standard Q3:
    # 0:000, 1:001, 2:010, 3:011, 4:100, 5:101, 6:110, 7:111
    # Edges: (0,1),(0,2),(0,4), (1,3),(1,5), (2,3),(2,6), (3,7), (4,5),(4,6), (5,7), (6,7)
    q3_edges_local = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    
    # Generate all 840 embeddings of Q3 into 8 labeled vertices (0..7)
    # We need to find all sets of 12 edges on 8 vertices that form Q3.
    # We can do this by permuting the vertices of the standard Q3.
    # There are 8! = 40320 permutations. Aut(Q3) has size 48.
    # So 40320 / 48 = 840 unique edge sets.
    
    unique_q3_edge_sets = set()
    base_q3 = set(q3_edges_local)
    
    for perm in itertools.permutations(range(8)):
        # Map original vertices to new vertices
        new_edges = set()
        for u, v in q3_edges_local:
            new_u = perm[u]
            new_v = perm[v]
            if new_u > new_v:
                new_u, new_v = new_v, new_u
            new_edges.add((new_u, new_v))
        unique_q3_edge_sets.add(frozenset(new_edges))
        
    # Convert to local masks (28 bits)
    # Local edge index: (i, j) with i<j. 
    # Index = i*8 - i*(i+1)/2 + (j-i-1) ? 
    # Simpler: create a mapping
    local_edge_to_bit = {}
    bit_idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_to_bit[(i, j)] = bit_idx
            bit_idx += 1
            
    q3_masks_list = []
    for edge_set in unique_q3_edge_sets:
        mask = 0
        for u, v in edge_set:
            mask |= (1 << local_edge_to_bit[(u, v)])
        q3_masks_list.append(mask)
        
    # Now for each 8-subset of the 16 vertices, we need to map global edges to local bits
    for vertices in _SUBSETS_8:
        v_to_local = {v: i for i, v in enumerate(vertices)}
        
        # Map the 28 local edges to global edge indices
        subset_global_indices = []
        local_to_global = {}
        for i in range(8):
            for j in range(i+1, 8):
                gu = vertices[i]
                gv = vertices[j]
                if gu > gv:
                    gu, gv = gv, gu
                g_idx = _EDGE_TO_GLOBAL_INDEX[(gu, gv)]
                subset_global_indices.append(g_idx)
                local_to_global[(i, j)] = g_idx
                
        _SUBSET_EDGE_INDICES.append(subset_global_indices)
        
        # The Q3 masks are the same for all subsets, just the mapping of local edges to global edges changes.
        # Wait, the Q3 masks are defined on local indices 0..7.
        # The local_to_global map tells us which global edge corresponds to local edge (i,j).
        # So for a given global edge set, we extract the 28 bits for this subset.
        # Then we check if any Q3 mask is a subset of these 28 bits.
        
        _Q3_LOCAL_MASKS.append(q3_masks_list)

def _elapsed():
    if _START is None:
        return 0.0
    return time.time() - _START

def _time_left():
    return _DEADLINE - _elapsed()

def _has_q3(edges_set):
    """Fast Q3 check using pre-computed masks."""
    if not edges_set:
        return False
        
    # Convert edges_set to a boolean list of global edge presence
    # edges_set contains frozensets or tuples. We assume frozensets of (u,v) with u<v.
    global_edge_present = [False] * 120
    for e in edges_set:
        u, v = min(e), max(e)
        idx = _EDGE_TO_GLOBAL_INDEX[(u, v)]
        global_edge_present[idx] = True
        
    # Check each 8-subset
    for i in range(len(_SUBSETS_8)):
        global_indices = _SUBSET_EDGE_INDICES[i]
        
        # Construct local 28-bit mask
        local_mask = 0
        for bit_pos, g_idx in enumerate(global_indices):
            if global_edge_present[g_idx]:
                local_mask |= (1 << bit_pos)
                
        # Quick check: if fewer than 12 edges, skip
        if bin(local_mask).count('1') < 12:
            continue
            
        # Check against all 840 Q3 masks
        for q3_mask in _Q3_LOCAL_MASKS[i]:
            if (local_mask & q3_mask) == q3_mask:
                return True
    return False

def construct():
    global _START
    _START = time.time()
    
    # Initialize pre-computation
    _init_precompute()
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Start with a valid Q3-free graph
    # K_{3,13} is Q3-free? 
    # Q3 is 3-regular. K_{3,13} has vertices of degree 13 and 3.
    # Any Q3 subgraph would need 8 vertices.
    # If we pick 3 from side A and 5 from side B, the subgraph is bipartite.
    # Max degree in subgraph is 3 (from A side).
    # Can it be 3-regular?
    # Vertices in A have degree 3 in K_{3,13}. In the subgraph, they connect to at most 3 vertices in B.
    # Vertices in B have degree 3 in the subgraph.
    # If we pick 4 from A and 4 from B? No, A has size 3.
    # So we can't have 4 vertices from A.
    # Thus, no Q3 in K_{3,13}.
    
    initial_edges = set()
    for a in range(3):
        for b in range(3, 16):
            initial_edges.add(frozenset((a, b)))
            
    best_edges = set(initial_edges)
    best_count = len(best_edges)
    
    # Greedy addition
    remaining = [e for e in all_edges if frozenset(e) not in best_edges]
    random.shuffle(remaining)
    
    for e in remaining:
        if _time_left() < 30:
            break
        test_edges = best_edges | {frozenset(e)}
        if _has_q3(test_edges):
            continue
        best_edges = test_edges
        best_count += 1
        
    # Simulated Annealing / Local Search
    current_edges = set(best_edges)
    current_count = len(current_edges)
    
    in_graph = set(current_edges)
    not_in_graph = [e for e in all_edges if frozenset(e) not in in_graph]
    
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    
    while _time_left() > 10:
        if T < T_min:
            T = T_min
            
        if len(not_in_graph) > 0 and random.random() < 0.7:
            idx = random.randint(0, len(not_in_graph)-1)
            e = not_in_graph[idx]
            test_edges = in_graph | {frozenset(e)}
            if _has_q3(test_edges):
                continue
            in_graph = test_edges
            not_in_graph.pop(idx)
            current_count += 1
            if current_count > best_count:
                best_edges = set(in_graph)
                best_count = current_count
        else:
            if not in_graph:
                continue
            e = random.choice(list(in_graph))
            in_graph.discard(e)
            not_in_graph.append(e)
            current_count -= 1
            
        T *= cooling
        
    # Final Greedy
    if _time_left() > 5:
        remaining2 = [e for e in all_edges if frozenset(e) not in best_edges]
        random.shuffle(remaining2)
        for e in remaining2:
            if _time_left() < 2:
                break
            test_edges = best_edges | {frozenset(e)}
            if not _has_q3(test_edges):
                best_edges = test_edges
                
    result = [[int(min(e)), int(max(e))] for e in best_edges]
    return result

if __name__ == "__main__":
    _START = time.time()
    result = construct()
    print(f"Edges: {len(result)}")
    es = set(frozenset((e[0], e[1])) for e in result)
    print(f"Q3-free: {not _has_q3(es)}")
