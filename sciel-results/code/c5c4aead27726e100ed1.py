import time
import random
import itertools

_N = 20
_COLORS = 20
_ALL_EDGES = [(i, j) for i in range(_N) for j in range(i + 1, _N)]
_EDGE_IDX = {(i, j): idx for idx, (i, j) in enumerate(_ALL_EDGES)}
_N_EDGES = len(_ALL_EDGES)

# Precompute all 4-cycles
# A 4-cycle is defined by 4 distinct vertices a, b, c, d.
# The edges are (a,b), (b,c), (c,d), (d,a).
# There are 3 ways to pair 4 vertices into a cycle:
# 1. a-b-c-d-a
# 2. a-b-d-c-a
# 3. a-c-b-d-a
# We will generate all unique 4-cycles (sets of 4 edge indices).

_CYCLES = []
_seen_cycles = set()

for i in range(_N):
    for j in range(i + 1, _N):
        for k in range(j + 1, _N):
            for l in range(k + 1, _N):
                # Vertices i, j, k, l
                # Cycle 1: i-j-k-l-i
                e1 = _EDGE_IDX[(min(i, j), max(i, j))]
                e2 = _EDGE_IDX[(min(j, k), max(j, k))]
                e3 = _EDGE_IDX[(min(k, l), max(k, l))]
                e4 = _EDGE_IDX[(min(l, i), max(l, i))]
                cycle1 = tuple(sorted([e1, e2, e3, e4]))
                if cycle1 not in _seen_cycles:
                    _seen_cycles.add(cycle1)
                    _CYCLES.append(cycle1)
                
                # Cycle 2: i-j-l-k-i
                e1 = _EDGE_IDX[(min(i, j), max(i, j))]
                e2 = _EDGE_IDX[(min(j, l), max(j, l))]
                e3 = _EDGE_IDX[(min(l, k), max(l, k))]
                e4 = _EDGE_IDX[(min(k, i), max(k, i))]
                cycle2 = tuple(sorted([e1, e2, e3, e4]))
                if cycle2 not in _seen_cycles:
                    _seen_cycles.add(cycle2)
                    _CYCLES.append(cycle2)
                
                # Cycle 3: i-k-j-l-i
                e1 = _EDGE_IDX[(min(i, k), max(i, k))]
                e2 = _EDGE_IDX[(min(k, j), max(k, j))]
                e3 = _EDGE_IDX[(min(j, l), max(j, l))]
                e4 = _EDGE_IDX[(min(l, i), max(l, i))]
                cycle3 = tuple(sorted([e1, e2, e3, e4]))
                if cycle3 not in _seen_cycles:
                    _seen_cycles.add(cycle3)
                    _CYCLES.append(cycle3)

# Map edge index to list of cycles it belongs to
_EDGE_TO_CYCLES = [[] for _ in range(_N_EDGES)]
for c_idx, cycle in enumerate(_CYCLES):
    for e_idx in cycle:
        _EDGE_TO_CYCLES[e_idx].append(c_idx)

def _is_valid_addition(edge_idx, color, present_edges, colors):
    """
    Check if adding edge edge_idx with 'color' is valid.
    present_edges: set of edge indices.
    colors: dict edge_idx -> color.
    """
    for c_idx in _EDGE_TO_CYCLES[edge_idx]:
        cycle = _CYCLES[c_idx]
        # Check if all other 3 edges in the cycle are present
        other_edges = [e for e in cycle if e != edge_idx]
        if all(e in present_edges for e in other_edges):
            c1 = colors[other_edges[0]]
            c2 = colors[other_edges[1]]
            c3 = colors[other_edges[2]]
            if len({c1, c2, c3, color}) < 4:
                return False
    return True

def _count_bad_cycles(present_edges, colors):
    """
    Count the number of bad 4-cycles in the current graph.
    A bad 4-cycle is one where all 4 edges are present but colors are not distinct.
    """
    bad_count = 0
    for cycle in _CYCLES:
        if all(e in present_edges for e in cycle):
            cols = [colors[e] for e in cycle]
            if len(set(cols)) < 4:
                bad_count += 1
    return bad_count

def _find_bad_cycle(present_edges, colors):
    """
    Find one bad 4-cycle. Returns tuple of edge indices or None.
    """
    for cycle in _CYCLES:
        if all(e in present_edges for e in cycle):
            cols = [colors[e] for e in cycle]
            if len(set(cols)) < 4:
                return cycle
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    rng = random.Random(42)
    
    best_edges = []
    best_count = 0
    
    # Run multiple restarts
    # Each restart:
    # 1. Greedy construction to get a valid graph.
    # 2. Local search to improve density (if possible) or maintain validity.
    
    while time.time() < deadline:
        present_edges = set()
        colors = {}
        
        # Random order of edges
        order = list(range(_N_EDGES))
        rng.shuffle(order)
        
        # Phase 1: Greedy Construction
        for e_idx in order:
            if time.time() > deadline:
                break
            
            color_order = list(range(_COLORS))
            rng.shuffle(color_order)
            
            added = False
            for c in color_order:
                if _is_valid_addition(e_idx, c, present_edges, colors):
                    present_edges.add(e_idx)
                    colors[e_idx] = c
                    added = True
                    break
            
        current_count = len(present_edges)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = [[_ALL_EDGES[e][0], _ALL_EDGES[e][1], colors[e]] for e in present_edges]
        
        # Phase 2: Local Search / Refinement
        # Try to remove edges that are in bad cycles? No, greedy ensures validity.
        # So current graph is valid.
        # Can we add more edges? The greedy loop already tried all edges in random order.
        # But maybe a different order or different color choice would allow more?
        # Let's try a local search step:
        # Pick a random missing edge, try to add it.
        
        missing_edges = [e for e in range(_N_EDGES) if e not in present_edges]
        
        # Limit local search steps to avoid timeout
        local_steps = 200
        for _ in range(local_steps):
            if time.time() > deadline:
                break
            if not missing_edges:
                break
            
            e_idx = rng.choice(missing_edges)
            
            color_order = list(range(_COLORS))
            rng.shuffle(color_order)
            
            added = False
            for c in color_order:
                if _is_valid_addition(e_idx, c, present_edges, colors):
                    present_edges.add(e_idx)
                    colors[e_idx] = c
                    missing_edges.remove(e_idx)
                    added = True
                    break
            
            if added:
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_edges = [[_ALL_EDGES[e][0], _ALL_EDGES[e][1], colors[e]] for e in present_edges]
        
        # If we've found a very dense graph, we might stop early?
        # Max possible is 190.
        # If we hit 100+, it's good.
        if best_count >= 100:
             # Continue to try to improve, but be careful with time
             pass

    return best_edges
