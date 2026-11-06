"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# EVOLVE-BLOCK-START

def _edge_index(i, j):
    """Map (i,j) with 0<=i<j<N to linear index in lexicographic order."""
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_edge_list():
    """Return list of (i,j) in lexicographic order."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

def _precompute_cycles():
    """
    Precompute for each edge index, the list of 4-cycles (as tuples of 4 edge indices)
    that contain that edge.
    """
    edges = _build_edge_list()
    edge_map = {e: idx for idx, e in enumerate(edges)}
    
    cycles_for_edge = [[] for _ in range(NUM_EDGES)]
    
    # Iterate over all 4-vertex subsets
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Vertices a, b, c, d
                    # 3 cycles:
                    # 1: (a,b), (b,c), (c,d), (d,a)
                    # 2: (a,b), (b,d), (d,c), (c,a)
                    # 3: (a,c), (c,b), (b,d), (d,a)
                    
                    # Get edge indices
                    # Note: edge_map expects (min, max)
                    def get_ei(u, v):
                        if u > v: u, v = v, u
                        return edge_map[(u, v)]
                    
                    # Cycle 1
                    e1 = get_ei(a, b)
                    e2 = get_ei(b, c)
                    e3 = get_ei(c, d)
                    e4 = get_ei(d, a)
                    cycle1 = (e1, e2, e3, e4)
                    
                    # Cycle 2
                    e5 = get_ei(a, b) # same as e1
                    e6 = get_ei(b, d)
                    e7 = get_ei(d, c) # same as e3
                    e8 = get_ei(c, a)
                    cycle2 = (e5, e6, e7, e8)
                    
                    # Cycle 3
                    e9 = get_ei(a, c)
                    e10 = get_ei(c, b) # same as e2
                    e11 = get_ei(b, d) # same as e6
                    e12 = get_ei(d, a) # same as e4
                    cycle3 = (e9, e10, e11, e12)
                    
                    # Add to lists for each edge in the cycle
                    # We only need to add the cycle once for each edge involved.
                    # To avoid duplicates, we can just append.
                    # Since we are iterating over unique 4-sets, and unique cycles,
                    # each (edge, cycle) pair will be generated exactly once?
                    # No, a cycle is a set of 4 edges.
                    # We want cycles_for_edge[e] to contain the cycle tuple.
                    # We should add the cycle tuple to the list of each of its 4 edges.
                    
                    for e in cycle1:
                        cycles_for_edge[e].append(cycle1)
                    for e in cycle2:
                        cycles_for_edge[e].append(cycle2)
                    for e in cycle3:
                        cycles_for_edge[e].append(cycle3)
                        
    return cycles_for_edge

# Global variable for precomputed cycles
_CYCLES_FOR_EDGE = None

def _init_cycles():
    global _CYCLES_FOR_EDGE
    if _CYCLES_FOR_EDGE is None:
        _CYCLES_FOR_EDGE = _precompute_cycles()

def _check_validity_after_change(colors, edge_idx, new_color):
    """
    Check if changing colors[edge_idx] to new_color results in a valid coloring,
    assuming the previous coloring was valid.
    """
    if _CYCLES_FOR_EDGE is None:
        _init_cycles()
        
    for cycle in _CYCLES_FOR_EDGE[edge_idx]:
        # Get colors of the other 3 edges
        # cycle is a tuple of 4 edge indices
        # Find the position of edge_idx in the cycle
        # Since we just need to check if the set of 4 colors has size 4,
        # we can construct the set.
        
        c0 = new_color if cycle[0] == edge_idx else colors[cycle[0]]
        c1 = new_color if cycle[1] == edge_idx else colors[cycle[1]]
        c2 = new_color if cycle[2] == edge_idx else colors[cycle[2]]
        c3 = new_color if cycle[3] == edge_idx else colors[cycle[3]]
        
        if c0 == c1 == c2 == c3:
            continue
        if c0 == c1 or c0 == c2 or c0 == c3:
            # If c0 matches any other, we need to check if the other 3 are distinct?
            # No, if c0==c1, the set size is at most 3 (c0, c2, c3).
            # So it's valid.
            continue
        if c1 == c2 or c1 == c3:
            continue
        if c2 == c3:
            continue
        
        # If we reach here, c0, c1, c2, c3 are all distinct
        return False
        
    return True

def _greedy_maximize_colors(seed):
    """
    Greedy approach with fast validity check.
    """
    _init_cycles()
    random.seed(seed)
    colors = [0] * NUM_EDGES
    next_color = 1
    
    edge_order = list(range(NUM_EDGES))
    random.shuffle(edge_order)
    
    # Pass 1: Try to assign new colors
    for edge_idx in edge_order:
        if _check_validity_after_change(colors, edge_idx, next_color):
            colors[edge_idx] = next_color
            next_color += 1
            
    # Pass 2: Try again in reverse order? Or another random order?
    # Let's do a second pass with a different order to catch missed opportunities
    random.shuffle(edge_order)
    for edge_idx in edge_order:
        if _check_validity_after_change(colors, edge_idx, next_color):
            colors[edge_idx] = next_color
            next_color += 1
            
    return colors, next_color

def construct():
    """
    Construct a coloring of the 120 edges of K_16 with colors in 0..119
    such that no 4-cycle has 4 distinct edge colors.
    Maximize the number of distinct colors used.
    """
    start_time = time.time()
    
    _init_cycles()
    
    best_colors = None
    best_count = 0
    
    # Run multiple seeds
    for seed in range(200):
        if time.time() - start_time > 50:
            break
            
        colors, count = _greedy_maximize_colors(seed)
        
        if count > best_count:
            best_count = count
            best_colors = colors[:]
            
    # If we have time, try to improve the best solution with local search
    if best_colors is not None and time.time() - start_time < 55:
        # Local search: try to change an edge to a new color
        # This is essentially re-running the greedy step on the existing coloring
        # but we can be more aggressive.
        
        improved_colors = best_colors[:]
        next_color = best_count
        
        # Try each edge
        for _ in range(10): # Multiple rounds
            if time.time() - start_time > 55:
                break
            random.seed(int(time.time() * 1000) % 10000)
            edge_order = list(range(NUM_EDGES))
            random.shuffle(edge_order)
            
            for edge_idx in edge_order:
                if _check_validity_after_change(improved_colors, edge_idx, next_color):
                    improved_colors[edge_idx] = next_color
                    next_color += 1
        
        if next_color > best_count:
            best_colors = improved_colors
            best_count = next_color

    if best_colors is None:
        best_colors = [0] * NUM_EDGES
        best_count = 1
        
    return best_colors

# EVOLVE-BLOCK-END
