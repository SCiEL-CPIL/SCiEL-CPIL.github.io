import time
import random
import math
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
NUM_K4 = 1820

# Pre-compute edge indices
EDGE_IDX = {}
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        EDGE_IDX[(i, j)] = idx
        idx += 1

# Pre-compute K4s and their edge indices
K4_EDGES = []
for quad in combinations(range(N), 4):
    a, b, c, d = quad
    edges = [
        EDGE_IDX[(a, b)], EDGE_IDX[(a, c)], EDGE_IDX[(a, d)],
        EDGE_IDX[(b, c)], EDGE_IDX[(b, d)], EDGE_IDX[(c, d)]
    ]
    K4_EDGES.append(tuple(edges))

# Pre-compute for each edge, which K4s it belongs to
EDGE_TO_K4 = [[] for _ in range(NUM_EDGES)]
for k_idx, edges in enumerate(K4_EDGES):
    for e in edges:
        EDGE_TO_K4[e].append(k_idx)

def _count_distinct_6(c0, c1, c2, c3, c4, c5):
    """
    Returns the number of distinct colors among 6 values.
    Using set is fast enough for 6 elements in CPython.
    """
    return len({c0, c1, c2, c3, c4, c5})

def _initial_violations(colors):
    """
    Compute initial violation status and count for a given coloring.
    Returns (k4_status, total_v) where k4_status is a list of bools.
    """
    k4_status = [False] * NUM_K4
    total_v = 0
    for k_idx, edges in enumerate(K4_EDGES):
        c0, c1, c2, c3, c4, c5 = (colors[e] for e in edges)
        if _count_distinct_6(c0, c1, c2, c3, c4, c5) < 5:
            k4_status[k_idx] = True
            total_v += 1
    return k4_status, total_v

def _local_search_sa(K, time_limit, seed):
    """
    Run Simulated Annealing local search for a valid coloring using exactly K colors.
    Returns the coloring if valid, else None.
    """
    random.seed(seed)
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Initialization
    # Try a few structured inits and random inits
    best_colors = None
    best_violations = float('inf')
    
    inits = []
    # Structured: (i+j) % K
    inits.append([(i + j) % K for i in range(N) for j in range(i + 1, N)])
    # Structured: (i*2+j) % K
    inits.append([(i * 2 + j) % K for i in range(N) for j in range(i + 1, N)])
    # Random
    for _ in range(5):
        inits.append([random.randint(0, K - 1) for _ in range(NUM_EDGES)])
        
    for init in inits:
        if time.time() > deadline - 1:
            break
        k4_status, total_v = _initial_violations(init)
        if total_v < best_violations:
            best_violations = total_v
            best_colors = init[:]
            best_status = k4_status[:]
            
        if total_v == 0:
            return init
            
    if best_colors is None:
        return None
        
    colors = best_colors
    k4_status = best_status
    total_v = best_violations
    
    if total_v == 0:
        return colors
        
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9999
    # We want to explore more if we are stuck
    # If we find a better solution, we can reset T slightly or keep cooling
    
    # Main Loop
    while time.time() < deadline - 0.1:
        # Pick a random edge
        e_idx = random.randint(0, NUM_EDGES - 1)
        old_c = colors[e_idx]
        
        # Pick a new color different from old
        new_c = random.randint(0, K - 1)
        if new_c == old_c:
            continue
            
        # Calculate delta in violations for affected K4s
        affected_k4s = EDGE_TO_K4[e_idx]
        delta_v = 0
        
        # We need to check if the change improves or worsens each affected K4
        # To do this efficiently, we compute the new distinct count and compare with old status
        # Old status: k4_status[k_idx] is True if distinct < 5
        # New status: distinct_new < 5
        
        # Optimization: We can just compute the new distinct count for each affected K4
        # and update the delta.
        
        for k_idx in affected_k4s:
            edges = K4_EDGES[k_idx]
            e0, e1, e2, e3, e4, e5 = edges
            
            # Get colors with new_c substituted
            c0 = colors[e0] if e0 != e_idx else new_c
            c1 = colors[e1] if e1 != e_idx else new_c
            c2 = colors[e2] if e2 != e_idx else new_c
            c3 = colors[e3] if e3 != e_idx else new_c
            c4 = colors[e4] if e4 != e_idx else new_c
            c5 = colors[e5] if e5 != e_idx else new_c
            
            new_distinct = _count_distinct_6(c0, c1, c2, c3, c4, c5)
            new_status = new_distinct < 5
            
            if k4_status[k_idx] and not new_status:
                delta_v -= 1
            elif not k4_status[k_idx] and new_status:
                delta_v += 1
                
        new_total_v = total_v + delta_v
        
        # Acceptance Criterion
        accept = False
        if delta_v < 0:
            accept = True
        elif delta_v == 0:
            # Accept equal moves with some probability to wander
            accept = random.random() < 0.5
        else:
            # Accept worse moves with probability exp(-delta_v / T)
            # delta_v is positive here
            prob = math.exp(-delta_v / T)
            accept = random.random() < prob
            
        if accept:
            # Apply move
            colors[e_idx] = new_c
            total_v = new_total_v
            
            # Update k4_status for affected K4s
            for k_idx in affected_k4s:
                edges = K4_EDGES[k_idx]
                e0, e1, e2, e3, e4, e5 = edges
                c0 = colors[e0]
                c1 = colors[e1]
                c2 = colors[e2]
                c3 = colors[e3]
                c4 = colors[e4]
                c5 = colors[e5]
                k4_status[k_idx] = _count_distinct_6(c0, c1, c2, c3, c4, c5) < 5
                
            if total_v == 0:
                return colors
                
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
        else:
            # Revert (no change to colors needed as we didn't apply it)
            # But we might want to cool down slowly even on rejection? 
            # Standard SA cools every step.
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115 # 5s margin for 120s limit
    
    # Try K from 13 to 15
    # If we find a valid coloring with K, we return it.
    # Score is seed/K. Lower K is better.
    
    # Time allocation:
    # K=13: 30s
    # K=14: 30s
    # K=15: 30s
    # Remaining: Fallback 17
    
    for K in [13, 14, 15]:
        if time.time() > deadline - 5:
            break
            
        time_left = deadline - time.time()
        # Allocate at most 30s per K, or all remaining if less
        alloc = min(time_left - 2, 30)
        if alloc <= 0:
            continue
            
        # Run local search
        res = _local_search_sa(K, alloc, seed=42 + K)
        if res is not None:
            return res
            
    # Fallback: 17 colors
    # (i+j) % 17 is a known valid solution
    return [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]

# EVOLVE-BLOCK-END
