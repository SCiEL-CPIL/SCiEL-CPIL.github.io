import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def _edge_index(i, j):
    idx = 0
    for k in range(i):
        idx += N - 1 - k
    idx += j - i - 1
    return idx

def _precompute():
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            edges.append((i, j))
    
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        e1 = (_edge_index(a,b), _edge_index(b,c), _edge_index(c,d), _edge_index(d,a))
        e2 = (_edge_index(a,b), _edge_index(b,d), _edge_index(c,d), _edge_index(a,c))
        e3 = (_edge_index(a,c), _edge_index(b,c), _edge_index(b,d), _edge_index(a,d))
        cycles.append(e1)
        cycles.append(e2)
        cycles.append(e3)
    
    num_cycles = len(cycles)
    edge_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for e in cycle:
            edge_cycles[e].append(ci)
    
    return edges, cycles, num_cycles, edge_cycles

_EDGES, _CYCLES, _NUM_CYCLES, _EDGE_CYCLES = _precompute()

def _is_valid(color_assignment, new_color, edge_idx):
    """Check if assigning new_color to edge_idx is valid given current partial assignment.
    
    Valid means: for every 4-cycle containing edge_idx, we don't create a situation
    where all 4 edges would have distinct colors.
    
    Since some edges may be unassigned, we check: for each cycle containing edge_idx,
    if the other 3 assigned edges have 3 distinct colors all different from new_color,
    then this assignment would be invalid (the 4th edge could be forced to create 4 distinct).
    
    Actually, we need to be more careful. The constraint is that NO 4-cycle has 4 distinct colors.
    So when we assign new_color to edge_idx, for each cycle containing edge_idx:
    - If the other 3 edges are all assigned and have 3 distinct colors, none equal to new_color,
      then this cycle would have 4 distinct colors -> invalid.
    - If any of the other 3 edges is unassigned, we can't yet determine, so we allow it.
    """
    for ci in _EDGE_CYCLES[edge_idx]:
        cycle = _CYCLES[ci]
        other_colors = set()
        all_assigned = True
        for e in cycle:
            if e == edge_idx:
                continue
            c = color_assignment[e]
            if c == -1:
                all_assigned = False
                break
            if c == new_color:
                # Same color, so not 4 distinct
                all_assigned = False  # Actually, we can break early
                break
            other_colors.add(c)
        if all_assigned and len(other_colors) == 3:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    best_result = None
    best_count = 0
    
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
            
        random.seed(attempt * 7919 + 3)
        
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        color_assignment = [-1] * NUM_EDGES
        used_colors = set()
        
        success = True
        for e in edge_order:
            if time.time() > deadline - 5:
                success = False
                break
            
            new_color = len(used_colors)
            
            if _is_valid(color_assignment, new_color, e):
                color_assignment[e] = new_color
                used_colors.add(new_color)
            else:
                assigned = False
                for c in sorted(used_colors):
                    if _is_valid(color_assignment, c, e):
                        color_assignment[e] = c
                        assigned = True
                        break
                if not assigned:
                    success = False
                    break
        
        if not success:
            continue
        
        # Verify no unassigned edges
        if -1 in color_assignment:
            continue
        
        count = len(used_colors)
        if count > best_count:
            best_count = count
            best_result = color_assignment[:]
        
        if best_count >= 30:
            break
    
    if best_result is None:
        return [0] * NUM_EDGES
    
    return best_result
