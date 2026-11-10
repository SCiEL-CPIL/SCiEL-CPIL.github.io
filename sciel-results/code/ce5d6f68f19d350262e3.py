"""
Strategy:
1. Vectorized Rainbow Check: Use NumPy to check all 4-cycles simultaneously.
2. Local Search with Forbidden-Set Pruning:
   - Precompute for each edge which cycles it belongs to.
   - For each cycle, maintain the count of each color on its 4 edges.
   - When considering recoloring an edge, compute the set of "forbidden" colors
     that would create a rainbow cycle.
   - Only try colors NOT in the forbidden set.
   - Greedy: always accept valid moves that increase distinct colors.
   - Random restarts with random initial colorings.
3. The 60s deadline is strictly enforced.
"""

import time
import numpy as np
import random
from itertools import combinations

N = 16
NEDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

def _get_edge_indices():
    """Return edge index mapping and cycle structures."""
    idx_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx_map[(i, j)] = idx
            idx_map[(j, i)] = idx
            idx += 1
    
    # Build edge list for reference
    edge_list = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_list.append((i, j))
            idx += 1
    
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cycles.append((idx_map[(a,b)], idx_map[(b,c)], idx_map[(c,d)], idx_map[(d,a)]))
        # Cycle 2: a-b-d-c-a
        cycles.append((idx_map[(a,b)], idx_map[(b,d)], idx_map[(c,d)], idx_map[(a,c)]))
        # Cycle 3: a-c-b-d-a
        cycles.append((idx_map[(a,c)], idx_map[(b,c)], idx_map[(b,d)], idx_map[(a,d)]))
    
    cycles_np = np.array(cycles, dtype=np.int32)
    
    # For each edge, find which cycles contain it
    edge_to_cycles = [[] for _ in range(NEDGES)]
    for ci, cycle in enumerate(cycles):
        for e in cycle:
            edge_to_cycles[e].append(ci)
    
    return idx_map, cycles_np, edge_to_cycles

def _is_valid_vectorized(colors_np):
    """Check if no cycle has 4 distinct colors."""
    cycle_colors = colors_np[CYCLES]
    sorted_colors = np.sort(cycle_colors, axis=1)
    adjacent_equal = (sorted_colors[:, 0] == sorted_colors[:, 1]) | \
                     (sorted_colors[:, 1] == sorted_colors[:, 2]) | \
                     (sorted_colors[:, 2] == sorted_colors[:, 3])
    return bool(np.all(adjacent_equal))

def _count_distinct(colors_np):
    return len(np.unique(colors_np))

# Precompute
IDX_MAP, CYCLES, EDGE_TO_CYCLES = _get_edge_indices()

def construct():
    start_time = time.time()
    deadline = start_time + 52.0  # Safety margin
    
    best_distinct = 0
    best_colors = None
    
    # Multiple restarts
    while time.time() < deadline:
        # Initialize with a valid 2-coloring: split vertices into two sets of 8
        colors = np.zeros(NEDGES, dtype=np.int32)
        for i in range(N):
            for j in range(i + 1, N):
                idx = IDX_MAP[(i, j)]
                if i >= 8 and j >= 8:
                    colors[idx] = 1
                else:
                    colors[idx] = 0
        
        current_distinct = _count_distinct(colors)
        if best_colors is None or current_distinct > best_distinct:
            best_distinct = current_distinct
            best_colors = colors.copy()
        
        # Local search with forbidden-set pruning
        # Maintain color counts per cycle for fast forbidden computation
        # cycle_color_count[cycle_idx][color] = count of that color on the 4 edges
        
        # Precompute cycle edge positions: for each cycle and each edge in it, what position (0-3)
        # cycle_edge_pos[cycle_idx][edge_idx] = position or -1
        cycle_edge_pos = {}
        for ci, cycle in enumerate(CYCLES):
            pos = {}
            for pi, e in enumerate(cycle):
                pos[e] = pi
            cycle_edge_pos[ci] = pos
        
        # For each edge, precompute the 4 cycle-edge-position pairs
        edge_cycle_info = []
        for e in range(NEDGES):
            info = []
            for ci in EDGE_TO_CYCLES[e]:
                pos = cycle_edge_pos[ci][e]
                info.append((ci, pos))
            edge_cycle_info.append(info)
        
        # Local search
        local_start = time.time()
        max_local_time = min(10.0, max(0.5, deadline - local_start))
        
        improved = True
        no_improve_count = 0
        
        while improved and (time.time() - local_start) < max_local_time:
            improved = False
            no_improve_count += 1
            
            # Shuffle edge order for diversity
            edge_order = list(range(NEDGES))
            random.shuffle(edge_order)
            
            for e in edge_order:
                if time.time() >= deadline:
                    return best_colors.tolist()
                
                old_color = colors[e]
                
                # Compute forbidden colors
                forbidden = set()
                for ci, pos in edge_cycle_info[e]:
                    cycle = CYCLES[ci]
                    other_edges = [cycle[k] for k in range(4) if k != pos]
                    other_colors = [colors[oe] for oe in other_edges]
                    
                    # If we set this edge to color c, the cycle is rainbow iff
                    # c is not in other_colors AND len(set(other_colors + [c])) == 4
                    # i.e., c not in set(other_colors) AND len(set(other_colors)) == 3
                    # Wait: if other_colors has 3 distinct, and c is different from all, then 4 distinct.
                    # If other_colors has < 3 distinct, then even with a new c, max distinct is 4 but
                    # we need exactly 4 distinct total. If other has 2 distinct, adding c gives at most 3.
                    # If other has 1 distinct, adding c gives at most 2.
                    # If other has 3 distinct, adding a 4th distinct c gives 4 distinct = rainbow.
                    # If other has 4 distinct... impossible, only 3 other edges.
                    
                    other_set = set(other_colors)
                    if len(other_set) == 3:
                        # Forbidden: any color not in other_set
                        # But that's 117 colors. We just need to know which to avoid.
                        # Actually, forbidden means: colors that WOULD create rainbow.
                        # If len(other_set) == 3, then c must be in other_set to avoid rainbow.
                        # So forbidden = all colors NOT in other_set.
                        # But we can just say: allowed = other_set (for this constraint)
                        for c in range(MAX_COLOR):
                            if c not in other_set:
                                forbidden.add(c)
                
                # Find a good new color
                new_color = None
                used_colors = set(np.unique(colors))
                
                # Try to introduce new colors first
                unused = [c for c in range(MAX_COLOR) if c not in used_colors]
                random.shuffle(unused)
                
                for c in unused[:30]:
                    if c not in forbidden and c != old_color:
                        new_color = c
                        break
                
                if new_color is None:
                    # Try existing colors
                    existing = [c for c in used_colors if c != old_color]
                    random.shuffle(existing)
                    for c in existing[:30]:
                        if c not in forbidden:
                            new_color = c
                            break
                
                if new_color is None or new_color == old_color:
                    continue
                
                colors[e] = new_color
                improved = True
                no_improve_count = 0
                
                current_distinct = _count_distinct(colors)
                if current_distinct > best_distinct:
                    best_distinct = current_distinct
                    best_colors = colors.copy()
                
                if time.time() >= deadline:
                    return best_colors.tolist()
        
        # If we're still within deadline, do another restart
        if time.time() >= deadline - 2.0:
            break
    
    if best_colors is None:
        # Fallback: return a valid 2-coloring
        colors = np.zeros(NEDGES, dtype=np.int32)
        for i in range(N):
            for j in range(i + 1, N):
                idx = IDX_MAP[(i, j)]
                if i >= 8 and j >= 8:
                    colors[idx] = 1
                else:
                    colors[idx] = 0
        best_colors = colors
    
    return best_colors.tolist()

# EVOLVE-BLOCK-END
