"""Optimized coloring search for K16 with no rainbow C4 using bitset constraints."""
import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119
NUM_COLORS = 120

# Precompute edge indices
def edge_idx(i, j):
    return i * (N - 1) - i * (i - 1) // 2 + j - i - 1

_edge_idx = [[0] * N for _ in range(N)]
for i in range(N):
    for j in range(i + 1, N):
        _edge_idx[i][j] = edge_idx(i, j)
        _edge_idx[j][i] = _edge_idx[i][j]

# Build all 4-cycles (three per 4-vertex subset)
def build_four_cycles():
    cycles = []
    for verts in combinations(range(N), 4):
        a, b, c, d = verts
        cycles.append((_edge_idx[a][b], _edge_idx[b][c], _edge_idx[c][d], _edge_idx[d][a]))
        cycles.append((_edge_idx[a][b], _edge_idx[b][d], _edge_idx[d][c], _edge_idx[c][a]))
        cycles.append((_edge_idx[a][c], _edge_idx[c][b], _edge_idx[b][d], _edge_idx[d][a]))
    return cycles

FOUR_CYCLES = build_four_cycles()
NUM_CYCLES = len(FOUR_CYCLES)

# For each edge, precompute the cycles it participates in with the other 3 edges
edge_cycle_info = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(FOUR_CYCLES):
    e0, e1, e2, e3 = cycle
    edge_cycle_info[e0].append((ci, [e1, e2, e3]))
    edge_cycle_info[e1].append((ci, [e0, e2, e3]))
    edge_cycle_info[e2].append((ci, [e0, e1, e3]))
    edge_cycle_info[e3].append((ci, [e0, e1, e2]))

# Precompute: for each edge, which other edges are in cycles with it
# This helps us quickly check validity
def is_valid_change(colors, edge, new_color):
    """Check if setting edge to new_color keeps all cycles valid."""
    for ci, other_edges in edge_cycle_info[edge]:
        c1 = colors[other_edges[0]]
        c2 = colors[other_edges[1]]
        c3 = colors[other_edges[2]]
        # If the other 3 are all distinct, new_color must match one of them
        if c1 != c2 and c1 != c3 and c2 != c3:
            if new_color != c1 and new_color != c2 and new_color != c3:
                return False
    return True

def count_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    random.seed(42)
    
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    # Strategy: Use a structured approach
    # 1. Start with a valid base coloring (all zeros)
    # 2. Systematically try to add new colors using constraint propagation
    # 3. Use local search with 2-move swaps to escape local optima
    
    # Phase 1: Greedy construction with multiple orderings
    for trial in range(20):
        if time.time() > deadline:
            break
        
        colors = [0] * NUM_EDGES
        used_colors = {0}
        
        # Different edge orderings
        edge_order = list(range(NUM_EDGES))
        if trial % 2 == 0:
            random.shuffle(edge_order)
        else:
            # Group by vertex to create structure
            edge_order.sort(key=lambda e: (e // (N - 1), e))
            random.shuffle(edge_order)
        
        for e in edge_order:
            if time.time() > deadline:
                break
            
            # Find the best new color for this edge
            # A color is valid if for every cycle containing e,
            # the other 3 edges don't have 3 distinct colors
            # OR if they do, the new color must be one of those 3
            
            # Collect forbidden colors
            forbidden = set()
            must_match = set()  # Colors that must be matched if 3 others are distinct
            
            for ci, other_edges in edge_cycle_info[e]:
                c1 = colors[other_edges[0]]
                c2 = colors[other_edges[1]]
                c3 = colors[other_edges[2]]
                
                if c1 != c2 and c1 != c3 and c2 != c3:
                    # All 3 distinct: new color must be one of {c1, c2, c3}
                    must_match.update([c1, c2, c3])
                else:
                    # At least two are the same: no restriction from this cycle
                    pass
            
            # If must_match is non-empty, we can only use colors in must_match
            # Otherwise, we can use any color
            if must_match:
                # Try to use a new color from must_match
                candidates = [c for c in must_match if c not in used_colors]
                if candidates:
                    colors[e] = random.choice(candidates)
                    used_colors.add(colors[e])
                else:
                    # Use an existing color from must_match
                    colors[e] = random.choice(list(must_match))
            else:
                # No restrictions: try to add a new color
                unused = [c for c in range(NUM_COLORS) if c not in used_colors]
                if unused:
                    colors[e] = random.choice(unused)
                    used_colors.add(colors[e])
        
        cur_count = count_colors(colors)
        if cur_count > best_count:
            best_count = cur_count
            best_colors = colors[:]
    
    # Phase 2: Local search to improve
    if time.time() < deadline - 5.0:
        colors = best_colors[:]
        used_colors = set(colors)
        cur_count = best_count
        
        # Try to add new colors to edges that currently have "common" colors
        # An edge with a common color might be able to take a new color
        # if we first change some neighboring edges
        
        # Simpler approach: try all edges, try all unused colors
        for iteration in range(100):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            for e in range(NUM_EDGES):
                if time.time() > deadline - 2.0:
                    break
                
                # Try to change this edge to a new color
                unused = [c for c in range(NUM_COLORS) if c not in used_colors]
                if not unused:
                    break
                
                for c in random.sample(unused, min(len(unused), 10)):
                    if is_valid_change(colors, e, c):
                        old_c = colors[e]
                        colors[e] = c
                        used_colors.add(c)
                        cur_count += 1
                        improved = True
                        if cur_count > best_count:
                            best_count = cur_count
                            best_colors = colors[:]
                        break
                
                if improved and time.time() > deadline - 2.0:
                    break
            
            if not improved:
                # Try 2-move swaps: change one edge to an existing color,
                # then try to add a new color to another edge
                break
        
        # Phase 3: 2-move local search
        if time.time() < deadline - 3.0:
            colors = best_colors[:]
            used_colors = set(colors)
            cur_count = best_count
            
            for iteration in range(50):
                if time.time() > deadline - 2.0:
                    break
                
                # Pick two random edges
                e1 = random.randint(0, NUM_EDGES - 1)
                e2 = random.randint(0, NUM_EDGES - 1)
                if e1 == e2:
                    continue
                
                # Try to change e1 to an existing color, then e2 to a new color
                used_list = list(used_colors)
                unused = [c for c in range(NUM_COLORS) if c not in used_colors]
                
                if not unused:
                    break
                
                for c1 in random.sample(used_list, min(len(used_list), 5)):
                    if c1 == colors[e1]:
                        continue
                    if not is_valid_change(colors, e1, c1):
                        continue
                    
                    old_e1 = colors[e1]
                    colors[e1] = c1
                    
                    # Now try to add a new color to e2
                    for c2 in random.sample(unused, min(len(unused), 5)):
                        if is_valid_change(colors, e2, c2):
                            colors[e2] = c2
                            used_colors.add(c2)
                            cur_count += 1
                            if cur_count > best_count:
                                best_count = cur_count
                                best_colors = colors[:]
                            # Keep the change
                            break
                    else:
                        # Revert e1
                        colors[e1] = old_e1
                        continue
                    break
    
    return best_colors
