import time
import numpy as np
import random
from itertools import combinations

N = 16
NEDGES = N * (N - 1) // 2
MAX_COLOR = 120

# Precompute edge indices and cycle structures
def _get_edge_indices():
    """Return edge index mapping, cycle structures, and edge-to-cycle mapping."""
    idx_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx_map[(i, j)] = idx
            idx_map[(j, i)] = idx
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
    
    # Map each edge to the list of (cycle_index, position_in_cycle)
    edge_to_cycles = [[] for _ in range(NEDGES)]
    for ci, cycle in enumerate(cycles):
        for pos in range(4):
            e = cycle[pos]
            edge_to_cycles[e].append((ci, pos))
            
    return idx_map, cycles_np, edge_to_cycles

IDX_MAP, CYCLES, EDGE_TO_CYCLES = _get_edge_indices()
NUM_CYCLES = len(CYCLES)

def _is_valid_vectorized(colors_np):
    """
    Check if colors_np (shape NEDGES,) is valid.
    Returns True if no cycle has 4 distinct colors.
    """
    cycle_colors = colors_np[CYCLES]
    sorted_colors = np.sort(cycle_colors, axis=1)
    adjacent_equal = (sorted_colors[:, 0] == sorted_colors[:, 1]) | \
                     (sorted_colors[:, 1] == sorted_colors[:, 2]) | \
                     (sorted_colors[:, 2] == sorted_colors[:, 3])
    return bool(np.all(adjacent_equal))

def _count_distinct(colors_np):
    return len(np.unique(colors_np))

def _initial_valid_coloring(rng):
    """
    Generate a valid 2-coloring.
    Split vertices into two sets of 8.
    Edges within set 0: color 0
    Edges within set 1: color 1
    Edges between sets: color 0
    This is valid because any 4-cycle will have at most 2 colors.
    We add some random valid perturbations to break symmetry.
    """
    colors = np.zeros(NEDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            idx = IDX_MAP[(i, j)]
            if i >= 8 and j >= 8:
                colors[idx] = 1
            else:
                colors[idx] = 0
                
    # Add random valid perturbations to diversify the starting point
    for _ in range(rng.randint(20, 50)):
        e = rng.randint(0, NEDGES - 1)
        old_color = colors[e]
        new_color = rng.randint(2, MAX_COLOR - 1)
        colors[e] = new_color
        if _is_valid_vectorized(colors):
            pass
        else:
            colors[e] = old_color
            
    return colors

def _local_search(colors, deadline, rng):
    """
    Perform local search on a single coloring.
    Uses forbidden-set pruning and color-frequency heuristic.
    """
    best_colors = colors.copy()
    best_distinct = _count_distinct(colors)
    
    # Maintain color counts for fast distinct count update
    color_counts = np.bincount(colors, minlength=MAX_COLOR)
    current_distinct = int(np.sum(color_counts > 0))
    
    # Precompute for each edge: list of (cycle_idx, position_in_cycle)
    # This is already done in global EDGE_TO_CYCLES
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 500 # Reset perturbation if stuck
    
    while time.time() < deadline:
        # Pick a random edge to modify
        e = rng.randint(0, NEDGES - 1)
        
        old_color = colors[e]
        
        # Compute allowed colors based on constraints from cycles containing e
        # If a cycle has 3 distinct colors on the other edges, the 4th edge MUST match one of them.
        allowed_colors = None
        
        for ci, pos in EDGE_TO_CYCLES[e]:
            cycle = CYCLES[ci]
            # Get the other 3 edges in this cycle
            other_edges = [cycle[k] for k in range(4) if k != pos]
            other_colors = [colors[oe] for oe in other_edges]
            other_set = set(other_colors)
            
            if len(other_set) == 3:
                # Constraint: color must be in other_set
                if allowed_colors is None:
                    allowed_colors = set(other_set)
                else:
                    allowed_colors &= other_set
            
            # If len(other_set) < 3, no constraint from this specific cycle
            # (any color is fine as it will create a duplicate)
            
        # Determine candidate colors
        if allowed_colors is not None:
            # Must pick from allowed_colors
            candidates = [c for c in allowed_colors if c != old_color]
            if not candidates:
                # No valid move for this edge that changes color
                continue
        else:
            # No constraints from cycles (or only duplicates in cycles)
            # Use color-frequency heuristic: prefer under-represented or unused colors
            # To maximize distinct colors, we want to introduce new colors or spread existing ones.
            
            # Find unused colors
            unused = [c for c in range(MAX_COLOR) if color_counts[c] == 0]
            
            if rng.random() < 0.7 and unused:
                # High probability of trying a completely new color
                candidates = unused[:10]
            else:
                # Try to spread low-frequency colors
                # Sort colors by frequency (ascending), exclude old_color
                # Note: This is O(MAX_COLOR log MAX_COLOR), but MAX_COLOR=120 is small
                freqs = [(color_counts[c], c) for c in range(MAX_COLOR) if c != old_color]
                freqs.sort()
                # Take the lowest frequency ones
                candidates = [c for _, c in freqs[:10]]
                
            if not candidates:
                continue
            
        # Pick a random candidate
        new_color = rng.choice(candidates)
        
        # Apply move
        colors[e] = new_color
        
        # Check validity (should be valid by construction if allowed_colors logic is correct, 
        # but we verify just in case of logic errors or if allowed_colors was None but 
        # some other subtle constraint exists - though for 4-cycles, the logic above is sufficient)
        if _is_valid_vectorized(colors):
            # Update state
            color_counts[old_color] -= 1
            color_counts[new_color] += 1
            
            if color_counts[old_color] == 0:
                current_distinct -= 1
            if color_counts[new_color] == 1:
                current_distinct += 1
                
            if current_distinct > best_distinct:
                best_distinct = current_distinct
                best_colors = colors.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # If we are stuck, try a perturbation
            if no_improve_count > max_no_improve:
                # Revert the last move if it didn't improve? Or just randomize?
                # Let's just randomize a few edges to escape local minima
                for _ in range(rng.randint(5, 15)):
                    pe = rng.randint(0, NEDGES - 1)
                    old_pc = colors[pe]
                    new_pc = rng.randint(0, MAX_COLOR - 1)
                    colors[pe] = new_pc
                    if _is_valid_vectorized(colors):
                        color_counts[old_pc] -= 1
                        color_counts[new_pc] += 1
                        if color_counts[old_pc] == 0:
                            current_distinct -= 1
                        if color_counts[new_pc] == 1:
                            current_distinct += 1
                    else:
                        colors[pe] = old_pc
                no_improve_count = 0
        else:
            # Revert if invalid (should be rare)
            colors[e] = old_color
            
    return best_colors, best_distinct

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_overall_colors = None
    best_overall_distinct = 0
    
    rng = random.Random(42)
    
    num_restarts = 0
    while time.time() < deadline - 5.0:
        # Generate initial valid coloring
        colors = _initial_valid_coloring(rng)
        
        # Determine time for this SA run
        # Use more time for later runs if we are close to the end? 
        # Or just fixed chunks? Fixed chunks are safer.
        local_deadline = min(time.time() + 8.0, deadline - 2.0)
        
        colors, distinct = _local_search(colors, local_deadline, rng)
        
        if distinct > best_overall_distinct:
            best_overall_distinct = distinct
            best_overall_colors = colors.copy()
            
        num_restarts += 1
        
        # If we have enough time, do another restart
        if time.time() > deadline - 10.0:
            break

    if best_overall_colors is None:
        best_overall_colors = _initial_valid_coloring(rng)
    
    return best_overall_colors.tolist()

# EVOLVE-BLOCK-END
