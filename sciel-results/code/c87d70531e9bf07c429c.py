import time
import random
import numpy as np
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

def _entropy_penalty(colors_np):
    """Compute negative entropy of color distribution (lower is better for high diversity)."""
    counts = np.bincount(colors_np, minlength=MAX_COLOR)
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        return 0.0
    probs = nonzero / nonzero.sum()
    return -np.sum(probs * np.log(probs))

# Precompute
IDX_MAP, CYCLES, EDGE_TO_CYCLES = _get_edge_indices()
NUM_CYCLES = len(CYCLES)

def _initial_valid_coloring():
    """Generate a valid 2-coloring as starting point."""
    colors = np.zeros(NEDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            idx = IDX_MAP[(i, j)]
            if i >= 8 and j >= 8:
                colors[idx] = 1
            else:
                colors[idx] = 0
    return colors

def _initial_algebraic_coloring(K):
    """Generate an algebraic coloring using (i+j) % K."""
    colors = np.zeros(NEDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            idx = IDX_MAP[(i, j)]
            colors[idx] = (i + j) % K
    return colors

def _is_cycle_valid(cycle_idx, colors):
    """Check if a single cycle is valid (not rainbow)."""
    c = CYCLES[cycle_idx]
    col0 = colors[c[0]]
    col1 = colors[c[1]]
    col2 = colors[c[2]]
    col3 = colors[c[3]]
    # Check if all four are distinct
    if col0 == col1 or col0 == col2 or col0 == col3:
        return True
    if col1 == col2 or col1 == col3:
        return True
    if col2 == col3:
        return True
    return False

def _local_search_sa(colors, deadline, rng):
    """
    Perform simulated annealing on a single coloring.
    Uses entropy-aware acceptance and incremental validity checking.
    Returns the best coloring found during this search.
    """
    best_colors = colors.copy()
    best_distinct = _count_distinct(colors)
    best_entropy = _entropy_penalty(colors)
    
    # Precompute for each edge: list of cycle indices
    edge_cycle_list = EDGE_TO_CYCLES
    
    start_time = time.time()
    
    # Simulated annealing parameters
    T_initial = 5.0
    T_final = 0.1
    cooling_rate = 0.9995
    
    current_distinct = _count_distinct(colors)
    current_entropy = _entropy_penalty(colors)
    
    while time.time() < deadline:
        T = T_initial * (T_final / T_initial) ** ((time.time() - start_time) / max(deadline - start_time, 1e-9))
        if T < T_final:
            T = T_final
            
        # Shuffle edge order
        edge_order = list(range(NEDGES))
        rng.shuffle(edge_order)
        
        for e in edge_order:
            if time.time() >= deadline:
                break
            
            old_color = colors[e]
            
            # Compute forbidden colors: colors that would create a rainbow cycle
            allowed_colors = None  # None means all colors allowed initially
            
            for ci in edge_cycle_list[e]:
                cycle = CYCLES[ci]
                other_edges = [cycle[k] for k in range(4) if cycle[k] != e]
                other_colors = [colors[oe] for oe in other_edges]
                other_set = set(other_colors)
                
                if len(other_set) == 3:
                    # This edge's color MUST be in other_set to avoid rainbow
                    if allowed_colors is None:
                        allowed_colors = set(other_set)
                    else:
                        allowed_colors &= other_set
            
            # Determine candidate colors
            if allowed_colors is not None:
                # Must pick from allowed_colors
                candidates = list(allowed_colors)
                if old_color in candidates:
                    candidates.remove(old_color)
                if not candidates:
                    continue
            else:
                # All colors are allowed
                # Use color-frequency heuristic: prefer under-represented colors
                color_counts = np.bincount(colors, minlength=MAX_COLOR)
                unused = [c for c in range(MAX_COLOR) if color_counts[c] == 0]
                if unused:
                    candidates = unused[:30]
                else:
                    freqs = [(color_counts[c], c) for c in range(MAX_COLOR) if color_counts[c] > 0]
                    freqs.sort()
                    candidates = [c for _, c in freqs[:30]]
                
                # Remove old color
                if old_color in candidates:
                    candidates.remove(old_color)
                if not candidates:
                    continue
            
            # Pick a candidate
            new_color = rng.choice(candidates)
            
            # Check validity incrementally
            valid = True
            for ci in edge_cycle_list[e]:
                colors[e] = new_color
                if not _is_cycle_valid(ci, colors):
                    valid = False
                    break
            
            if not valid:
                colors[e] = old_color
                continue
            
            # Valid move - compute new metrics
            new_distinct = _count_distinct(colors)
            new_entropy = _entropy_penalty(colors)
            
            # Simulated annealing acceptance
            delta_distinct = new_distinct - current_distinct
            delta_entropy = new_entropy - current_entropy
            
            # Objective: maximize distinct, secondary: maximize entropy
            # Use a combined score with entropy as smooth surrogate
            if delta_distinct > 0 or (delta_distinct == 0 and delta_entropy > 0):
                # Always accept improvements
                accept = True
            else:
                # Compute acceptance probability
                # Penalty for decreasing distinct (strong) and decreasing entropy (weak)
                penalty = -10.0 * delta_distinct - 0.5 * delta_entropy
                if penalty >= 0:
                    accept = True
                else:
                    accept = (rng.random() < np.exp(penalty / T))
            
            if accept:
                current_distinct = new_distinct
                current_entropy = new_entropy
                if new_distinct > best_distinct or (new_distinct == best_distinct and new_entropy > best_entropy):
                    best_distinct = new_distinct
                    best_entropy = new_entropy
                    best_colors = colors.copy()
            else:
                colors[e] = old_color
        
    return best_colors, best_distinct

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_overall_colors = None
    best_overall_distinct = 0
    
    rng = random.Random(42)
    
    # Try algebraic colorings first as they can give high diversity quickly
    for K in range(2, 20):
        colors = _initial_algebraic_coloring(K)
        if _is_valid_vectorized(colors):
            distinct = _count_distinct(colors)
            if distinct > best_overall_distinct:
                best_overall_distinct = distinct
                best_overall_colors = colors.copy()
    
    # Multi-start simulated annealing
    num_restarts = 0
    while time.time() < deadline - 5.0:
        # Generate initial valid coloring - mix of algebraic and random perturbations
        if rng.random() < 0.5:
            K = rng.randint(2, 10)
            colors = _initial_algebraic_coloring(K)
            if not _is_valid_vectorized(colors):
                colors = _initial_valid_coloring()
        else:
            colors = _initial_valid_coloring()
        
        # Add some random valid perturbations to diversify
        for _ in range(rng.randint(0, 15)):
            e = rng.randint(0, NEDGES - 1)
            new_color = rng.randint(0, MAX_COLOR - 1)
            colors[e] = new_color
            if not _is_valid_vectorized(colors):
                colors[e] = 0  # Revert to safe color
        
        # Run simulated annealing with time limit
        local_deadline = min(time.time() + 10.0, deadline - 2.0)
        colors, distinct = _local_search_sa(colors, local_deadline, rng)
        
        if distinct > best_overall_distinct:
            best_overall_distinct = distinct
            best_overall_colors = colors.copy()
        
        num_restarts += 1
    
    # Final polish pass: try to improve the best solution found
    if best_overall_colors is not None and time.time() < deadline - 1.0:
        polish_deadline = deadline - 0.5
        best_overall_colors, best_overall_distinct = _local_search_sa(
            best_overall_colors, polish_deadline, rng
        )
    
    if best_overall_colors is None:
        best_overall_colors = _initial_valid_coloring()
    
    return best_overall_colors.tolist()
