import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def _edge_index(i, j):
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_cycles():
    cycles = []
    for a in range(N - 3):
        for b in range(a + 1, N - 2):
            for c in range(b + 1, N - 1):
                for d in range(c + 1, N):
                    cycles.append((_edge_index(a,b), _edge_index(b,c), _edge_index(c,d), _edge_index(a,d)))
                    cycles.append((_edge_index(a,b), _edge_index(b,d), _edge_index(c,d), _edge_index(a,c)))
                    cycles.append((_edge_index(a,c), _edge_index(b,c), _edge_index(b,d), _edge_index(a,d)))
    return cycles

CYCLES = _build_cycles()
NUM_CYCLES = len(CYCLES)

def _build_edge_to_cycles():
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(CYCLES):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return edge_to_cycles

EDGE_TO_CYCLES = _build_edge_to_cycles()
EDGE_DEG = np.array([len(EDGE_TO_CYCLES[e]) for e in range(NUM_EDGES)], dtype=np.int32)

def _count_violations(colors):
    colors_arr = np.asarray(colors, dtype=np.int32)
    violations = 0
    for cyc in CYCLES:
        c0, c1, c2, c3 = colors_arr[cyc[0]], colors_arr[cyc[1]], colors_arr[cyc[2]], colors_arr[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            violations += 1
    return violations

def _count_distinct_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    random.seed(42)
    
    # Precompute cycle edge arrays for fast access
    cycle_edges = np.array(CYCLES, dtype=np.int32)  # shape (NUM_CYCLES, 4)
    
    best_colors = None
    best_score = -1  # score = distinct_colors - penalty
    
    def _score(colors_arr):
        """Compute objective: maximize distinct colors, penalize violations."""
        distinct = len(np.unique(colors_arr))
        # Count violations vectorized
        # For each cycle, check if all 4 colors are distinct
        cyc_colors = colors_arr[cycle_edges]  # shape (NUM_CYCLES, 4)
        # A cycle is a violation if all 4 values are distinct
        # Check: for each row, are all 4 values different?
        v0, v1, v2, v3 = cyc_colors[:,0], cyc_colors[:,1], cyc_colors[:,2], cyc_colors[:,3]
        viol = (v0 != v1) & (v0 != v2) & (v0 != v3) & (v1 != v2) & (v1 != v3) & (v2 != v3)
        n_viol = np.sum(viol)
        # Objective: high diversity, no violations
        # Use a large penalty for violations to ensure validity is prioritized
        penalty = n_viol * 100
        return distinct - penalty
    
    def _is_valid(colors_arr):
        cyc_colors = colors_arr[cycle_edges]
        v0, v1, v2, v3 = cyc_colors[:,0], cyc_colors[:,1], cyc_colors[:,2], cyc_colors[:,3]
        viol = (v0 != v1) & (v0 != v2) & (v0 != v3) & (v1 != v2) & (v1 != v3) & (v2 != v3)
        return np.sum(viol) == 0
    
    def _n_violations(colors_arr):
        cyc_colors = colors_arr[cycle_edges]
        v0, v1, v2, v3 = cyc_colors[:,0], cyc_colors[:,1], cyc_colors[:,2], cyc_colors[:,3]
        viol = (v0 != v1) & (v0 != v2) & (v0 != v3) & (v1 != v2) & (v1 != v3) & (v2 != v3)
        return np.sum(viol)
    
    # Simulated annealing
    # Initial: start with a structured low-violation coloring
    # Strategy: use a "block" coloring where edges within certain groups share colors
    
    # Start with all zeros
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    current_score = _score(colors)
    
    # Simulated annealing parameters
    T_start = 50.0
    T_end = 0.1
    cooling_rate = 0.9995
    
    T = T_start
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Check time more frequently
        if random.random() < 0.01 and time.time() > deadline - 5.0:
            break
        
        # Pick a random edge to change
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Choose new color: biased towards unused colors to increase diversity
        # with some probability, pick a truly random color
        if random.random() < 0.7:
            # Prefer colors not currently used much
            new_color = random.randint(0, MAX_COLOR)
        else:
            # Pick from a smaller range to encourage diversity in a structured way
            new_color = random.randint(0, min(MAX_COLOR, 30 + int(T)))
        
        if new_color == old_color:
            continue
        
        # Compute delta score
        # Only need to check cycles containing edge e
        affected_cycles = EDGE_TO_CYCLES[e]
        
        # Compute old violations in affected cycles
        old_viol = 0
        new_viol = 0
        for ci in affected_cycles:
            cyc = CYCLES[ci]
            # Old colors
            oc = [colors[cyc[i]] for i in range(4)]
            # New colors
            nc = oc[:]
            for idx in range(4):
                if cyc[idx] == e:
                    nc[idx] = new_color
                    break
            old_viol += 1 if len(set(oc)) == 4 else 0
            new_viol += 1 if len(set(nc)) == 4 else 0
        
        # Old distinct count
        old_colors_set = set(colors.tolist())
        old_distinct = len(old_colors_set)
        
        # New distinct count
        new_colors_set = set(colors.tolist())
        # Remove old_color if it was the only instance
        # Actually, let's just compute it properly
        # Since we're changing one edge, the distinct count can change by at most 1
        # If new_color is not in old_colors_set, distinct increases by 1
        # If old_color appears only once and new_color is in old_colors_set, distinct decreases by 1
        # Otherwise, distinct stays the same
        
        # Count occurrences of old_color
        old_count = np.sum(colors == old_color)
        new_in_set = new_color in old_colors_set
        
        if new_in_set:
            if old_count == 1:
                new_distinct = old_distinct - 1
            else:
                new_distinct = old_distinct
        else:
            new_distinct = old_distinct + 1
        
        old_penalty = old_viol * 100
        new_penalty = new_viol * 100
        
        delta = (new_distinct - new_penalty) - (old_distinct - old_penalty)
        
        # Accept if improving, or with probability exp(-delta/T) if worsening
        if delta >= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            colors[e] = new_color
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_colors = colors.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Cool down
        T = max(T * cooling_rate, T_end)
        
        # Reset temperature if stuck for too long
        if no_improve_count > max_no_improve:
            T = min(T * 10, T_start)
            no_improve_count = 0
    
    # Final check: ensure best_colors is valid
    if best_colors is not None:
        if _is_valid(best_colors):
            return best_colors.tolist()
    
    # Fallback: try to repair current colors
    # If current has violations, try to fix them
    for _ in range(10000):
        if time.time() > deadline - 2.0:
            break
        nv = _n_violations(colors)
        if nv == 0:
            break
        # Find a violating cycle
        cyc_colors = colors[cycle_edges]
        v0, v1, v2, v3 = cyc_colors[:,0], cyc_colors[:,1], cyc_colors[:,2], cyc_colors[:,3]
        viol_mask = (v0 != v1) & (v0 != v2) & (v0 != v3) & (v1 != v2) & (v1 != v3) & (v2 != v3)
        viol_indices = np.where(viol_mask)[0]
        if len(viol_indices) == 0:
            break
        ci = random.choice(viol_indices.tolist())
        cyc = CYCLES[ci]
        # Pick an edge in this cycle and change it to match another edge's color
        e = random.choice(cyc)
        other_edges = [cyc[i] for i in range(4) if cyc[i] != e]
        target_color = colors[random.choice(other_edges)]
        colors[e] = target_color
    
    if _is_valid(colors):
        return colors.tolist()
    
    # Last resort: all zeros
    return [0] * NUM_EDGES
