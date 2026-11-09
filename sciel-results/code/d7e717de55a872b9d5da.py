"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
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

# Precompute K4 edge arrays as numpy for fast access
K4_EDGES_ARR = np.array(K4_EDGES, dtype=np.int32)  # shape (1820, 6)

def _count_distinct_6(c0, c1, c2, c3, c4, c5):
    return len({c0, c1, c2, c3, c4, c5})

def _compute_violations(colors):
    """Compute violation status and count using numpy."""
    colors = np.asarray(colors, dtype=np.int32)
    # For each K4, get the 6 edge colors
    k4_colors = colors[K4_EDGES_ARR]  # shape (1820, 6)
    # Count distinct colors per K4
    # A K4 has < 5 distinct colors iff at least 2 pairs share a color
    # More simply: use set logic via sorting
    # For speed, check: distinct < 5 means some color appears >= 2 times
    # and total distinct <= 4
    # Fast check: sort and count unique
    k4_sorted = np.sort(k4_colors, axis=1)
    # Count distinct: number of positions where value changes + 1
    diffs = np.diff(k4_sorted, axis=1)
    distinct = 1 + np.sum(diffs != 0, axis=1)
    violations = distinct < 5
    return violations, int(np.sum(violations))

def _construct():
    deadline = time.time() + 110
    random.seed(42)
    np.random.seed(42)
    
    # Try multiple initial constructions
    best = None
    best_violations = float('inf')
    best_violations_arr = None
    
    # Try (i+j) % k for various k
    for k in range(13, 25):
        if time.time() > deadline - 5:
            break
        cand = np.array([(i + j) % k for i in range(N) for j in range(i + 1, N)], dtype=np.int32)
        viol, tv = _compute_violations(cand)
        if tv < best_violations:
            best_violations = tv
            best = cand
            best_violations_arr = viol
        if tv == 0:
            break
    
    # Also try random starts
    for _ in range(100):
        if time.time() > deadline - 3:
            break
        max_c = random.randint(13, 20)
        cand = np.array([random.randint(0, max_c - 1) for _ in range(NUM_EDGES)], dtype=np.int32)
        viol, tv = _compute_violations(cand)
        if tv < best_violations:
            best_violations = tv
            best = cand
            best_violations_arr = viol
        if tv == 0:
            break
    
    if best is None:
        return [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]
    
    current = best.copy()
    current_violations = best_violations
    current_violations_arr = best_violations_arr.copy()
    
    # Track color frequencies
    MAX_COLOR = 30
    color_freq = np.zeros(MAX_COLOR, dtype=np.int32)
    for c in current:
        if c < MAX_COLOR:
            color_freq[c] += 1
    current_colors = int(np.sum(color_freq > 0))
    
    # Simulated annealing to reduce violations
    temp = 5.0
    cooling = 0.9995
    min_temp = 0.1
    
    while time.time() < deadline - 5:
        if current_violations == 0:
            # Phase 1: Reduce colors via local search
            # Find edges whose color appears only once or rarely
            for _ in range(5000):
                if time.time() > deadline - 3:
                    break
                if current_violations > 0:
                    break
                    
                # Find a color with minimum frequency > 0
                nonzero_freq = color_freq[color_freq > 0]
                if len(nonzero_freq) == 0:
                    break
                min_freq = int(np.min(nonzero_freq))
                
                # Find a color with this frequency
                rare_colors = np.where(color_freq == min_freq)[0]
                old_c = int(random.choice(rare_colors))
                
                # Find an edge with this color
                edges_with_old = np.where(current == old_c)[0]
                if len(edges_with_old) == 0:
                    continue
                e_idx = int(random.choice(edges_with_old))
                
                # Choose target color: one that's already used
                used_colors = np.where((color_freq > 0) & (np.arange(MAX_COLOR) != old_c))[0]
                if len(used_colors) == 0:
                    continue
                new_c = int(random.choice(used_colors))
                
                if new_c == old_c:
                    continue
                
                # Check if move is valid (doesn't create violations)
                affected_k4s = EDGE_TO_K4[e_idx]
                valid = True
                new_violations = current_violations
                
                for k_idx in affected_k4s:
                    edges = K4_EDGES[k_idx]
                    c0 = int(current[edges[0]]) if edges[0] != e_idx else new_c
                    c1 = int(current[edges[1]]) if edges[1] != e_idx else new_c
                    c2 = int(current[edges[2]]) if edges[2] != e_idx else new_c
                    c3 = int(current[edges[3]]) if edges[3] != e_idx else new_c
                    c4 = int(current[edges[4]]) if edges[4] != e_idx else new_c
                    c5 = int(current[edges[5]]) if edges[5] != e_idx else new_c
                    
                    new_distinct = _count_distinct_6(c0, c1, c2, c3, c4, c5)
                    new_status = new_distinct < 5
                    
                    if current_violations_arr[k_idx] and not new_status:
                        new_violations -= 1
                    elif not current_violations_arr[k_idx] and new_status:
                        new_violations += 1
                        valid = False
                        break
                
                if not valid or new_violations > 0:
                    continue
                
                # Check color count improvement
                new_color_count = current_colors
                if color_freq[old_c] == 1:
                    new_color_count -= 1
                if color_freq[new_c] == 0:
                    new_color_count += 1
                
                if new_color_count < current_colors:
                    # Apply move
                    current[e_idx] = new_c
                    color_freq[old_c] -= 1
                    color_freq[new_c] += 1
                    current_colors = new_color_count
                    current_violations = new_violations
                    
                    # Update K4 statuses
                    for k_idx in affected_k4s:
                        edges = K4_EDGES[k_idx]
                        current_violations_arr[k_idx] = _count_distinct_6(
                            int(current[edges[0]]), int(current[edges[1]]),
                            int(current[edges[2]]), int(current[edges[3]]),
                            int(current[edges[4]]), int(current[edges[5]])
                        ) < 5
        else:
            # Phase 2: Reduce violations via simulated annealing
            violated_k4s = np.where(current_violations_arr)[0]
            if len(violated_k4s) == 0:
                current_violations = 0
                continue
                
            k_idx = int(random.choice(violated_k4s))
            edges = K4_EDGES[k_idx]
            e_idx = int(random.choice(edges))
            old_c = int(current[e_idx])
            
            # Try different new colors
            improved = False
            for _ in range(30):
                new_c = random.randint(0, 25)
                if new_c == old_c:
                    continue
                
                valid = True
                new_violations = current_violations
                delta = 0
                
                for k2 in EDGE_TO_K4[e_idx]:
                    e0, e1, e2, e3, e4, e5 = K4_EDGES[k2]
                    c0 = int(current[e0]) if e0 != e_idx else new_c
                    c1 = int(current[e1]) if e1 != e_idx else new_c
                    c2 = int(current[e2]) if e2 != e_idx else new_c
                    c3 = int(current[e3]) if e3 != e_idx else new_c
                    c4 = int(current[e4]) if e4 != e_idx else new_c
                    c5 = int(current[e5]) if e5 != e_idx else new_c
                    
                    new_distinct = _count_distinct_6(c0, c1, c2, c3, c4, c5)
                    new_status = new_distinct < 5
                    old_status = current_violations_arr[k2]
                    
                    if old_status and not new_status:
                        delta -= 1
                    elif not old_status and new_status:
                        delta += 1
                        if delta > 2:
                            valid = False
                            break
                    
                    # Tentatively update
                    current_violations_arr[k2] = new_status
                
                if not valid:
                    # Revert
                    for k2 in EDGE_TO_K4[e_idx]:
                        e0, e1, e2, e3, e4, e5 = K4_EDGES[k2]
                        current_violations_arr[k2] = _count_distinct_6(
                            int(current[e0]), int(current[e1]),
                            int(current[e2]), int(current[e3]),
                            int(current[e4]), int(current[e5])
                        ) < 5
                    continue
                
                new_violations = current_violations + delta
                
                if new_violations < current_violations:
                    # Accept
                    current[e_idx] = new_c
                    color_freq[old_c] -= 1
                    color_freq[new_c] += 1
                    current_violations = new_violations
                    improved = True
                    break
                else:
                    # Accept with probability exp(-delta / temp)
                    if delta > 0:
                        prob = np.exp(-delta / max(temp, 0.01))
                        if random.random() < prob:
                            current[e_idx] = new_c
                            color_freq[old_c] -= 1
                            color_freq[new_c] += 1
                            current_violations = new_violations
                            improved = True
                            break
                    # Revert
                    for k2 in EDGE_TO_K4[e_idx]:
                        e0, e1, e2, e3, e4, e5 = K4_EDGES[k2]
                        current_violations_arr[k2] = _count_distinct_6(
                            int(current[e0]), int(current[e1]),
                            int(current[e2]), int(current[e3]),
                            int(current[e4]), int(current[e5])
                        ) < 5
            
            if not improved:
                # Try a random edge
                e_idx = random.randint(0, NUM_EDGES - 1)
                old_c = int(current[e_idx])
                new_c = random.randint(0, 25)
                if new_c != old_c:
                    delta = 0
                    for k2 in EDGE_TO_K4[e_idx]:
                        e0, e1, e2, e3, e4, e5 = K4_EDGES[k2]
                        c0 = int(current[e0]) if e0 != e_idx else new_c
                        c1 = int(current[e1]) if e1 != e_idx else new_c
                        c2 = int(current[e2]) if e2 != e_idx else new_c
                        c3 = int(current[e3]) if e3 != e_idx else new_c
                        c4 = int(current[e4]) if e4 != e_idx else new_c
                        c5 = int(current[e5]) if e5 != e_idx else new_c
                        
                        new_distinct = _count_distinct_6(c0, c1, c2, c3, c4, c5)
                        new_status = new_distinct < 5
                        old_status = current_violations_arr[k2]
                        
                        if old_status and not new_status:
                            delta -= 1
                        elif not old_status and new_status:
                            delta += 1
                        
                        current_violations_arr[k2] = new_status
                    
                    new_violations = current_violations + delta
                    if new_violations < current_violations:
                        current[e_idx] = new_c
                        color_freq[old_c] -= 1
                        color_freq[new_c] += 1
                        current_violations = new_violations
                    else:
                        if delta > 0:
                            prob = np.exp(-delta / max(temp, 0.01))
                            if random.random() < prob:
                                current[e_idx] = new_c
                                color_freq[old_c] -= 1
                                color_freq[new_c] += 1
                                current_violations = new_violations
                            else:
                                # Revert
                                for k2 in EDGE_TO_K4[e_idx]:
                                    e0, e1, e2, e3, e4, e5 = K4_EDGES[k2]
                                    current_violations_arr[k2] = _count_distinct_6(
                                        int(current[e0]), int(current[e1]),
                                        int(current[e2]), int(current[e3]),
                                        int(current[e4]), int(current[e5])
                                    ) < 5
                        else:
                            current[e_idx] = new_c
                            color_freq[old_c] -= 1
                            color_freq[new_c] += 1
                            current_violations = new_violations
            
            temp *= cooling
            if temp < min_temp:
                temp = min_temp
    
    # Final: count distinct colors used
    return [int(c) for c in current]

def construct():
    return _construct()
# EVOLVE-BLOCK-END
