"""Improved program using simulated annealing for edge-colouring of K16."""
# EVOLVE-BLOCK-START
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

# Pre-compute for each K4, the 6 edge positions
K4_EDGE_POS = []
for edges in K4_EDGES:
    K4_EDGE_POS.append(list(edges))

def _count_distinct_fast(colors, edges):
    """Count distinct colors in 6 edges using a small set."""
    seen = set()
    for e in edges:
        seen.add(colors[e])
    return len(seen)

def _compute_violations(colors):
    """Compute violation status and count."""
    k4_status = [False] * NUM_K4
    total_v = 0
    for k_idx in range(NUM_K4):
        edges = K4_EDGES[k_idx]
        c0, c1, c2, c3, c4, c5 = colors[edges[0]], colors[edges[1]], colors[edges[2]], \
                                    colors[edges[3]], colors[edges[4]], colors[edges[5]]
        # Fast distinct count
        if c0 == c1 == c2 == c3 == c4 == c5:
            distinct = 1
        elif c0 == c1 and c0 == c2 and c0 == c3 and c0 == c4 and c0 == c5:
            distinct = 1
        else:
            # Use a small trick: check all pairs
            distinct = 1
            if c1 != c0: distinct += 1
            if c2 != c0 and c2 != c1: distinct += 1
            if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
            if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: distinct += 1
            if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: distinct += 1
        
        if distinct < 5:
            k4_status[k_idx] = True
            total_v += 1
    return k4_status, total_v

def _evaluate_move(colors, k4_status, current_violations, e_idx, new_c):
    """Evaluate changing edge e_idx to new_c. Returns (new_violations, valid, k4_status_changes)."""
    old_c = colors[e_idx]
    if old_c == new_c:
        return current_violations, True, []
    
    new_violations = current_violations
    changes = []
    
    for k_idx in EDGE_TO_K4[e_idx]:
        edges = K4_EDGES[k_idx]
        c0, c1, c2, c3, c4, c5 = colors[edges[0]], colors[edges[1]], colors[edges[2]], \
                                    colors[edges[3]], colors[edges[4]], colors[edges[5]]
        
        # Replace the color at e_idx
        for pos, e in enumerate(edges):
            if e == e_idx:
                if pos == 0: c0 = new_c
                elif pos == 1: c1 = new_c
                elif pos == 2: c2 = new_c
                elif pos == 3: c3 = new_c
                elif pos == 4: c4 = new_c
                elif pos == 5: c5 = new_c
        
        # Count distinct
        if c0 == c1 and c0 == c2 and c0 == c3 and c0 == c4 and c0 == c5:
            distinct = 1
        else:
            distinct = 1
            if c1 != c0: distinct += 1
            if c2 != c0 and c2 != c1: distinct += 1
            if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
            if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: distinct += 1
            if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: distinct += 1
        
        new_status = distinct < 5
        old_status = k4_status[k_idx]
        
        if old_status and not new_status:
            new_violations -= 1
            changes.append((k_idx, False))
        elif not old_status and new_status:
            new_violations += 1
            changes.append((k_idx, True))
    
    return new_violations, True, changes

def _construct():
    deadline = time.time() + 115
    random.seed(42)
    
    # Try multiple initial constructions
    best = None
    best_violations = float('inf')
    best_k4_status = None
    
    # Try (i+j) % k for various k
    for k in range(13, 25):
        if time.time() > deadline - 10:
            break
        cand = [(i + j) % k for i in range(N) for j in range(i + 1, N)]
        ks, tv = _compute_violations(cand)
        if tv < best_violations:
            best_violations = tv
            best = cand
            best_k4_status = ks
        if tv == 0:
            break
    
    # Also try random starts with fewer colors
    for _ in range(300):
        if time.time() > deadline - 8:
            break
        max_c = random.randint(13, 20)
        cand = [random.randint(0, max_c - 1) for _ in range(NUM_EDGES)]
        ks, tv = _compute_violations(cand)
        if tv < best_violations:
            best_violations = tv
            best = cand
            best_k4_status = ks
        if tv == 0:
            break
    
    if best is None:
        return [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]
    
    current = list(best)
    current_violations = best_violations
    current_k4_status = list(best_k4_status)
    
    # Track color frequencies
    MAX_COLOR = 30
    color_freq = [0] * MAX_COLOR
    for c in current:
        if c < MAX_COLOR:
            color_freq[c] += 1
    current_colors = sum(1 for f in color_freq if f > 0)
    
    # Simulated annealing
    T_start = 5.0
    T_end = 0.01
    cooling_rate = (T_end / T_start) ** (1.0 / 100000)
    T = T_start
    
    best_solution = list(current)
    best_colors = current_colors
    best_violations_sa = current_violations
    
    for step in range(100000):
        if time.time() > deadline - 2:
            break
        
        if current_violations == 0:
            # Phase 1: Try to reduce colors
            # Find edges whose color appears only once
            single_use_colors = [c for c in range(MAX_COLOR) if color_freq[c] == 1]
            
            if single_use_colors:
                old_c = random.choice(single_use_colors)
                edges_with_old = [e for e in range(NUM_EDGES) if current[e] == old_c]
                if edges_with_old:
                    e_idx = random.choice(edges_with_old)
                    used_colors = [c for c in range(MAX_COLOR) if color_freq[c] > 0 and c != old_c]
                    if used_colors:
                        new_c = random.choice(used_colors)
                        new_violations, valid, changes = _evaluate_move(current, current_k4_status, current_violations, e_idx, new_c)
                        
                        if valid and new_violations == 0:
                            # Check if this reduces color count
                            new_color_count = current_colors
                            if color_freq[old_c] == 1:
                                new_color_count -= 1
                            if color_freq[new_c] == 0:
                                new_color_count += 1
                            
                            if new_color_count < current_colors:
                                current[e_idx] = new_c
                                color_freq[old_c] -= 1
                                color_freq[new_c] += 1
                                current_colors = new_color_count
                                for k_idx, new_status in changes:
                                    current_k4_status[k_idx] = new_status
                                
                                # Update best
                                if current_colors < best_colors:
                                    best_solution = list(current)
                                    best_colors = current_colors
                                    best_violations_sa = current_violations
            else:
                # No single-use colors, try merging rare colors
                used_colors_list = [c for c in range(MAX_COLOR) if color_freq[c] > 0]
                if len(used_colors_list) >= 2:
                    # Pick a rare color
                    rare_color = min(used_colors_list, key=lambda c: color_freq[c])
                    edges_with_rare = [e for e in range(NUM_EDGES) if current[e] == rare_color]
                    if edges_with_rare:
                        e_idx = random.choice(edges_with_rare)
                        target_colors = [c for c in used_colors_list if c != rare_color]
                        if target_colors:
                            new_c = random.choice(target_colors)
                            new_violations, valid, changes = _evaluate_move(current, current_k4_status, current_violations, e_idx, new_c)
                            
                            if valid and new_violations == 0:
                                current[e_idx] = new_c
                                color_freq[rare_color] -= 1
                                color_freq[new_c] += 1
                                for k_idx, new_status in changes:
                                    current_k4_status[k_idx] = new_status
                                
                                # Recalculate colors
                                current_colors = sum(1 for f in color_freq if f > 0)
                                if current_colors < best_colors:
                                    best_solution = list(current)
                                    best_colors = current_colors
                                    best_violations_sa = current_violations
        else:
            # Phase 2: Reduce violations using simulated annealing
            # Pick a violated K4 and change one of its edges
            violated_k4s = [k for k in range(NUM_K4) if current_k4_status[k]]
            if not violated_k4s:
                current_violations = 0
                continue
                
            k_idx = random.choice(violated_k4s)
            edges = K4_EDGES[k_idx]
            e_idx = random.choice(edges)
            old_c = current[e_idx]
            
            # Try a new color
            new_c = random.randint(0, 25)
            if new_c == old_c:
                continue
            
            new_violations, valid, changes = _evaluate_move(current, current_k4_status, current_violations, e_idx, new_c)
            
            if valid:
                delta_v = new_violations - current_violations
                
                # Accept if better, or with probability exp(-delta/T) if worse
                if delta_v <= 0 or random.random() < math.exp(-delta_v / max(T, 0.001)):
                    # Apply move
                    current[e_idx] = new_c
                    color_freq[old_c] -= 1
                    if new_c < MAX_COLOR:
                        color_freq[new_c] += 1
                    current_violations = new_violations
                    for k_idx2, new_status in changes:
                        current_k4_status[k_idx2] = new_status
                    
                    # Update best if no violations
                    if current_violations == 0:
                        current_colors = sum(1 for f in color_freq if f > 0)
                        if current_colors < best_colors:
                            best_solution = list(current)
                            best_colors = current_colors
                            best_violations_sa = current_violations
        
        # Cool down
        T *= cooling_rate
    
    # Final polish: try to reduce colors further
    if best_violations_sa == 0:
        current = list(best_solution)
        current_k4_status = [False] * NUM_K4
        current_violations = 0
        color_freq = [0] * MAX_COLOR
        for c in current:
            if c < MAX_COLOR:
                color_freq[c] += 1
        current_colors = sum(1 for f in color_freq if f > 0)
        
        # Aggressive color reduction
        for _ in range(5000):
            if time.time() > deadline - 1:
                break
            
            single_use_colors = [c for c in range(MAX_COLOR) if color_freq[c] == 1]
            if single_use_colors:
                old_c = random.choice(single_use_colors)
                edges_with_old = [e for e in range(NUM_EDGES) if current[e] == old_c]
                if edges_with_old:
                    e_idx = random.choice(edges_with_old)
                    used_colors = [c for c in range(MAX_COLOR) if color_freq[c] > 0 and c != old_c]
                    if used_colors:
                        for new_c in used_colors:
                            new_violations, valid, changes = _evaluate_move(current, current_k4_status, 0, e_idx, new_c)
                            if valid and new_violations == 0:
                                current[e_idx] = new_c
                                color_freq[old_c] -= 1
                                color_freq[new_c] += 1
                                for k_idx, new_status in changes:
                                    current_k4_status[k_idx] = new_status
                                current_colors = sum(1 for f in color_freq if f > 0)
                                break
        
        if current_violations == 0 and current_colors < best_colors:
            best_solution = list(current)
            best_colors = current_colors
    
    # Return the best solution found
    if best_violations_sa == 0:
        return best_solution
    else:
        # Return a valid solution if we found one, otherwise the best we have
        if current_violations == 0:
            return current
        else:
            # Fallback to a known valid coloring
            return [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]

def construct():
    return _construct()
# EVOLVE-BLOCK-END
