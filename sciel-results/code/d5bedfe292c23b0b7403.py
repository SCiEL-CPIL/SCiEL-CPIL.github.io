import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 16
    # Generate all edges
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)  # 120
    
    # Precompute K4s and their edge indices
    # Map (i,j) to edge index
    edge_idx_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx_map[(i, j)] = idx
            idx += 1
            
    k4s = []
    k4_edge_indices = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    e = [
                        edge_idx_map[(a, b)],
                        edge_idx_map[(a, c)],
                        edge_idx_map[(a, d)],
                        edge_idx_map[(b, c)],
                        edge_idx_map[(b, d)],
                        edge_idx_map[(c, d)]
                    ]
                    k4s.append((a, b, c, d))
                    k4_edge_indices.append(e)
                    
    num_k4s = len(k4_edge_indices) # 1820
    
    # Precompute edge_to_k4s for fast updates
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k_idx, edge_idxs in enumerate(k4_edge_indices):
        for e_idx in edge_idxs:
            edge_to_k4s[e_idx].append(k_idx)

    # Helper to check if a K4 is valid
    def is_violation(colors, k4_edges):
        c = [colors[e] for e in k4_edges]
        return len(set(c)) < 5

    def count_violations(colors):
        count = 0
        for e in k4_edge_indices:
            if is_violation(colors, e):
                count += 1
        return count

    def get_used_colors(colors):
        return set(colors.tolist()) if isinstance(colors, np.ndarray) else set(colors)
    
    def try_merge_color(colors, target_color, source_color):
        # Change all edges with source_color to target_color
        edges_to_change = [i for i in range(num_edges) if colors[i] == source_color]
        
        if not edges_to_change:
            return True, colors
        
        # Affected K4s
        affected_k4s = set()
        for e in edges_to_change:
            for k_idx in edge_to_k4s[e]:
                affected_k4s.add(k_idx)
        
        # Apply change temporarily
        for e in edges_to_change:
            colors[e] = target_color
            
        # Check validity of affected K4s
        valid = True
        for k_idx in affected_k4s:
            if is_violation(colors, k4_edge_indices[k_idx]):
                valid = False
                break
                
        if not valid:
            # Revert
            for e in edges_to_change:
                colors[e] = source_color
            return False, colors
        else:
            return True, colors

    # Try multiple initial constructions and pick the best
    initial_candidates = []
    
    # Construction 1: (i + j) % 17
    init1 = [(i + j) % 17 for i, j in edges]
    initial_candidates.append(init1)
    
    # Construction 2: (i - j) % 16 + 1 (uses 16 colors)
    init2 = [((i - j) % 16) + 1 for i, j in edges]
    initial_candidates.append(init2)
    
    # Construction 3: (i * 2 + j) % 15 + 1 (uses 15 colors)
    init3 = [((i * 2 + j) % 15) + 1 for i, j in edges]
    initial_candidates.append(init3)
    
    # Construction 4: (i + 2*j) % 14 + 1 (uses 14 colors)
    init4 = [((i + 2*j) % 14) + 1 for i, j in edges]
    initial_candidates.append(init4)
    
    # Construction 5: Random with 15 colors
    random.seed(42)
    init5 = [random.randint(0, 14) for _ in range(num_edges)]
    initial_candidates.append(init5)
    
    # Construction 6: Random with 14 colors
    init6 = [random.randint(0, 13) for _ in range(num_edges)]
    initial_candidates.append(init6)
    
    # Pick the valid candidate with fewest colors, or best if none valid
    best_init = None
    best_init_colors = None
    best_init_count = float('inf')
    
    for cand in initial_candidates:
        colors = np.array(cand)
        violations = count_violations(colors)
        num_colors = len(set(colors))
        
        if violations == 0:
            if num_colors < best_init_count:
                best_init_count = num_colors
                best_init_colors = colors.copy()
        elif best_init is None:
            # Fallback to first candidate
            best_init = colors.copy()
            best_init_count = num_colors
    
    if best_init_colors is not None:
        current_colors = best_init_colors.copy()
    else:
        current_colors = np.array(initial_candidates[0])
    
    best_colors = current_colors.copy()
    best_num_colors = len(set(best_colors))
    
    # Phase 1: Systematic color merging - try to merge colors in order of frequency
    def systematic_merge(colors):
        colors = colors.copy()
        improved = True
        while improved:
            improved = False
            used = list(get_used_colors(colors))
            if len(used) <= 13:
                break
            
            # Sort by frequency (ascending) - try to eliminate rare colors first
            freq = Counter(colors.tolist())
            used_sorted = sorted(used, key=lambda c: freq[c])
            
            for source_color in used_sorted:
                if source_color not in get_used_colors(colors):
                    continue
                for target_color in used_sorted:
                    if target_color == source_color:
                        continue
                    if target_color not in get_used_colors(colors):
                        continue
                    success, new_colors = try_merge_color(colors, target_color, source_color)
                    if success:
                        colors = new_colors
                        improved = True
                        break
                if improved:
                    break
        
        return colors
    
    # Run systematic merge
    current_colors = systematic_merge(current_colors)
    current_num_colors = len(set(current_colors))
    if current_num_colors < best_num_colors:
        best_colors = current_colors.copy()
        best_num_colors = current_num_colors
    
    # Phase 2: Random local search with restarts
    max_iterations = 200000
    iter_count = 0
    restart_count = 0
    max_restarts = 5
    
    while time.time() < deadline and iter_count < max_iterations:
        iter_count += 1
        
        used_colors = list(get_used_colors(current_colors))
        if len(used_colors) <= 13:
            break
            
        # Occasionally restart from a random state
        if restart_count < max_restarts and iter_count % 20000 == 0:
            restart_count += 1
            # Restart from best found so far with some randomization
            current_colors = best_colors.copy()
            # Randomly change some edges
            for _ in range(10):
                e_idx = random.randint(0, num_edges - 1)
                new_c = random.randint(0, best_num_colors - 1)
                old_c = current_colors[e_idx]
                if new_c != old_c:
                    affected = set(edge_to_k4s[e_idx])
                    current_colors[e_idx] = new_c
                    valid = True
                    for k_idx in affected:
                        if is_violation(current_colors, k4_edge_indices[k_idx]):
                            valid = False
                            break
                    if not valid:
                        current_colors[e_idx] = old_c
            continue
        
        # Pick a random color to eliminate
        source_color = random.choice(used_colors)
        other_colors = [c for c in used_colors if c != source_color]
        if not other_colors:
            break
        target_color = random.choice(other_colors)
        
        # Try merge
        success, new_colors = try_merge_color(current_colors, target_color, source_color)
        
        if success:
            current_colors = new_colors
            new_num_colors = len(set(current_colors))
            if new_num_colors < best_num_colors:
                best_colors = current_colors.copy()
                best_num_colors = new_num_colors
        else:
            # Failed to merge. Try a random local move to escape local minima
            if random.random() < 0.3:
                e_idx = random.randint(0, num_edges - 1)
                old_c = current_colors[e_idx]
                new_c = random.choice(list(get_used_colors(current_colors)))
                
                if new_c != old_c:
                    affected = set(edge_to_k4s[e_idx])
                    current_colors[e_idx] = new_c
                    valid = True
                    for k_idx in affected:
                        if is_violation(current_colors, k4_edge_indices[k_idx]):
                            valid = False
                            break
                    if not valid:
                        current_colors[e_idx] = old_c
                    else:
                        # Accept and check if we improved
                        pass
    
    # Phase 3: Final polish - try systematic merge on best solution
    polished = systematic_merge(best_colors)
    polished_colors = len(set(polished))
    if polished_colors < best_num_colors:
        best_colors = polished
        best_num_colors = polished_colors
    
    # Final check: ensure best_colors is valid
    if count_violations(best_colors) > 0:
        # Fallback to initial
        best_colors = np.array(initial_candidates[0])
        
    return best_colors.tolist()
