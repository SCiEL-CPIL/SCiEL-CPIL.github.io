import time
import random
import numpy as np
from collections import Counter
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit
    
    n = 16
    # Generate all edges in lexicographic order
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)  # 120
    
    # Map (i,j) to edge index
    edge_idx_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx_map[(i, j)] = idx
            idx += 1
            
    # Precompute K4s and their edge indices
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
                    
    num_k4s = len(k4_edge_indices)  # 1820
    
    # Precompute edge_to_k4s for fast updates
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k_idx, edge_idxs in enumerate(k4_edge_indices):
        for e_idx in edge_idxs:
            edge_to_k4s[e_idx].append(k_idx)

    def count_violations(colors):
        """Count K4s with fewer than 5 distinct colors."""
        count = 0
        for k_idx in range(num_k4s):
            c = [colors[e] for e in k4_edge_indices[k_idx]]
            if len(set(c)) < 5:
                count += 1
        return count

    def is_violation(colors, k4_edges):
        c = [colors[e] for e in k4_edges]
        return len(set(c)) < 5

    def count_violations_fast(colors, k4_indices):
        """Count violations only for given K4 indices."""
        count = 0
        for k_idx in k4_indices:
            c = [colors[e] for e in k4_edge_indices[k_idx]]
            if len(set(c)) < 5:
                count += 1
        return count

    def try_change_edge(colors, edge_idx, new_color):
        """Try changing one edge's color. Returns True if valid, updates colors in place."""
        old_color = colors[edge_idx]
        if old_color == new_color:
            return True
        colors[edge_idx] = new_color
        for k_idx in edge_to_k4s[edge_idx]:
            if is_violation(colors, k4_edge_indices[k_idx]):
                colors[edge_idx] = old_color
                return False
        return True

    def try_merge_color(colors, target_color, source_color):
        """Try to change all edges with source_color to target_color."""
        edges_to_change = [i for i in range(num_edges) if colors[i] == source_color]
        
        if not edges_to_change:
            return True
        
        # Apply change
        for e in edges_to_change:
            colors[e] = target_color
            
        # Check validity of affected K4s
        affected_k4s = set()
        for e in edges_to_change:
            for k_idx in edge_to_k4s[e]:
                affected_k4s.add(k_idx)
                
        for k_idx in affected_k4s:
            if is_violation(colors, k4_edge_indices[k_idx]):
                # Revert
                for e in edges_to_change:
                    colors[e] = source_color
                return False
        return True

    # Generate many initial candidates using various algebraic constructions
    initial_candidates = []
    
    # Construction 1: (i + j) % 17 - known to use 15 colors
    init1 = [(i + j) % 17 for i, j in edges]
    initial_candidates.append(init1)
    
    # Construction 2: (i - j) % 16 + 1
    init2 = [((i - j) % 16) + 1 for i, j in edges]
    initial_candidates.append(init2)
    
    # Construction 3: (i * 2 + j) % 15 + 1
    init3 = [((i * 2 + j) % 15) + 1 for i, j in edges]
    initial_candidates.append(init3)
    
    # Construction 4: (i + 2*j) % 14 + 1
    init4 = [((i + 2*j) % 14) + 1 for i, j in edges]
    initial_candidates.append(init4)
    
    # Construction 5: (i*3 + j) % 13 + 1
    init5 = [((i * 3 + j) % 13) + 1 for i, j in edges]
    initial_candidates.append(init5)
    
    # Construction 6: (i + 3*j) % 13 + 1
    init6 = [((i + 3*j) % 13) + 1 for i, j in edges]
    initial_candidates.append(init6)
    
    # Construction 7: Multiplicative scheme (i * j) % p for various primes
    for p in [17, 19, 23, 29, 31]:
        init_m = [(i * j) % p for i, j in edges]
        initial_candidates.append(init_m)
    
    # Construction 8: (i * j + i + j) % p
    for p in [17, 19, 23]:
        init_m2 = [(i * j + i + j) % p for i, j in edges]
        initial_candidates.append(init_m2)
    
    # Construction 9: (i^2 + j) % 17
    init9 = [(i*i + j) % 17 for i, j in edges]
    initial_candidates.append(init9)
    
    # Construction 10: (i + j^2) % 17
    init10 = [(i + j*j) % 17 for i, j in edges]
    initial_candidates.append(init10)
    
    # Construction 11: (i * j + i - j) % 17
    init11 = [(i * j + i - j) % 17 for i, j in edges]
    initial_candidates.append(init11)
    
    # Construction 12: (i * j - i + j) % 17
    init12 = [(i * j - i + j) % 17 for i, j in edges]
    initial_candidates.append(init12)
    
    # Random initializations with various color counts
    random.seed(42)
    for nc in [15, 14, 13, 12, 11]:
        for trial in range(3):
            init_r = [random.randint(0, nc - 1) for _ in range(num_edges)]
            initial_candidates.append(init_r)
    
    # Pick the best initial candidate
    best_init_colors = None
    best_init_count = float('inf')
    best_init_violations = float('inf')
    
    for cand in initial_candidates:
        colors = np.array(cand, dtype=np.int32)
        violations = count_violations(colors)
        num_colors = len(set(colors.tolist()))
        
        if violations == 0:
            if num_colors < best_init_count:
                best_init_count = num_colors
                best_init_colors = colors.copy()
                best_init_violations = 0
        elif violations < best_init_violations:
            best_init_violations = violations
            best_init_count = num_colors
            best_init_colors = colors.copy()
    
    if best_init_colors is None:
        best_init_colors = np.array(initial_candidates[0], dtype=np.int32)
    
    current_colors = best_init_colors.copy()
    best_colors = current_colors.copy()
    best_num_colors = len(set(best_colors.tolist()))
    
    # Phase 1: Systematic color merging
    def systematic_merge(colors, time_limit):
        colors = colors.copy()
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            used = list(set(colors.tolist()))
            if len(used) <= 13:
                break
            
            freq = Counter(colors.tolist())
            used_sorted = sorted(used, key=lambda c: freq[c])
            
            for source_color in used_sorted:
                if source_color not in set(colors.tolist()):
                    continue
                for target_color in used_sorted:
                    if target_color == source_color:
                        continue
                    if target_color not in set(colors.tolist()):
                        continue
                    if try_merge_color(colors, target_color, source_color):
                        improved = True
                        break
                if improved:
                    break
        
        return colors

    current_colors = systematic_merge(current_colors, deadline - 10)
    current_num_colors = len(set(current_colors.tolist()))
    if current_num_colors < best_num_colors:
        best_colors = current_colors.copy()
        best_num_colors = current_num_colors

    # Phase 2: Simulated annealing for color reduction
    def simulated_annealing_merge(colors, time_limit):
        """Simulated annealing to reduce number of colors."""
        colors = colors.copy()
        current_temp = 1.0
        cooling_rate = 0.9995
        min_temp = 0.01
        best_colors = colors.copy()
        best_num_colors = len(set(colors.tolist()))
        
        while time.time() < time_limit and current_temp > min_temp:
            used_colors = list(set(colors.tolist()))
            if len(used_colors) <= best_num_colors:
                break
            
            # Try to merge a random color pair
            if len(used_colors) < 2:
                break
                
            source_color = random.choice(used_colors)
            other_colors = [c for c in used_colors if c != source_color]
            if not other_colors:
                break
            target_color = random.choice(other_colors)
            
            # Try the merge
            edges_to_change = [i for i in range(num_edges) if colors[i] == source_color]
            if not edges_to_change:
                current_temp *= cooling_rate
                continue
            
            # Save old state
            old_colors = colors.copy()
            
            # Apply merge
            for e in edges_to_change:
                colors[e] = target_color
            
            # Count violations
            affected_k4s = set()
            for e in edges_to_change:
                for k_idx in edge_to_k4s[e]:
                    affected_k4s.add(k_idx)
            
            num_violations = 0
            for k_idx in affected_k4s:
                if is_violation(colors, k4_edge_indices[k_idx]):
                    num_violations += 1
            
            new_num_colors = len(set(colors.tolist()))
            
            # Accept if better, or with probability based on temperature
            if new_num_colors < best_num_colors:
                best_colors = colors.copy()
                best_num_colors = new_num_colors
            elif num_violations == 0:
                # Valid, accept
                pass
            else:
                # Invalid - accept with probability exp(-delta/T)
                # delta = number of violations
                delta = num_violations
                if delta > 0:
                    prob = np.exp(-delta / (current_temp * 10))
                    if random.random() > prob:
                        # Revert
                        colors = old_colors
                # If accepted invalid state, keep it temporarily
            
            current_temp *= cooling_rate
        
        return best_colors, best_num_colors

    # Run simulated annealing multiple times with different seeds
    for sa_trial in range(5):
        if time.time() > deadline - 10:
            break
        
        random.seed(100 + sa_trial)
        sa_colors, sa_num = simulated_annealing_merge(best_colors, deadline - 10 - sa_trial * 20)
        if sa_num < best_num_colors:
            best_colors = sa_colors
            best_num_colors = sa_num

    # Phase 3: Local search with random moves
    def get_violation_set(colors):
        """Get set of K4 indices that are violations."""
        vs = set()
        for k_idx in range(num_k4s):
            if is_violation(colors, k4_edge_indices[k_idx]):
                vs.add(k_idx)
        return vs

    current_colors = best_colors.copy()
    current_violations = get_violation_set(current_colors)
    
    # Local search: try single edge color changes
    max_iterations = 500000
    iter_count = 0
    stuck_count = 0
    max_stuck = 5000
    
    while time.time() < deadline - 5 and iter_count < max_iterations:
        iter_count += 1
        
        if len(current_violations) == 0:
            # Valid solution - try to reduce colors
            used_colors = list(set(current_colors.tolist()))
            if len(used_colors) <= 13:
                break
                
            # Try to merge a color
            if iter_count % 100 == 0:
                source_color = random.choice(used_colors)
                other_colors = [c for c in used_colors if c != source_color]
                if other_colors:
                    target_color = random.choice(other_colors)
                    if try_merge_color(current_colors, target_color, source_color):
                        new_num = len(set(current_colors.tolist()))
                        if new_num < best_num_colors:
                            best_colors = current_colors.copy()
                            best_num_colors = new_num
                            stuck_count = 0
            else:
                # Try random edge change to explore
                e_idx = random.randint(0, num_edges - 1)
                old_c = current_colors[e_idx]
                new_c = random.randint(0, best_num_colors - 1)
                
                if new_c != old_c:
                    if try_change_edge(current_colors, e_idx, new_c):
                        # Check if this improved things
                        new_violations = get_violation_set(current_colors)
                        if len(new_violations) <= len(current_violations):
                            current_violations = new_violations
                            stuck_count = 0
                        else:
                            # Revert
                            current_colors[e_idx] = old_c
                            current_violations = get_violation_set(current_colors)
                            stuck_count += 1
                    else:
                        stuck_count += 1
        else:
            # Has violations - try to fix them
            if stuck_count > max_stuck:
                # Restart from best
                current_colors = best_colors.copy()
                current_violations = get_violation_set(current_colors)
                stuck_count = 0
                iter_count = 0
                continue
            
            # Pick a random violated K4 and try to fix it
            if current_violations:
                k_idx = random.choice(list(current_violations))
                k4_edges = k4_edge_indices[k_idx]
                
                # Try changing one of the edges in this K4
                e_idx = random.choice(k4_edges)
                old_c = current_colors[e_idx]
                # Try a new color that might help
                new_c = random.randint(0, best_num_colors)
                
                if new_c != old_c:
                    if try_change_edge(current_colors, e_idx, new_c):
                        new_violations = get_violation_set(current_colors)
                        if len(new_violations) < len(current_violations):
                            current_violations = new_violations
                            stuck_count = 0
                            # Update best if no violations
                            if len(new_violations) == 0:
                                new_num = len(set(current_colors.tolist()))
                                if new_num < best_num_colors:
                                    best_colors = current_colors.copy()
                                    best_num_colors = new_num
                        else:
                            current_colors[e_idx] = old_c
                            current_violations = get_violation_set(current_colors)
                            stuck_count += 1
                    else:
                        stuck_count += 1
            else:
                stuck_count += 1

    # Phase 4: Final polish
    if time.time() < deadline - 2:
        polished = systematic_merge(best_colors, deadline - 1)
        polished_colors = len(set(polished.tolist()))
        if polished_colors < best_num_colors:
            best_colors = polished
            best_num_colors = polished_colors

    # Final validation
    if count_violations(best_colors) > 0:
        # Fallback
        for cand in initial_candidates:
            colors = np.array(cand, dtype=np.int32)
            if count_violations(colors) == 0:
                best_colors = colors
                best_num_colors = len(set(best_colors.tolist()))
                break
    
    # Renumber colors to 0..k-1
    unique_colors = sorted(set(best_colors.tolist()))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    result = [color_map[c] for c in best_colors.tolist()]
    
    return result
