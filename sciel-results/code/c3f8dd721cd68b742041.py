"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n = 16
    edge_idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edge_idx)
    
    # Map edge (i,j) to index
    edge_to_idx = {}
    for idx, (i, j) in enumerate(edge_idx):
        edge_to_idx[(i, j)] = idx
        edge_to_idx[(j, i)] = idx
        
    # Precompute all K4s and their edge indices
    k4s = list(combinations(range(n), 4))
    k4_edges = []
    for k4 in k4s:
        edges = []
        for a, b in combinations(k4, 2):
            edges.append(edge_to_idx[(a, b)])
        k4_edges.append(edges)
        
    # For each edge, which K4s does it belong to?
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_k4s[e].append(k4_idx)

    def count_colors_in_k4(colors, edge_indices):
        return len(set(colors[e] for e in edge_indices))

    def is_valid(colors):
        for edges in k4_edges:
            if count_colors_in_k4(colors, edges) < 5:
                return False
        return True

    best_result = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_colors = 17

    def setup_state(colors):
        k4_color_counts = [{} for _ in range(len(k4s))]
        k4_distinct = [0] * len(k4s)
        violations = []
        
        for k4_idx, edges in enumerate(k4_edges):
            counts = {}
            for e in edges:
                c = colors[e]
                counts[c] = counts.get(c, 0) + 1
            k4_color_counts[k4_idx] = counts
            k4_distinct[k4_idx] = len(counts)
            if k4_distinct[k4_idx] < 5:
                violations.append(k4_idx)
                
        return k4_color_counts, k4_distinct, violations

    def try_move_safe(colors, k4_color_counts, k4_distinct, violations, num_c):
        if not violations:
            return True
        
        v_idx = random.choice(violations)
        v_k4_edges = k4_edges[v_idx]
        v_counts = k4_color_counts[v_idx]
        
        # Find colors missing in this K4
        missing_colors = [c for c in range(num_c) if c not in v_counts]
        if not missing_colors:
            return False
            
        edges_to_try = v_k4_edges[:]
        random.shuffle(edges_to_try)
        
        for e in edges_to_try:
            old_c = colors[e]
            count_old = v_counts.get(old_c, 0)
            # Only change if old_c appears more than once in this K4,
            # so that we don't lose a distinct color while gaining a new one (net 0)
            # We want net +1 distinct colors.
            if count_old == 1:
                continue
                
            new_c = random.choice(missing_colors)
            if new_c == old_c:
                continue
                
            affected_k4s = edge_k4s[e]
            delta_violations = 0
            
            # Check all affected K4s
            for k_idx in affected_k4s:
                k_counts = k4_color_counts[k_idx]
                k_dist = k4_distinct[k_idx]
                was_violated = k_dist < 5
                
                # Simulate
                cnt_old = k_counts.get(old_c, 0)
                if cnt_old == 1:
                    new_dist = k_dist - 1
                else:
                    new_dist = k_dist
                    
                cnt_new = k_counts.get(new_c, 0)
                if cnt_new == 0:
                    new_dist += 1
                # else new_dist unchanged
                
                is_violated = new_dist < 5
                
                if was_violated and not is_violated:
                    delta_violations -= 1
                elif not was_violated and is_violated:
                    delta_violations += 1
                    
            if delta_violations < 0:
                # Apply
                colors[e] = new_c
                for k_idx in affected_k4s:
                    k_counts = k4_color_counts[k_idx]
                    k_dist = k4_distinct[k_idx]
                    was_violated = k_dist < 5
                    
                    cnt_old = k_counts.get(old_c, 0)
                    if cnt_old == 1:
                        del k_counts[old_c]
                        new_dist = k_dist - 1
                    else:
                        k_counts[old_c] = cnt_old - 1
                        new_dist = k_dist
                        
                    cnt_new = k_counts.get(new_c, 0)
                    if cnt_new == 0:
                        k_counts[new_c] = 1
                        new_dist += 1
                    else:
                        k_counts[new_c] = cnt_new + 1
                        
                    is_violated = new_dist < 5
                    k4_distinct[k_idx] = new_dist
                    
                    if was_violated and not is_violated:
                        try:
                            violations.remove(k_idx)
                        except ValueError:
                            pass
                    elif not was_violated and is_violated:
                        violations.append(k_idx)
                return True
            else:
                continue
                
        return False

    # Search Loop
    for num_c in [13, 14, 15, 16]:
        if time.time() > deadline - 2:
            break
            
        for restart in range(50):
            if time.time() > deadline - 2:
                break
                
            # Initialize
            colors = [random.randint(0, num_c - 1) for _ in range(num_edges)]
            
            # Setup state
            k4_color_counts, k4_distinct, violations = setup_state(colors)
            
            # Local Search
            for step in range(100000):
                if time.time() > deadline - 2:
                    break
                    
                if not violations:
                    # Valid!
                    used = len(set(colors))
                    if used < best_colors:
                        best_colors = used
                        best_result = colors[:]
                    break
                    
                if not try_move_safe(colors, k4_color_counts, k4_distinct, violations, num_c):
                    # Stuck
                    break
                         
    if best_colors > 17:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        
    return best_result
# EVOLVE-BLOCK-END
