"""Improved program: faster local search with color-reduction phase."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 115  # safety margin before 120s limit
    n = 16
    num_edges = n * (n - 1) // 2  # 120
    
    # Precompute edge indices and K4 structures
    edge_to_idx = {}
    edge_idx_list = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_to_idx[(i, j)] = len(edge_idx_list)
            edge_idx_list.append((i, j))
    
    k4s = list(combinations(range(n), 4))
    num_k4s = len(k4s)
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
    
    # Known valid 17-color solution
    best_result = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_colors = 17
    
    def setup_state(colors):
        """Precompute K4 color counts and violations."""
        k4_color_counts = [{} for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
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
    
    def apply_change(colors, k4_color_counts, k4_distinct, violations, e, old_c, new_c):
        """Apply a color change and update all state."""
        colors[e] = new_c
        for k_idx in edge_k4s[e]:
            k_counts = k4_color_counts[k_idx]
            k_dist = k4_distinct[k_idx]
            
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
                
            k4_distinct[k_idx] = new_dist
            
            if k_idx in violations and new_dist >= 5:
                violations.remove(k_idx)
            elif k_idx not in violations and new_dist < 5:
                violations.append(k_idx)
    
    def try_move(colors, k4_color_counts, k4_distinct, violations, num_c):
        """Try to fix a violated K4 by changing one edge color."""
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
            
            # Only change if old_c appears more than once in this K4
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
                
                # Simulate the change
                cnt_old = k_counts.get(old_c, 0)
                if cnt_old == 1:
                    new_dist = k_dist - 1
                else:
                    new_dist = k_dist
                    
                cnt_new = k_counts.get(new_c, 0)
                if cnt_new == 0:
                    new_dist += 1
                    
                is_violated = new_dist < 5
                
                if was_violated and not is_violated:
                    delta_violations -= 1
                elif not was_violated and is_violated:
                    delta_violations += 1
                    
            if delta_violations < 0:
                apply_change(colors, k4_color_counts, k4_distinct, violations, e, old_c, new_c)
                return True
            elif delta_violations == 0:
                # Accept neutral moves occasionally to escape local minima
                if random.random() < 0.15:
                    apply_change(colors, k4_color_counts, k4_distinct, violations, e, old_c, new_c)
                    return True
                    
        return False
    
    def reduce_colors(colors, k4_color_counts, k4_distinct, violations, num_c, max_iters=20000):
        """Try to reduce the number of colors used while maintaining validity."""
        if violations:
            return False
            
        used_colors = set(colors)
        current_num_c = len(used_colors)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            if current_num_c <= 13:
                break
                
            # Try to eliminate each used color
            for c in list(used_colors):
                if time.time() > deadline - 5:
                    break
                    
                # Find edges with color c
                edges_with_c = [e for e in range(num_edges) if colors[e] == c]
                
                # Try to recolor each edge to a different used color
                success = True
                for e in edges_with_c:
                    # Try each other used color
                    other_colors = [oc for oc in used_colors if oc != c]
                    random.shuffle(other_colors)
                    
                    found = False
                    for new_c in other_colors:
                        # Check if this change would create violations
                        affected_k4s = edge_k4s[e]
                        would_violate = False
                        
                        for k_idx in affected_k4s:
                            k_counts = k4_color_counts[k_idx]
                            k_dist = k4_distinct[k_idx]
                            
                            cnt_old = k_counts.get(c, 0)
                            if cnt_old == 1:
                                new_dist = k_dist - 1
                            else:
                                new_dist = k_dist
                                
                            cnt_new = k_counts.get(new_c, 0)
                            if cnt_new == 0:
                                new_dist += 1
                                
                            if new_dist < 5:
                                would_violate = True
                                break
                        
                        if not would_violate:
                            # Apply the change
                            apply_change(colors, k4_color_counts, k4_distinct, violations, e, c, new_c)
                            used_colors.discard(c)
                            used_colors.add(new_c)
                            found = True
                            break
                    
                    if not found:
                        success = False
                        break
                
                if not success:
                    continue
                    
                if c in used_colors:
                    used_colors.discard(c)
                    current_num_c = len(used_colors)
        
        return True
    
    # Try local search with different numbers of colors
    for num_c in [13, 14, 15]:
        if time.time() > deadline - 10:
            break
        
        # Multiple restarts
        for restart in range(50):
            if time.time() > deadline - 10:
                break
            
            # Initialize with structured pattern for better convergence
            if restart % 3 == 0:
                # Modular pattern
                colors = [(i * 3 + j) % num_c for i in range(16) for j in range(i + 1, 16)]
            elif restart % 3 == 1:
                # XOR-like pattern
                colors = [(i ^ j) % num_c for i in range(16) for j in range(i + 1, 16)]
            else:
                # Random
                colors = [random.randint(0, num_c - 1) for _ in range(num_edges)]
            
            # Local search: try to fix violations
            k4_color_counts, k4_distinct, violations = setup_state(colors)
            
            for iteration in range(80000):
                if time.time() > deadline - 10:
                    break
                
                if not violations:
                    break
                    
                success = try_move(colors, k4_color_counts, k4_distinct, violations, num_c)
                
                # If we're stuck, try a random perturbation
                if not success and random.random() < 0.08:
                    e = random.randint(0, num_edges - 1)
                    old_c = colors[e]
                    new_c = random.randint(0, num_c - 1)
                    apply_change(colors, k4_color_counts, k4_distinct, violations, e, old_c, new_c)
            
            if not violations:
                # Try to reduce colors further
                reduce_colors(colors, k4_color_counts, k4_distinct, violations, num_c)
                
                if not violations:
                    used = len(set(colors))
                    if used < best_colors:
                        best_colors = used
                        best_result = colors[:]
                    break
    
    # Final fallback: return the known valid 17-color solution if nothing better
    if best_colors > 17:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    return best_result
# EVOLVE-BLOCK-END
