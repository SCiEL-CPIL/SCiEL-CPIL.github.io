"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
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
    
    num_k4s = len(k4s)
    
    # Start with the known valid 17-color solution
    best_result = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_colors = 17
    
    def calculate_violations(colors):
        """Calculate number of violations and list of violating K4s."""
        k4_distinct = np.zeros(num_k4s, dtype=np.int8)
        violating_k4s = []
        
        for k4_idx, edges in enumerate(k4_edges):
            # Use a set to count distinct colors
            distinct = len(set(colors[e] for e in edges))
            k4_distinct[k4_idx] = distinct
            if distinct < 5:
                violating_k4s.append(k4_idx)
                
        return k4_distinct, violating_k4s

    def get_delta_violations(colors, e, new_c, k4_distinct):
        """Calculate change in number of violations if edge e changes to new_c."""
        old_c = colors[e]
        if old_c == new_c:
            return 0
        
        delta = 0
        for k_idx in edge_k4s[e]:
            edges = k4_edges[k_idx]
            # Count distinct colors before
            # We can't just use k4_distinct[k_idx] directly for the 'after' state easily without recounting
            # But we can simulate the set change
            
            # Get current colors in this K4
            current_colors = [colors[ed] for ed in edges]
            old_set = set(current_colors)
            
            # Modify
            # Find index of e in edges
            # This is slow. Let's optimize.
            pass
        return 0 # Placeholder, will implement faster version

    # Optimized state management
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

    def apply_change(colors, e, new_c, k4_color_counts, k4_distinct, violations):
        """Apply a color change to edge e and update state."""
        old_c = colors[e]
        if old_c == new_c:
            return
        
        colors[e] = new_c
        for k_idx in edge_k4s[e]:
            k_counts = k4_color_counts[k_idx]
            k_dist = k4_distinct[k_idx]
            
            # Update counts
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
            
            # Update violations list
            # Using a set for violations would be faster for lookup, but list is fine if we rebuild or use a boolean array
            # For simplicity and speed in small lists, we'll just check and update
            if k_idx in violations and new_dist >= 5:
                violations.remove(k_idx)
            elif k_idx not in violations and new_dist < 5:
                violations.append(k_idx)

    def min_conflicts_search(colors, num_c):
        """Min-conflicts heuristic to find a valid coloring."""
        k4_color_counts, k4_distinct, violations = setup_state(colors)
        
        # If already valid
        if not violations:
            return True, colors
        
        max_iterations = 20000
        for i in range(max_iterations):
            if time.time() > deadline - 2:
                return False, colors
                
            if not violations:
                return True, colors
                
            # Pick a random violated K4
            v_idx = random.choice(violations)
            v_edges = k4_edges[v_idx]
            
            # Pick a random edge in this K4
            e = random.choice(v_edges)
            
            # Try all colors for this edge to find the best one
            best_new_c = None
            best_delta = 0 # We want to minimize violations, so delta < 0 is good
            # Actually, let's just pick a color that reduces violations the most
            # Or simply pick a random color if we are stuck
            
            # Evaluate all num_c colors
            min_delta = float('inf')
            best_c = None
            
            for new_c in range(num_c):
                if new_c == colors[e]:
                    continue
                
                delta = 0
                for k_idx in edge_k4s[e]:
                    k_counts = k4_color_counts[k_idx]
                    k_dist = k4_distinct[k_idx]
                    was_violated = k_dist < 5
                    
                    old_c = colors[e]
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
                        delta -= 1
                    elif not was_violated and is_violated:
                        delta += 1
                        
                if delta < min_delta:
                    min_delta = delta
                    best_c = new_c
                    if min_delta < 0:
                        break # Found an improving move, take it immediately
            
            if best_c is not None:
                apply_change(colors, e, best_c, k4_color_counts, k4_distinct, violations)
            else:
                # No improving move found, pick a random move to escape local minimum
                new_c = random.randint(0, num_c - 1)
                apply_change(colors, e, new_c, k4_color_counts, k4_distinct, violations)
                
        return len(violations) == 0, colors

    # Try local search with different numbers of colors
    for num_c in [13, 14, 15]:
        if time.time() > deadline - 5:
            break
        
        # Multiple restarts
        for restart in range(100):
            if time.time() > deadline - 5:
                break
            
            # Initialize randomly
            colors = [random.randint(0, num_c - 1) for _ in range(num_edges)]
            
            # Run min-conflicts search
            is_valid, final_colors = min_conflicts_search(colors, num_c)
            
            if is_valid:
                used = len(set(final_colors))
                if used < best_colors:
                    best_colors = used
                    best_result = final_colors[:]
                break
    
    # Final fallback: return the known valid 17-color solution if nothing better
    if best_colors > 17:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    return best_result
# EVOLVE-BLOCK-END
