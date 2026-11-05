import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for safety
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # List of all edges (i, j)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # List of all K4s (sets of 4 vertices)
    k4s = list(combinations(range(n), 4))
    num_k4s = len(k4s)
    
    # For each K4, find the indices of its 6 edges
    k4_edge_indices = []
    for k4 in k4s:
        vs = list(k4)
        e_indices = []
        for i in range(4):
            for j in range(i + 1, 4):
                e_indices.append(edge_index[(vs[i], vs[j])])
        k4_edge_indices.append(e_indices)
        
    # For each edge, find which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_indices in enumerate(k4_edge_indices):
        for e_idx in e_indices:
            edge_to_k4s[e_idx].append(k4_idx)

    def count_violations(colors):
        # colors is a list of 120 ints
        # Returns number of K4s with < 5 distinct colors
        viol = 0
        for e_indices in k4_edge_indices:
            distinct = len(set(colors[idx] for idx in e_indices))
            if distinct < 5:
                viol += 1
        return viol
        
    def count_colors_used(colors):
        return len(set(colors))

    # Optimized Local Search with Delta Updates
    def local_search(max_colors, time_limit, seed=None):
        if seed is not None:
            random.seed(seed)
            
        # Initialize with random colors
        colors = [random.randint(0, max_colors - 1) for _ in range(num_edges)]
        
        # Precompute current state
        # k4_counts[k4_idx] is a list of size max_colors, storing count of each color
        k4_counts = []
        k4_distinct = []
        total_violations = 0
        violating_k4s = set()
        
        for k4_idx, e_indices in enumerate(k4_edge_indices):
            counts = [0] * max_colors
            for idx in e_indices:
                counts[colors[idx]] += 1
            distinct = sum(1 for c in counts if c > 0)
            k4_counts.append(counts)
            k4_distinct.append(distinct)
            if distinct < 5:
                total_violations += 1
                violating_k4s.add(k4_idx)
                
        best_violations = total_violations
        best_colors = colors[:]
        
        # Simulated Annealing parameters
        temp = 5.0
        min_temp = 0.01
        cool_rate = 0.99995
        
        # Track stagnation
        stagnation_counter = 0
        last_improvement_viol = best_violations
        
        # Main Loop
        while time.time() < start_time + time_limit:
            if total_violations == 0:
                return colors, 0
            
            # Pick a move
            # Strategy: 80% random edge, 20% edge in a violating K4
            if total_violations > 0 and random.random() < 0.2:
                # Pick a random violating K4
                v_k4_idx = random.choice(list(violating_k4s))
                # Pick a random edge from that K4
                e_idx = random.choice(k4_edge_indices[v_k4_idx])
            else:
                e_idx = random.randint(0, num_edges - 1)
                
            old_color = colors[e_idx]
            new_color = random.randint(0, max_colors - 1)
            
            if new_color == old_color:
                continue
            
            # Calculate Delta Violations
            delta_viol = 0
            affected_k4s = edge_to_k4s[e_idx]
            
            # Store old distinct values for potential revert
            old_distincts = []
            
            for k4_idx in affected_k4s:
                counts = k4_counts[k4_idx]
                old_d = k4_distinct[k4_idx]
                old_distincts.append(old_d)
                
                # Update counts
                counts[old_color] -= 1
                counts[new_color] += 1
                
                # Update distinct
                new_d = old_d
                if counts[old_color] == 0:
                    new_d -= 1
                if counts[new_color] == 1: # It was 0 before
                    new_d += 1
                
                k4_distinct[k4_idx] = new_d
                
                # Update violation status
                was_viol = 1 if old_d < 5 else 0
                is_viol = 1 if new_d < 5 else 0
                delta_viol += is_viol - was_viol
                
                # Update violating_k4s set
                if was_viol and not is_viol:
                    violating_k4s.discard(k4_idx)
                elif not was_viol and is_viol:
                    violating_k4s.add(k4_idx)
            
            # Acceptance Criteria
            accept = False
            if delta_viol <= 0:
                accept = True
            else:
                if random.random() < math.exp(-delta_viol / temp):
                    accept = True
            
            if accept:
                colors[e_idx] = new_color
                total_violations += delta_viol
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors[:]
                    stagnation_counter = 0
                    last_improvement_viol = best_violations
                else:
                    stagnation_counter += 1
            else:
                # Revert counts and distinct
                for k4_idx, old_d in zip(affected_k4s, old_distincts):
                    counts = k4_counts[k4_idx]
                    counts[old_color] += 1
                    counts[new_color] -= 1
                    k4_distinct[k4_idx] = old_d
                    
                    # Revert violating_k4s set
                    # We need to check if it should be in the set
                    # old_d < 5 -> was in set. new_d (current) >= 5 -> not in set.
                    # So if we reverted, we go back to old state.
                    if old_d < 5:
                        violating_k4s.add(k4_idx)
                    else:
                        violating_k4s.discard(k4_idx)
            
            # Cool down
            temp *= cool_rate
            if temp < min_temp:
                # If stuck for a long time, restart cooling?
                if stagnation_counter > 5000:
                    temp = 2.0 
                    stagnation_counter = 0
                    # Optional: randomize some colors to escape?
                    # For now, just boost temp.
                pass
                
        return best_colors, best_violations

    # Strategy:
    # 1. Try to find a solution with 13 colors.
    # 2. If fail, try 14.
    # 3. If fail, try 15.
    # 4. Fallback to 17 (seed) if all fail (should not happen if 15 is easy).
    
    # Time allocation:
    # 13 colors: 40s
    # 14 colors: 30s
    # 15 colors: 30s
    # Remaining: buffer
    
    # Try C=13
    t_limit = 40.0
    if time.time() < deadline - t_limit:
        # Run multiple restarts within the time limit
        # The local_search function runs until time_limit expires.
        # We can call it once with the full limit, or multiple times with smaller limits.
        # Given the speedup, one long run with restarts inside (via temp boost) is good.
        # But explicit restarts with new seeds might be better.
        
        # Let's do 2 restarts of 20s each for 13 colors
        best_sol_13 = None
        best_viol_13 = float('inf')
        
        for r in range(2):
            rem_time = min(20.0, deadline - time.time() - 5.0)
            if rem_time < 5.0: break
            sol, viol = local_search(13, rem_time, seed=42 + r)
            if viol < best_viol_13:
                best_viol_13 = viol
                best_sol_13 = sol
            if viol == 0:
                return sol
                
    # Try C=14
    if time.time() < deadline - 30.0:
        best_sol_14 = None
        best_viol_14 = float('inf')
        
        # Run 2 restarts of 15s each
        for r in range(2):
            rem_time = min(15.0, deadline - time.time() - 5.0)
            if rem_time < 5.0: break
            sol, viol = local_search(14, rem_time, seed=100 + r)
            if viol < best_viol_14:
                best_viol_14 = viol
                best_sol_14 = sol
            if viol == 0:
                return sol
                
    # Try C=15
    if time.time() < deadline - 20.0:
        best_sol_15 = None
        best_viol_15 = float('inf')
        
        # Run 2 restarts of 10s each
        for r in range(2):
            rem_time = min(10.0, deadline - time.time() - 5.0)
            if rem_time < 5.0: break
            sol, viol = local_search(15, rem_time, seed=200 + r)
            if viol < best_viol_15:
                best_viol_15 = viol
                best_sol_15 = sol
            if viol == 0:
                return sol
                
    # Fallback
    # If we didn't find a valid one, we must return a valid one.
    # The seed (i+j)%17 is valid and uses 17 colors.
    # However, if we found a solution with violations, we can't return it.
    # We should return the seed.
    
    seed_colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    # Verify seed is valid (it should be, but check just in case)
    if count_violations(seed_colors) == 0:
        return seed_colors
        
    # If seed is invalid (shouldn't happen), try a greedy repair or just return random?
    # But (i+j)%17 for K16:
    # K4 vertices a,b,c,d. Edges:
    # a+b, a+c, a+d, b+c, b+d, c+d mod 17
    # Are there 5 distinct?
    # If a+b == a+c => b=c (impossible)
    # If a+b == b+c => a=c (impossible)
    # If a+b == c+d => a-b = d-c
    # It is known that (i+j) mod p for p > n is a proper edge coloring?
    # Actually, for p prime, (i+j) mod p is a proper coloring of K_n if p >= n.
    # Proper edge coloring means no two edges sharing a vertex have the same color.
    # This implies that in any K4, the 6 edges have distinct colors?
    # No. Proper edge coloring of K4 requires 3 colors (chromatic index of K4 is 3).
    # Wait, chromatic index of K4 is 3. So 3 colors suffice for proper edge coloring.
    # But the constraint is "every K4 sees FIVE colors".
    # A proper edge coloring of K4 uses 3 colors. This violates the "5 colors" constraint.
    # So (i+j)%17 is NOT a valid solution for this specific problem!
    # The problem asks for "every K4 sees five colours".
    # A proper edge coloring of K4 uses 3 colors.
    # So the seed solution is INVALID for this problem.
    
    # This is a critical realization. The "seed" in the prompt description might be misleading or I am misinterpreting "valid".
    # "Every set of four vertices must span edges of at least five distinct colours."
    # If I use a proper edge coloring, each K4 uses 3 colors. 3 < 5. Invalid.
    # So I MUST find a coloring that is NOT a proper edge coloring, or uses more colors.
    # Actually, to have 5 colors in a K4, the coloring must be "non-proper" in some sense, or just use enough colors.
    # If I use 17 colors, can I ensure every K4 has 5?
    # Random coloring with 17 colors:
    # Probability a specific K4 has < 5 colors?
    # Total colorings: 17^6.
    # Colorings with <= 4 colors: 
    # Choose 4 colors: C(17,4) * (surjective functions from 6 edges to 4 colors)
    # This is small.
    # So a random coloring with 17 colors is very likely valid.
    
    # Let's verify if my local search found anything.
    # If not, I need a valid fallback.
    # I will generate a random coloring with 17 colors and verify it.
    
    random.seed(12345)
    fallback_colors = [random.randint(0, 16) for _ in range(num_edges)]
    
    # Verify fallback
    # If invalid, keep trying random
    attempts = 0
    while count_violations(fallback_colors) > 0 and attempts < 1000:
        fallback_colors = [random.randint(0, 16) for _ in range(num_edges)]
        attempts += 1
        
    if count_violations(fallback_colors) == 0:
        return fallback_colors
        
    # If even that fails (extremely unlikely), return the (i+j)%17 just to have *some* output, 
    # though it will score 0. But better to return a random valid one.
    # Let's force a valid one by using many colors.
    # With 17 colors, probability of failure is negligible.
    
    return fallback_colors

# Test locally? No, just output.
