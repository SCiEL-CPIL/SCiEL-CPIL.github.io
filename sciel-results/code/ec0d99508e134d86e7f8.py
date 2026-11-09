"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    n = 128
    
    # Precompute all nonconstant 4-term progressions
    # A 4-term progression: a, a+d, a+2d, a+3d where d>0
    # Positions are 0-indexed: 0..127
    progressions = []
    for a in range(n):
        for d in range(1, n):
            a3 = a + 3*d
            if a3 < n:
                progressions.append((a, a+d, a+2*d, a3))
    
    num_prog = len(progressions)
    
    # We want to minimize the number of distinct colors used
    # while ensuring every 4-term progression has at least 3 distinct colors
    
    # Strategy: Try with k colors (k=2,3,4,...) and find the minimum k that works
    # For each k, use a constraint satisfaction / local search approach
    
    best_solution = None
    best_num_colors = n + 1
    
    # Try increasing number of colors
    for num_colors in range(2, 8):
        if time.time() > deadline - 5.0:
            break
        
        # Try multiple random restarts for this number of colors
        found = False
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Initialize randomly
            colors = np.random.randint(0, num_colors, size=n)
            
            # Local search: iteratively fix violations
            # A violation is a 4-term progression with < 3 distinct colors
            # We'll use a simulated annealing / hill climbing approach
            
            def count_violations(c):
                """Count number of 4-term progressions with < 3 distinct colors."""
                violations = 0
                for (a, b, c2, d) in progressions:
                    distinct = len({c[a], c[b], c[c2], c[d]})
                    if distinct < 3:
                        violations += 1
                return violations
            
            def count_distinct_colors(c):
                return len(set(c.tolist()))
            
            # Check initial solution
            init_viol = count_violations(colors)
            if init_viol == 0:
                best_solution = colors.copy()
                best_num_colors = num_colors
                found = True
                break
            
            # Simulated annealing
            current_viol = init_viol
            best_local_viol = current_viol
            best_local_colors = colors.copy()
            
            T = 1.0
            T_min = 0.001
            alpha = 0.995
            max_iter = 5000
            
            for it in range(max_iter):
                if time.time() > deadline - 5.0:
                    break
                
                # Pick a random position to change
                pos = random.randint(0, n-1)
                old_color = colors[pos]
                new_color = random.randint(0, num_colors - 1)
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                
                # Compute delta in violations
                # Only need to check progressions that include position pos
                delta_viol = 0
                for (a, b, c2, d) in progressions:
                    if pos in (a, b, c2, d):
                        # Check before
                        # We need to recompute, which is expensive
                        # Let's just recompute total violations periodically
                        pass
                
                # For efficiency, recompute violations every few steps
                if it % 10 == 0:
                    new_viol = count_violations(colors)
                else:
                    new_viol = current_viol + random.randint(-2, 2)  # approximation
                
                delta = new_viol - current_viol
                if delta < 0 or random.random() < np.exp(-delta / max(T, T_min)):
                    current_viol = new_viol
                    if current_viol < best_local_viol:
                        best_local_viol = current_viol
                        best_local_colors = colors.copy()
                        if current_viol == 0:
                            found = True
                            break
                else:
                    colors[pos] = old_color
                
                T *= alpha
            
            if found:
                break
            
            # Check best local
            if best_local_viol == 0:
                best_solution = best_local_colors.copy()
                best_num_colors = num_colors
                found = True
                break
        
        if found:
            break
    
    # If no solution found with limited colors, fall back to the greedy approach
    if best_solution is None:
        colors = []
        for x in range(1, 129):
            if time.time() > deadline - 2.0:
                # Emergency: fill rest with a simple pattern
                for remaining in range(len(colors), n):
                    colors.append(remaining % 4)
                break
            for color in range(128):
                valid = True
                # Check all 4-term progressions ending at x-1 (0-indexed)
                # x-1 is the new position (0-indexed)
                pos = x - 1
                for (a, b, c2, d) in progressions:
                    if d == pos:
                        # This progression ends at our new position
                        # Check if adding this color would make it have < 3 distinct colors
                        existing = [colors[a], colors[b], colors[c2]]
                        distinct_with_new = len(set(existing) | {color})
                        if distinct_with_new < 3:
                            valid = False
                            break
                    elif a == pos or b == pos or c2 == pos:
                        # Our new position is in the middle or start
                        # We can't fully check yet, but we can check partial constraints
                        pass
                if valid:
                    colors.append(color)
                    break
            else:
                # No color worked, use 0 as fallback
                colors.append(0)
        
        best_solution = np.array(colors)
    
    # Convert to list of integers
    result = [int(c) for c in best_solution]
    
    # Ensure length is exactly 128
    if len(result) < 128:
        for i in range(len(result), 128):
            result.append(i % 4)
    elif len(result) > 128:
        result = result[:128]
    
    return result
# EVOLVE-BLOCK-END
