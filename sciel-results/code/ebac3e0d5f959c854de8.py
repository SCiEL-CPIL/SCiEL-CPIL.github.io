"""Rewritten with structural change: enumerate small anchors, solve MWIS on large numbers via greedy + local search."""
import time
import sys
from math import gcd
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    def lcm_gt_256(a, b):
        """Check if lcm(a,b) > 256"""
        return a * b // gcd(a, b) > 256
    
    # Precompute compatibility for all pairs
    # We'll use a set-based approach for efficiency
    
    # Strategy: 
    # 1. Try each possible "small" anchor (2-30) 
    # 2. For each anchor, find all large numbers (31-256) compatible with it
    # 3. Solve MWIS on those large numbers using greedy + local search
    # 4. Also try solutions without any small anchor (just large numbers 31-256)
    
    best_set = []
    best_sum = Fraction(0)
    
    def update_best(candidate_set):
        nonlocal best_set, best_sum
        if not candidate_set:
            return
        # Verify the set is valid
        s = sorted(candidate_set)
        valid = True
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                if not lcm_gt_256(s[i], s[j]):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            total = sum(Fraction(1, x) for x in candidate_set)
            if total > best_sum:
                best_sum = total
                best_set = list(candidate_set)
    
    def solve_mwis_large(numbers, time_budget):
        """Solve max weight independent set on large numbers using greedy + local search"""
        if time.time() > deadline:
            return [], Fraction(0)
        
        n = len(numbers)
        if n == 0:
            return [], Fraction(0)
        
        # Build adjacency: i and j are adjacent if NOT compatible (lcm <= 256)
        # We want to select a set with no edges between them
        
        # Greedy: pick numbers in order of decreasing weight (1/x), add if compatible
        def greedy():
            selected = []
            remaining = set(numbers)
            # Sort by 1/x descending (i.e., x ascending)
            for x in sorted(remaining, key=lambda v: -1/v):  # wrong, let me fix
                pass
            # Actually sort by x ascending to get 1/x descending
            for x in sorted(numbers, key=lambda v: v):
                compatible = True
                for y in selected:
                    if not lcm_gt_256(x, y):
                        compatible = False
                        break
                if compatible:
                    selected.append(x)
            return selected
        
        # Local search: try to improve by swaps
        def local_search(initial_set, max_iters=500):
            current = list(initial_set)
            current_sum = sum(Fraction(1, x) for x in current)
            
            # Precompute which numbers are compatible with each selected
            # Try: remove one, add up to two better ones
            for _ in range(max_iters):
                if time.time() > deadline:
                    break
                improved = False
                current_set = set(current)
                
                # Try single swaps: remove x, add y where y not in set
                for i, x in enumerate(current):
                    # Try removing x
                    candidate_without = [v for v in current if v != x]
                    
                    # Try adding each unselected number
                    for y in numbers:
                        if y in current_set:
                            continue
                        # Check if y is compatible with all in candidate_without
                        compatible = True
                        for z in candidate_without:
                            if not lcm_gt_256(y, z):
                                compatible = False
                                break
                        if not compatible:
                            continue
                        
                        new_sum = sum(Fraction(1, v) for v in candidate_without) + Fraction(1, y)
                        if new_sum > current_sum:
                            current = candidate_without + [y]
                            current_sum = new_sum
                            improved = True
                            break
                    if improved:
                        break
                
                # Try double swaps: remove two, add one
                if not improved:
                    for i in range(len(current)):
                        for j in range(i+1, len(current)):
                            x, y = current[i], current[j]
                            candidate_without = [v for v in current if v not in (x, y)]
                            
                            for z in numbers:
                                if z in current_set:
                                    continue
                                compatible = True
                                for w in candidate_without:
                                    if not lcm_gt_256(z, w):
                                        compatible = False
                                        break
                                if not compatible:
                                    continue
                                
                                new_sum = sum(Fraction(1, v) for v in candidate_without) + Fraction(1, z)
                                if new_sum > current_sum:
                                    current = candidate_without + [z]
                                    current_sum = new_sum
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                
                if not improved:
                    break
            
            return current, current_sum
        
        # Run greedy to get initial solution
        greedy_sol = greedy()
        
        # Run local search from greedy solution
        final_sol, final_sum = local_search(greedy_sol)
        
        # Also try a few random restarts if time permits
        import random
        rng = random.Random(42)
        for restart in range(10):
            if time.time() > deadline:
                break
            
            # Random greedy: shuffle and pick
            shuffled = list(numbers)
            rng.shuffle(shuffled)
            random_sol = []
            for x in shuffled:
                compatible = True
                for y in random_sol:
                    if not lcm_gt_256(x, y):
                        compatible = False
                        break
                if compatible:
                    random_sol.append(x)
            
            ls_sol, ls_sum = local_search(random_sol, max_iters=200)
            if ls_sum > final_sum:
                final_sol, final_sum = ls_sol, ls_sum
        
        return final_sol, final_sum
    
    # Large numbers: 31 to 256 (these have fewer conflicts among themselves)
    large_numbers = list(range(31, 257))
    
    # Case 1: No small anchor, just large numbers
    large_sol, large_sum = solve_mwis_large(large_numbers, 20.0)
    update_best(large_sol)
    
    # Case 2: Try each small anchor from 2 to 30
    for anchor in range(2, 31):
        if time.time() > deadline:
            break
        
        # Find all large numbers compatible with anchor
        compatible_large = [x for x in large_numbers if lcm_gt_256(anchor, x)]
        
        if not compatible_large:
            continue
        
        # Solve MWIS on compatible large numbers
        sol, s = solve_mwis_large(compatible_large, 10.0)
        full_set = [anchor] + sol
        update_best(full_set)
    
    # Case 3: Try pairs of small anchors
    for a in range(2, 20):
        for b in range(a+1, 21):
            if time.time() > deadline:
                break
            if not lcm_gt_256(a, b):
                continue
            
            compatible_large = [x for x in large_numbers if lcm_gt_256(a, x) and lcm_gt_256(b, x)]
            if not compatible_large:
                continue
            
            sol, s = solve_mwis_large(compatible_large, 5.0)
            full_set = [a, b] + sol
            update_best(full_set)
    
    if not best_set:
        # Fallback: just return a single element (any single element is valid)
        best_set = [2]
    
    return best_set
