import time
import math
from fractions import Fraction
import numpy as np
import sys

# EVOLVE-BLOCK-START

def _sieve(n):
    """Generate primes up to n."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]

# Precompute primes up to 1000
PRIMES = _sieve(1000)
PRIME_SET = set(PRIMES)
# Sort primes by size, small primes first (higher density, higher cost)
PRIMES_SORTED = sorted(PRIMES, key=lambda x: x)

def _calculate_coverage(classes, limit=10000):
    """
    Calculate the largest L such that 1..L are covered by the given classes.
    Uses a bytearray for speed.
    Returns L.
    """
    if not classes:
        return 0
    
    covered = bytearray(limit + 1)
    
    # Mark covered positions
    for p, r in classes:
        start = r if r > 0 else p
        if start < 1:
            start += p
        for n in range(start, limit + 1, p):
            covered[n] = 1
            
    # Find the largest L such that 1..L are all True
    L = 0
    for i in range(1, limit + 1):
        if covered[i]:
            L = i
        else:
            break
            
    return L

def _sum_reciprocals_exact(classes):
    """Exact Fraction sum for final verification."""
    total = Fraction(0, 1)
    for p, r in classes:
        total += Fraction(1, p)
    return total

def _sum_reciprocals_float(classes):
    """Fast float sum for pruning."""
    return sum(1.0 / p for p, r in classes)

def _solve_csp(L_target, time_limit=10.0):
    """
    Solve the covering problem using Backtracking with Forward Checking.
    We decide for each prime (in order of size) whether to include it and which residue.
    
    State:
    - current_classes: list of [p, r]
    - remaining_budget: float
    - covered_mask: bytearray of length L_target+1, 1 if covered
    - uncovered_count: number of n in 1..L_target not covered
    
    Heuristic:
    - Process primes in increasing order (small primes cover more, are more critical).
    - For each prime, try residues that cover the most *currently uncovered* numbers.
    - Prune if remaining_budget < 0 or if it's impossible to cover remaining gaps 
      (simple bound: max coverage per remaining prime is p, but we use a looser bound 
      based on total remaining capacity vs uncovered count).
    """
    start_time = time.time()
    
    # Precompute for each n in 1..L_target, which primes can cover it?
    # Actually, for forward checking, we need to know, for a given prime p and residue r,
    # which uncovered numbers it covers.
    
    # To speed up, we maintain a list of uncovered numbers.
    # But updating this list efficiently is key.
    
    # Let's use a simpler approach:
    # Recursion depth = number of primes considered.
    # We don't need to consider all primes. We can stop when uncovered_count == 0.
    # Also, if remaining_budget is too small to cover any more, stop.
    
    # Optimization: Only consider primes that are "useful".
    # A prime p is useful if there exists an r such that it covers at least one uncovered number.
    
    # We'll use a global counter for nodes visited to enforce time limit.
    nodes_visited = 0
    
    # Initial state
    initial_covered = bytearray(L_target + 1)
    initial_uncovered = list(range(1, L_target + 1))
    
    # We will search for a solution.
    # To make it faster, we can use a "most constrained variable" heuristic?
    # No, the primes are the variables. The order is fixed (sorted by p).
    # But we can skip primes that don't help?
    # If a prime p is large, it covers few numbers.
    
    # Let's implement the backtracking.
    
    # To avoid recursion limit issues, we use an iterative stack or careful recursion.
    # Python recursion limit is ~1000. We have 168 primes. Depth 168 is fine.
    
    def backtrack(p_idx, current_classes, remaining_budget, covered, uncovered_set):
        nonlocal nodes_visited
        nodes_visited += 1
        if nodes_visited % 1000 == 0:
            if time.time() - start_time > time_limit:
                return None
        
        if not uncovered_set:
            return current_classes[:]
        
        # Pruning: If remaining budget is insufficient to cover any more numbers?
        # Minimum cost is 1/max_prime. But we need to cover |uncovered_set| numbers.
        # This bound is weak.
        
        # If we've run out of primes
        if p_idx >= len(PRIMES_SORTED):
            return None
            
        p = PRIMES_SORTED[p_idx]
        
        # Option 1: Skip this prime
        # Only skip if we can still cover the rest with remaining primes?
        # Hard to know. Just try it.
        
        # Option 2: Use this prime with some residue r
        # Find the residue r that maximizes coverage of uncovered_set
        best_r = -1
        best_gain = 0
        best_covered_new = None
        best_uncovered_new = None
        
        # Iterate through possible residues?
        # There are p residues. For large p, this is expensive.
        # But p is at most 1000.
        # We can optimize by only checking residues that hit at least one uncovered number.
        
        # Let's find which uncovered numbers are divisible by p? No, n % p == r.
        # So for each uncovered n, r = n % p.
        # We can group uncovered numbers by their residue mod p.
        
        residues_count = {}
        for n in uncovered_set:
            r = n % p
            residues_count[r] = residues_count.get(r, 0) + 1
            
        # The best residue is the one with the highest count
        if residues_count:
            # Find max count
            max_count = max(residues_count.values())
            # If multiple, pick any? Or pick the one that leaves the "easiest" uncovered set?
            # For now, pick the one with max count.
            best_r = max(residues_count, key=residues_count.get)
            best_gain = max_count
            
            # If gain is 0, skip this prime (it covers no new numbers)
            if best_gain == 0:
                # Try skipping
                return backtrack(p_idx + 1, current_classes, remaining_budget, covered, uncovered_set)
            
            # Calculate new uncovered set
            # This is O(|uncovered_set|)
            new_uncovered = []
            for n in uncovered_set:
                if n % p != best_r:
                    new_uncovered.append(n)
            
            # Check budget
            cost = 1.0 / p
            if remaining_budget >= cost - 1e-9:
                # Try this branch
                new_classes = current_classes + [[p, best_r]]
                new_budget = remaining_budget - cost
                # Note: 'covered' bytearray is not strictly necessary if we use 'uncovered_set'
                # We can pass 'uncovered_set' as the state.
                
                res = backtrack(p_idx + 1, new_classes, new_budget, covered, new_uncovered)
                if res is not None:
                    return res
        
        # Try skipping this prime (if we didn't already return from the skip branch above)
        # Wait, the logic above:
        # If best_gain == 0, we skip.
        # If best_gain > 0, we try the "use" branch. If it fails, we should try "skip" branch.
        # So we need to structure it:
        # 1. Try "use" branch if beneficial.
        # 2. Try "skip" branch.
        
        # Refactor:
        # We already tried "use" if best_gain > 0 and it failed (returned None).
        # Now try "skip".
        
        return backtrack(p_idx + 1, current_classes, remaining_budget, covered, uncovered_set)

    # The above recursion is a bit messy with the "skip" logic.
    # Let's rewrite cleanly.
    
    def bt(p_idx, classes, budget, uncovered):
        nonlocal nodes_visited
        nodes_visited += 1
        if nodes_visited % 500 == 0:
            if time.time() - start_time > time_limit:
                return None
        
        if not uncovered:
            return classes[:]
        
        if p_idx >= len(PRIMES_SORTED):
            return None
            
        p = PRIMES_SORTED[p_idx]
        
        # 1. Try to use prime p
        # Find best residue
        res_counts = {}
        for n in uncovered:
            r = n % p
            res_counts[r] = res_counts.get(r, 0) + 1
            
        if res_counts:
            best_r = max(res_counts, key=res_counts.get)
            gain = res_counts[best_r]
            
            if gain > 0 and budget >= 1.0/p - 1e-9:
                # New uncovered set
                new_uncovered = [n for n in uncovered if n % p != best_r]
                new_classes = classes + [[p, best_r]]
                new_budget = budget - 1.0/p
                
                res = bt(p_idx + 1, new_classes, new_budget, new_uncovered)
                if res is not None:
                    return res
        
        # 2. Try to skip prime p
        # Optimization: If skipping p means we can't cover the rest?
        # Simple check: Is there any remaining prime that can cover the uncovered numbers?
        # If not, return None.
        # For now, just recurse.
        
        return bt(p_idx + 1, classes, budget, uncovered)

    # Start with all numbers 1..L_target uncovered
    initial_uncovered = list(range(1, L_target + 1))
    initial_budget = 1.0
    initial_classes = []
    
    return bt(0, initial_classes, initial_budget, initial_uncovered)

def _local_search_refinement(classes, L_current, time_limit=5.0):
    """
    Refine the solution to extend L further.
    """
    start_time = time.time()
    current_classes = [c[:] for c in classes]
    current_L = L_current
    
    while time.time() - start_time < time_limit:
        # Find first gap
        check_limit = current_L + 100
        if check_limit > 10000: check_limit = 10000
        
        covered = bytearray(check_limit + 1)
        for p, r in current_classes:
            start = r if r > 0 else p
            if start < 1: start += p
            for n in range(start, check_limit + 1, p):
                covered[n] = 1
                
        gap = -1
        for i in range(1, check_limit + 1):
            if not covered[i]:
                gap = i
                break
                
        if gap == -1:
            if check_limit > current_L:
                current_L = check_limit
            else:
                break # No improvement possible in this window
                
        # Try to fix gap
        best_new_L = current_L
        best_new_classes = None
        
        # Option 1: Change residue of existing prime
        for idx, (p, r) in enumerate(current_classes):
            new_r = gap % p
            if new_r == r: continue
            trial = [c[:] for c in current_classes]
            trial[idx] = [p, new_r]
            L_trial = _calculate_coverage(trial, limit=current_L + 100)
            if L_trial > best_new_L:
                best_new_L = L_trial
                best_new_classes = trial
                
        # Option 2: Add new prime
        used_primes = set(p for p, r in current_classes)
        budget = sum(1.0/p for p, r in current_classes)
        for p in PRIMES_SORTED:
            if p in used_primes: continue
            if budget + 1.0/p <= 1.0:
                new_r = gap % p
                trial = [c[:] for c in current_classes]
                trial.append([p, new_r])
                L_trial = _calculate_coverage(trial, limit=current_L + 100)
                if L_trial > best_new_L:
                    best_new_L = L_trial
                    best_new_classes = trial
                break # Try smallest prime first
                
        if best_new_classes is not None and best_new_L > current_L:
            current_classes = best_new_classes
            current_L = best_new_L
        else:
            break # Stuck
            
    return current_classes, current_L

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # 1. Initial Greedy Solution
    # Use small primes to get a baseline
    initial_classes = []
    used_primes = set()
    budget = 0.0
    covered = [False] * 1000
    for n in range(1, 1000):
        if covered[n]: continue
        for p in PRIMES_SORTED:
            if p in used_primes: continue
            if budget + 1.0/p <= 1.0:
                r = n % p
                initial_classes.append([p, r])
                used_primes.add(p)
                budget += 1.0/p
                start = r if r > 0 else p
                if start < 1: start += p
                for k in range(start, 1000, p):
                    covered[k] = True
                break
        if time.time() > deadline - 130: break
        
    if not initial_classes:
        return [[2, 1], [3, 2], [5, 4]]
        
    current_best_L = _calculate_coverage(initial_classes, limit=5000)
    best_classes = initial_classes
    
    # 2. CSP Search for larger L
    # We try to find a solution for L = current_best_L + step
    # CSP is exponential, so we must be careful with time limits.
    
    # Try increasing L in steps
    # Start from current_best_L
    # We want to maximize L.
    
    # Let's try a few targets
    # Target 1: current_best_L + 50
    # Target 2: current_best_L + 100
    # etc.
    
    # Since CSP can be slow, we limit the time per CSP call.
    # We will try to solve for increasing L values.
    
    # Note: The CSP solver as written is a pure backtracking.
    # It might be too slow for L > 1000.
    # Let's add a heuristic: If L is large, the number of uncovered numbers is large,
    # and the branching factor is high.
    
    # Alternative: Use the CSP to *repair* or *improve* a greedy solution?
    # Or use it to find a solution from scratch for a specific L.
    
    # Let's try solving for L = 1000 first, then 1100, etc.
    # If it fails, we keep the best found.
    
    # To make CSP faster, we can use the greedy solution as a "hint" or starting point?
    # The current CSP starts from scratch.
    
    # Let's try a hybrid:
    # 1. Greedy to get L0.
    # 2. CSP to try to reach L0 + delta.
    
    # We will iterate L_target from L0+10 to L0+200 in steps of 10.
    # For each, run CSP with time_limit = 5s.
    
    last_feasible_L = current_best_L
    last_feasible_classes = best_classes
    
    # We can also try to solve for L=1000 directly if we haven't reached it.
    # If current_best_L < 1000, try 1000.
    
    targets = []
    if current_best_L < 1000:
        targets.append(1000)
    
    # Add incremental targets
    next_target = current_best_L + 20
    while next_target < 2000 and time.time() < deadline - 30:
        targets.append(next_target)
        next_target += 20
        
    # Sort targets to try smaller ones first? Or larger?
    # Smaller ones are easier. Let's try smaller first to establish a baseline, then bigger.
    targets = sorted(set(targets))
    
    for L_target in targets:
        if time.time() > deadline - 20:
            break
            
        # Only try if L_target > last_feasible_L
        if L_target <= last_feasible_L:
            continue
            
        # Run CSP
        # Time limit depends on how much time we have left
        time_left = deadline - time.time()
        t_limit = min(10.0, time_left * 0.5)
        
        if t_limit < 1.0:
            break
            
        classes = _solve_csp(L_target, time_limit=t_limit)
        
        if classes is not None:
            # Verify
            if _sum_reciprocals_exact(classes) <= 1:
                L_check = _calculate_coverage(classes, limit=10000)
                if L_check >= L_target:
                    last_feasible_L = L_check
                    last_feasible_classes = classes
                    # Update current_best_L for subsequent targets?
                    # Yes, we can use this as a new baseline.
                    current_best_L = L_check
                    best_classes = classes
                    # Re-calculate targets?
                    # For now, just continue with the pre-computed list.
                    # If we found a much better L, we might want to try even higher.
                    # But the list is fixed.
                    
    # 3. Local Search Refinement
    if last_feasible_classes:
        refined_classes, refined_L = _local_search_refinement(last_feasible_classes, last_feasible_L, time_limit=10.0)
        if refined_L > last_feasible_L:
            last_feasible_L = refined_L
            last_feasible_classes = refined_classes
    
    # Final check
    if last_feasible_classes and _sum_reciprocals_exact(last_feasible_classes) <= 1:
        return last_feasible_classes
    else:
        return [[2, 1], [3, 2], [5, 4]]

# EVOLVE-BLOCK-END
