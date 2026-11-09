"""Simulated Annealing for reciprocal weight of 3-progression-free sets on [1,256]."""
import time
import random
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def _can_add(x, S_set, S_list):
    """Check if x can be added to S without creating a 3-AP. O(K) time."""
    for a in S_list:
        if a == x:
            continue
        # Case 1: x is the middle term. a, x, c. c = 2x - a.
        c = 2 * x - a
        if 1 <= c <= 256 and c != x and c in S_set:
            return False
        # Case 2: x is an endpoint. a, b, x or x, b, a. b = (a+x)/2.
        s_sum = a + x
        if s_sum % 2 == 0:
            b = s_sum // 2
            if b != x and b in S_set:
                return False
    return True

def _recip_sum_float(selected):
    """Compute float sum of reciprocals for speed during SA."""
    return sum(1.0/x for x in selected)

def _recip_sum_exact(selected):
    """Compute exact sum of reciprocals."""
    total = Fraction(0, 1)
    for x in selected:
        total += Fraction(1, x)
    return total

def _greedy_by_reciprocal(n=256):
    """Greedy: add elements in order of decreasing 1/x (i.e., smallest x first)."""
    selected = []
    selected_set = set()
    for x in range(1, n+1):
        if _can_add(x, selected_set, selected):
            selected.append(x)
            selected_set.add(x)
    return selected

def _random_greedy(n=256, seed=0):
    """Greedy with random ordering weighted by reciprocal."""
    rng = random.Random(seed)
    candidates = list(range(1, n+1))
    weights = [1.0/x for x in candidates]
    noisy_weights = [w * (0.5 + rng.random()) for w in weights]
    order = sorted(candidates, key=lambda x: noisy_weights[x-1], reverse=True)
    
    selected = []
    selected_set = set()
    for x in order:
        if _can_add(x, selected_set, selected):
            selected.append(x)
            selected_set.add(x)
    
    return selected

def _simulated_annealing(start_set, n=256, deadline=None, seed=0):
    """
    Simulated Annealing to improve the reciprocal sum.
    Moves: 
    1. Remove a random element.
    2. Add a random valid element.
    3. Swap a random element with a random valid element.
    """
    rng = random.Random(seed)
    current = list(start_set)
    current_set = set(current)
    current_val = _recip_sum_float(current)
    
    best = list(current)
    best_val = current_val
    
    # Initial temperature
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.999
    
    start_time = time.time()
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate a neighbor
        move_type = rng.random()
        
        if move_type < 0.3:
            # Remove a random element
            if not current:
                continue
            idx = rng.randrange(len(current))
            removed = current[idx]
            new_set = current_set - {removed}
            new_list = [x for x in current if x != removed]
            new_val = current_val - 1.0/removed
            
            # Acceptance: always accept removal if it allows better additions later? 
            # No, SA accepts based on energy difference.
            # Delta is negative (worse). Accept with prob exp(delta/T).
            delta = new_val - current_val
            if delta > 0 or rng.random() < math.exp(delta / T):
                current = new_list
                current_set = new_set
                current_val = new_val
                if current_val > best_val:
                    best_val = current_val
                    best = list(current)
                    
        elif move_type < 0.6:
            # Add a random valid element
            # Find candidates that can be added
            # To be fast, we pick a random number of trials to find a valid one
            # Or just try a random number
            if len(current) < n:
                # Pick a random candidate from 1..n not in set
                # Efficient way: just pick random x and check
                trials = 0
                while trials < 10:
                    x = rng.randint(1, n)
                    if x not in current_set:
                        if _can_add(x, current_set, current):
                            new_list = current + [x]
                            new_val = current_val + 1.0/x
                            delta = new_val - current_val
                            if delta > 0 or rng.random() < math.exp(delta / T):
                                current = new_list
                                current_set = current_set | {x}
                                current_val = new_val
                                if current_val > best_val:
                                    best_val = current_val
                                    best = list(current)
                            break
                    trials += 1
        else:
            # Swap: Remove one, Add one
            if not current:
                continue
            idx = rng.randrange(len(current))
            removed = current[idx]
            
            # Try to find a valid add candidate
            trials = 0
            while trials < 10:
                x = rng.randint(1, n)
                if x not in current_set:
                    # Check if x can be added to current - {removed}
                    temp_set = current_set - {removed}
                    temp_list = [a for a in current if a != removed]
                    if _can_add(x, temp_set, temp_list):
                        new_list = [a for a in current if a != removed]
                        new_list.append(x)
                        new_val = current_val - 1.0/removed + 1.0/x
                        delta = new_val - current_val
                        if delta > 0 or rng.random() < math.exp(delta / T):
                            current = new_list
                            current_set = temp_set | {x}
                            current_val = new_val
                            if current_val > best_val:
                                best_val = current_val
                                best = list(current)
                        break
                trials += 1
                
        T *= cooling_rate
        
    return best

def construct():
    """Main entry point: find a 3-free subset of [1,256] maximizing sum of reciprocals."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Generate diverse starting points
    # Greedy
    greedy_set = _greedy_by_reciprocal(256)
    greedy_val = _recip_sum_float(greedy_set)
    
    best_set = greedy_set
    best_val = greedy_val
    
    # Random Greedy
    for i in range(50):
        if time.time() > deadline - 10:
            break
        cand = _random_greedy(256, seed=i)
        val = _recip_sum_float(cand)
        if val > best_val:
            best_val = val
            best_set = cand
            
    # 2. Simulated Annealing from best starting point
    # Run multiple SA runs with different seeds and temperatures if time permits
    # Since SA is stochastic, running it on the best greedy solution is a good start.
    # We can also run SA on other good greedy solutions.
    
    # Sort starting points by value
    # Actually, let's just run SA on the current best.
    # To be more robust, we can run SA on the top 5 greedy solutions.
    
    # Let's collect top 5 greedy solutions
    greedy_solutions = [greedy_set]
    for i in range(50):
        if time.time() > deadline - 20:
            break
        cand = _random_greedy(256, seed=100+i)
        greedy_solutions.append(cand)
        
    greedy_solutions.sort(key=_recip_sum_float, reverse=True)
    
    # Run SA on the top 3 unique solutions
    seen_sets = set()
    sa_candidates = []
    for sol in greedy_solutions:
        if time.time() > deadline - 5:
            break
        sol_tuple = tuple(sorted(sol))
        if sol_tuple in seen_sets:
            continue
        seen_sets.add(sol_tuple)
        sa_candidates.append(sol)
        if len(sa_candidates) >= 3:
            break
            
    # Allocate remaining time to SA runs
    time_left = deadline - time.time()
    if time_left > 1:
        # Run SA on the best candidate
        # We can split time among candidates or just run on the best
        # Running on the best is usually sufficient if the landscape is smooth enough
        # But diverse SA runs can help. Let's run on the best one with the full remaining time.
        # Or split it. Let's try splitting if we have multiple good candidates.
        
        # For simplicity and robustness, let's run SA on the single best greedy solution
        # with the full remaining time.
        final_set = _simulated_annealing(sa_candidates[0], 256, deadline=deadline, seed=42)
        final_val = _recip_sum_float(final_set)
        
        if final_val > best_val:
            best_val = final_val
            best_set = final_set
            
        # If we have time, run SA on the second best greedy solution too?
        # The deadline is shared. Let's just use the best one.
        # Actually, let's try to run SA on the best, and if it finishes early, run on next.
        # But _simulated_annealing runs until deadline. 
        # So we just run it on the best one.
        
    # Final verification and exact sum calculation (not strictly needed for return, but good for sanity)
    # The return value is the set. The evaluator computes the score.
    
    return best_set

# EVOLVE-BLOCK-END
