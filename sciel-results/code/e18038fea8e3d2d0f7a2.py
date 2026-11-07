# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    n = 256
    # Precompute prime factors for each number 1..256
    prime_factors = [[] for _ in range(n + 1)]
    for i in range(2, n + 1):
        if not prime_factors[i]:
            for j in range(i, n + 1, i):
                prime_factors[j].append(i)
    
    # Function to check if a number is coprime to all in a set
    # We represent the set of selected numbers by their prime factors union
    # If a number's prime factors intersect with the union, it's not coprime to some element.
    
    def get_greedy_solution():
        selected = []
        used_primes = set()
        # Start with 1
        selected.append(1)
        # Iterate from 256 down to 2
        for x in range(n, 1, -1):
            # Check if x shares any prime factor with used_primes
            conflict = False
            for p in prime_factors[x]:
                if p in used_primes:
                    conflict = True
                    break
            if not conflict:
                selected.append(x)
                for p in prime_factors[x]:
                    used_primes.add(p)
        return selected, used_primes

    def evaluate_solution(selected):
        return sum(selected)

    def is_coprime_to_all(x, selected_set, used_primes):
        for p in prime_factors[x]:
            if p in used_primes:
                return False
        return True

    def local_improve(selected, used_primes, time_limit):
        # Convert to set for O(1) lookup
        selected_set = set(selected)
        current_sum = sum(selected)
        improved = True
        start_time = time.time()
        
        while improved and (time.time() - start_time < time_limit):
            improved = False
            # 1. Try to add any missing number that fits
            for x in range(n, 1, -1):
                if x not in selected_set:
                    if is_coprime_to_all(x, selected_set, used_primes):
                        selected.append(x)
                        selected_set.add(x)
                        for p in prime_factors[x]:
                            used_primes.add(p)
                        current_sum += x
                        improved = True
            if improved:
                continue # Restart loop to add more if possible

            # 2. Try to swap one element u with one element v (v > u)
            # Sort selected by value to prioritize replacing small ones
            # We iterate through unselected candidates v in descending order
            # and check if we can swap out a smaller u
            unselected = [x for x in range(n, 1, -1) if x not in selected_set]
            # We want to replace u with v if v > u and v is compatible with (selected - {u})
            # To be efficient, we check v from largest to smallest.
            # For each v, find if there is a u in selected such that u < v and v is compatible with rest.
            
            # Optimization: Precompute for each v, the set of primes it has.
            # A u can be removed if its primes are a subset of (primes of v + primes of rest)? No.
            # v is compatible with rest if no prime of v is in (used_primes - primes_of_u).
            # So primes_of_v & (used_primes - primes_of_u) must be empty.
            
            # Let's iterate v descending.
            for v in unselected:
                if time.time() - start_time > time_limit:
                    break
                if not is_coprime_to_all(v, selected_set, used_primes):
                    # v conflicts with some u in selected.
                    # Identify which u's conflict with v.
                    # If we remove ALL conflicting u's, can we add v?
                    # That's a multi-swap. Let's stick to 1-for-1 first.
                    # Find u in selected that conflicts with v.
                    # If there is exactly one u conflicting, and u < v, and v is compatible with others, swap.
                    # If multiple u's conflict, we might need to remove multiple. 
                    # For 1-for-1:
                    conflicting_us = []
                    for u in selected:
                        if math.gcd(u, v) != 1:
                            conflicting_us.append(u)
                    
                    if len(conflicting_us) == 1:
                        u = conflicting_us[0]
                        if u < v:
                            # Check if v is compatible with selected - {u}
                            # This is true if the only conflict was u.
                            # Since we found u by gcd != 1, and if there were others, len > 1.
                            # So v is compatible with rest.
                            # Perform swap
                            selected.remove(u)
                            selected_set.remove(u)
                            # Remove u's primes from used_primes, but only if they are not in other selected numbers
                            # Actually, easier to rebuild used_primes or update carefully.
                            # Let's update used_primes:
                            for p in prime_factors[u]:
                                # Check if p is in any other selected number
                                # This is O(N) per prime. 
                                # Alternative: Just rebuild used_primes if we swap? 
                                # Or maintain a count of primes.
                                # Let's use a prime_count dict.
                                pass 
                            # To keep it simple and robust, let's just rebuild used_primes if we make a change.
                            # But rebuilding is O(N * log N). With 256 numbers, it's fast.
                            
                            # Let's implement a helper to rebuild
                            new_selected = [x for x in selected if x != u] + [v]
                            new_used = set()
                            for x in new_selected:
                                for p in prime_factors[x]:
                                    new_used.add(p)
                            
                            # Verify sum increase
                            if v > u:
                                selected = new_selected
                                selected_set = set(selected)
                                used_primes = new_used
                                current_sum += (v - u)
                                improved = True
                                break # Restart inner loop for v, or outer? 
                                # Breaking inner v loop, but we should restart the whole improvement loop 
                                # because the set changed.
            
            if improved:
                continue
                
            # 3. Try 1-for-2 swap: Remove u, add v, w
            # This is more complex. 
            # For now, 1-for-1 and Add are good.
            
            # 4. Random Kick if stuck
            # If no improvement found in a full pass, try a random restart with a subset
            # Actually, the "improved" flag handles the loop. If we exit while, we are local optimal.
            
        return selected, used_primes, current_sum

    # Initial Greedy
    selected, used_primes = get_greedy_solution()
    
    # Local Improvement
    # We run local improvement multiple times or with randomization if stuck?
    # The current local_improve is deterministic. It will find a local optimum.
    # To find a better one, we can try different greedy strategies or random restarts.
    
    best_sum = sum(selected)
    best_selected = list(selected)
    
    start_time = time.time()
    TIME_LIMIT = 58.0
    
    # Try multiple random restarts or perturbations
    # Strategy: 
    # 1. Greedy descending (done)
    # 2. Greedy ascending? (Usually worse)
    # 3. Randomized Greedy: Pick random valid number, repeat.
    # 4. Perturb current best: Remove k random elements, re-greedy.
    
    # Let's do Perturbation + Local Improve
    attempts = 0
    while time.time() - start_time < TIME_LIMIT:
        attempts += 1
        # Create a perturbed version of the best solution
        # Remove 10% of elements randomly
        curr_sel = list(best_selected)
        # Ensure 1 is kept? Not strictly necessary but good.
        # Randomly remove some elements
        num_to_remove = max(1, len(curr_sel) // 10)
        # Don't remove 1 if possible?
        removable = [x for x in curr_sel if x != 1]
        if not removable:
            removable = curr_sel[:]
        
        to_remove = random.sample(removable, min(num_to_remove, len(removable)))
        for x in to_remove:
            curr_sel.remove(x)
        
        # Re-greedy from this state
        # We need to reconstruct used_primes for curr_sel
        used_p = set()
        for x in curr_sel:
            for p in prime_factors[x]:
                used_p.add(p)
        
        # Add more numbers greedily
        for x in range(n, 1, -1):
            if x not in curr_sel:
                if is_coprime_to_all(x, set(curr_sel), used_p):
                    curr_sel.append(x)
                    for p in prime_factors[x]:
                        used_p.add(p)
        
        curr_sum = sum(curr_sel)
        
        # Local Improve
        curr_sel, used_p, curr_sum = local_improve(curr_sel, used_p, 2.0) # Limit time per improvement
        
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_selected = list(curr_sel)
            
    return best_selected
# EVOLVE-BLOCK-END
