import math
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin for safety
    
    N = 256
    domain = list(range(2, N + 1))
    n = len(domain)
    
    # Precompute weights as Fractions for exact comparison, 
    # but use floats for speed in heuristic scoring if needed.
    # Actually, for local search, comparing sums of reciprocals 
    # can be done with floats if we are careful, or Fractions for precision.
    # Given n=255, sum of 1/x is small. Float precision is ~15-16 digits.
    # Differences between optimal and near-optimal sets are likely > 1e-6.
    # We will use float for speed in the inner loop, and verify at the end.
    weights = [1.0 / x for x in domain]
    
    # Precompute conflict matrix: conflict[i][j] is True if lcm(domain[i], domain[j]) <= 256
    # We need this for fast O(1) or O(deg) checks during moves.
    # n=255, matrix is 255x255 = 65k entries. Very small.
    conflict = [[False]*n for _ in range(n)]
    for i in range(n):
        vi = domain[i]
        for j in range(i+1, n):
            vj = domain[j]
            if math.lcm(vi, vj) <= 256:
                conflict[i][j] = True
                conflict[j][i] = True

    def is_valid_addition(idx, current_set):
        """Check if adding idx to current_set (set of indices) is valid."""
        for j in current_set:
            if conflict[idx][j]:
                return False
        return True

    def get_score(indices):
        """Calculate sum of reciprocals for the set of indices."""
        s = 0.0
        for i in indices:
            s += weights[i]
        return s

    # Strategy: Multi-start Local Search
    # 1. Generate several diverse initial solutions.
    #    - Greedy by weight (smallest first)
    #    - Greedy by weight excluding small numbers (e.g., skip 2, skip 2&3, etc.)
    #    - Random valid sets
    # 2. For each, perform local search (Add, Remove, Swap).
    
    best_set = set()
    best_score = 0.0
    
    def local_search(initial_set, max_iterations=5000):
        """
        Perform local search starting from initial_set.
        Moves:
        - Add a valid element.
        - Remove an element (only if it allows adding a better one later? 
          Actually, simple hill climbing on sum: 
          - Try to add any valid element.
          - Try to swap: remove one, add two? Or remove one, add one?
          Since we maximize sum, adding is always good if valid.
          Removing is only good if it enables adding multiple larger-weight elements.
        """
        current_set = set(initial_set)
        current_score = get_score(current_set)
        
        # Precompute list of all indices for random access
        all_indices = list(range(n))
        
        # Simple Hill Climbing: 
        # 1. Try to add any valid element. If multiple, pick the one with highest weight?
        #    Or just any? Picking highest weight greedily is a common heuristic.
        #    Let's try: while there exists a valid addition, add the one with max weight.
        #    This leads to a maximal independent set (w.r.t. weight order).
        #    But this might miss a combination where a lower weight element allows two higher ones?
        #    No, if A conflicts with B and C, but B and C don't conflict, and w(B)+w(C) > w(A), 
        #    we should swap A for B,C.
        
        # Better Local Search:
        # Phase 1: Greedy Add. While any valid element can be added, add the one with max weight.
        # Phase 2: Local Swap. Try to replace a subset of size k with a subset of size l 
        #           such that the new set is valid and score increases.
        #           Simplest: Swap 1-for-1, 1-for-2, 2-for-1.
        
        improved = True
        while improved:
            improved = False
            
            # Phase 1: Add best valid element
            best_add_idx = -1
            best_add_w = -1.0
            for i in all_indices:
                if i not in current_set and is_valid_addition(i, current_set):
                    if weights[i] > best_add_w:
                        best_add_w = weights[i]
                        best_add_idx = i
            
            if best_add_idx != -1:
                current_set.add(best_add_idx)
                current_score += weights[best_add_idx]
                improved = True
                continue # Restart loop to check for more additions
            
            # Phase 2: Try Swaps
            # 2a. 1-for-1 swap: Remove i, add j.
            #     Condition: j not in set, i in set. 
            #     New set valid if: 
            #       1. j doesn't conflict with (current_set - {i})
            #       2. i didn't conflict with j (implied by above if we remove i first? No, we just check j vs rest)
            #     Actually, if we remove i, the rest is valid. We just need j to be valid against the rest.
            #     Gain: weights[j] - weights[i] > 0
            
            # 2b. 1-for-2 swap: Remove i, add j, k.
            #     Condition: j, k not in set. j, k valid against (current_set - {i}) and against each other.
            #     Gain: weights[j] + weights[k] - weights[i] > 0
            
            # 2c. 2-for-1 swap: Remove i, j, add k.
            #     Condition: k not in set. k valid against (current_set - {i, j}).
            #     Gain: weights[k] - weights[i] - weights[j] > 0

            # Optimization: Only try swaps if current set is not empty.
            if not current_set:
                break
                
            # Try 1-for-1
            # For each i in current_set, find best j not in set, valid against current_set - {i}, with weights[j] > weights[i]
            # This is O(|S| * |V| * deg). |S|~50, |V|~250. 12k * 50 = 600k ops. Fast.
            best_swap_gain = 0
            best_swap_type = None
            best_swap_params = None
            
            # 1-for-1
            for i in current_set:
                # Check all j not in current_set
                for j in all_indices:
                    if j in current_set:
                        continue
                    if weights[j] <= weights[i]:
                        continue
                    # Check if j is valid against current_set - {i}
                    valid = True
                    for k in current_set:
                        if k == i:
                            continue
                        if conflict[j][k]:
                            valid = False
                            break
                    if valid:
                        gain = weights[j] - weights[i]
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap_type = '1-1'
                            best_swap_params = (i, j)
            
            # 1-for-2
            # For each i in current_set, find j, k not in set, j<k, 
            # valid against current_set-{i}, j,k valid against each other.
            # This is O(|S| * |V|^2). 50 * 250^2 = 3M ops. Might be a bit slow in Python if done every iteration.
            # We can limit this to less frequent checks or smaller sets.
            # Let's try 1-for-2 only if 1-for-1 didn't improve and set size is small?
            # Or just do it every time but break early?
            
            if best_swap_gain <= 0:
                 # Try 1-for-2
                 # To speed up, we can pre-sort candidates?
                 # For now, simple loop.
                 for i in current_set:
                     # We want to replace i with j and k
                     # j, k must be valid against S\{i}
                     # j, k must be valid against each other
                     # Gain: w(j)+w(k)-w(i) > 0
                     
                     # Optimization: If the max possible gain is <= 0, skip.
                     # Max possible gain is sum of top 2 weights in V\S.
                     # But we need to check validity.
                     
                     # Simple approach: iterate j, k
                     # To reduce, only consider j, k with high weights.
                     # Let's just do a limited search: try top 50 weights for j and k?
                     # Or just full search if set is small.
                     
                     # Let's do a full search but optimized:
                     # Sort indices by weight desc.
                     # We already have all_indices.
                     
                     # For efficiency, we can break if we find a good swap, 
                     # but we want the BEST swap to climb faster.
                     # Given time limits, finding ANY improving swap is good enough for hill climbing.
                     # Let's find the first improving 1-for-2 swap.
                     
                     found_swap = False
                     # Iterate j
                     for j in all_indices:
                         if j in current_set:
                             continue
                         # Check j valid against S\{i}
                         valid_j = True
                         for k in current_set:
                             if k == i:
                                 continue
                             if conflict[j][k]:
                                 valid_j = False
                                 break
                         if not valid_j:
                             continue
                         
                         # Iterate k
                         for k in all_indices:
                             if k == j or k in current_set:
                                 continue
                             # Check k valid against S\{i}
                             valid_k = True
                             for m in current_set:
                                 if m == i:
                                     continue
                                 if conflict[k][m]:
                                     valid_k = False
                                     break
                             if not valid_k:
                                 continue
                             
                             # Check j, k valid against each other
                             if conflict[j][k]:
                                 continue
                             
                             gain = weights[j] + weights[k] - weights[i]
                             if gain > 0:
                                 # Found a valid improving swap
                                 best_swap_gain = gain
                                 best_swap_type = '1-2'
                                 best_swap_params = (i, j, k)
                                 found_swap = True
                                 break
                         if found_swap:
                             break
                     
                     if found_swap:
                         break # Found best 1-for-2 for this i? No, we found the first. 
                         # To be better, we could check all i's, but this is a heuristic.
                         # Let's break to apply the swap.
            
            if best_swap_gain > 0:
                if best_swap_type == '1-1':
                    i, j = best_swap_params
                    current_set.remove(i)
                    current_set.add(j)
                elif best_swap_type == '1-2':
                    i, j, k = best_swap_params
                    current_set.remove(i)
                    current_set.add(j)
                    current_set.add(k)
                
                current_score += best_swap_gain
                improved = True
                continue # Restart loop
            
            # If no 1-1 or 1-2 swap found, try 2-for-1?
            # 2-for-1: Remove i, j. Add k.
            # Gain: w(k) - w(i) - w(j) > 0
            # This is harder to find.
            
            # If no improvement, break.
            break
            
        return current_set, current_score

    # Multi-start
    # Start 1: Greedy from scratch (smallest first)
    init_set_1 = set()
    for i in sorted(range(n), key=lambda x: domain[x]):
        if is_valid_addition(i, init_set_1):
            init_set_1.add(i)
    
    # Start 2: Greedy skipping 2
    init_set_2 = set()
    for i in sorted(range(n), key=lambda x: domain[x]):
        if domain[i] == 2:
            continue
        if is_valid_addition(i, init_set_2):
            init_set_2.add(i)
            
    # Start 3: Greedy skipping 2, 3
    init_set_3 = set()
    for i in sorted(range(n), key=lambda x: domain[x]):
        if domain[i] in (2, 3):
            continue
        if is_valid_addition(i, init_set_3):
            init_set_3.add(i)

    # Start 4: Random
    random.seed(42)
    init_set_4 = set()
    # Randomly add elements if valid
    indices_copy = list(range(n))
    random.shuffle(indices_copy)
    for i in indices_copy:
        if is_valid_addition(i, init_set_4):
            init_set_4.add(i)
            
    # Start 5: Random with bias towards smaller numbers
    init_set_5 = set()
    indices_copy = list(range(n))
    # Weighted shuffle? Just sort by random + small penalty
    random.shuffle(indices_copy)
    for i in indices_copy:
        if is_valid_addition(i, init_set_5):
            init_set_5.add(i)

    initial_sets = [init_set_1, init_set_2, init_set_3, init_set_4, init_set_5]
    
    # Run local search on each
    for init_set in initial_sets:
        if time.time() > deadline - 5:
            break
            
        final_set, final_score = local_search(init_set)
        
        if final_score > best_score:
            best_score = final_score
            best_set = final_set
            
    # Final verification and conversion to list of values
    # Verify validity
    valid = True
    for i in best_set:
        for j in best_set:
            if i < j and conflict[i][j]:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid set: just {256}? Or {2}?
        # {2} is always valid.
        best_set = {0} # Index 0 is value 2
        
    return [domain[i] for i in best_set]
