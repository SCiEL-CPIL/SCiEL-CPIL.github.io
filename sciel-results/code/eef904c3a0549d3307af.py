import itertools
import math
import time

def construct():
    t_start = time.time()
    TIME_LIMIT = 85.0
    
    # Precompute gcd table for values up to 256
    max_sum = 256
    gcd_table = [[0] * (max_sum + 1) for _ in range(max_sum + 1)]
    for i in range(1, max_sum + 1):
        for j in range(1, max_sum + 1):
            gcd_table[i][j] = math.gcd(i, j)
    
    def is_pairwise_coprime(values):
        """Check if all distinct values are pairwise coprime."""
        distinct = list(set(values))
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                if gcd_table[distinct[i]][distinct[j]] > 1:
                    return False
        return True
    
    best_count = 0
    best_result = [[1, 7, 13], [4, 10, 16]]  # fallback
    
    # Try all combinations of 3 distinct elements from [1, 128]
    # This is C(128,3)^2 which is too large, so we need to be smarter
    
    # Strategy: 
    # 1. First try to find sets where all 9 sums are pairwise coprime
    # 2. If not possible, find sets with as many pairwise coprime sums as possible
    
    # Key observation: if a sum is even, it shares factor 2 with any other even sum
    # So at most one sum can be even (or all sums must be odd, but sum of two odds is even)
    # Actually: odd+odd=even, odd+even=odd, even+even=even
    # So to have all sums odd, we need one set all odd and other all even, or vice versa
    # But then all sums are odd, and we need them pairwise coprime
    
    # Let's try: A all odd, B all even (or vice versa) to get all odd sums
    # Or mix to get a good set of coprime numbers
    
    # Approach: iterate over possible A sets, then for each, find best B set
    # To keep it tractable, use heuristics
    
    # First, let's try a more targeted search
    # Use numbers that are likely to produce coprime sums
    
    # Precompute: for each pair (a, b), check if a+b is "good"
    # A number n is "good" if it's coprime to many other numbers in [2, 256]
    
    # Let's just do a randomized + local search approach with time budget
    
    import random
    random.seed(42)
    
    def score_pair(A, B):
        """Return (num_distinct, all_coprime) for the pair."""
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        count = len(distinct)
        coprime = is_pairwise_coprime(distinct)
        return count, coprime
    
    # Phase 1: Random search
    candidates = []
    
    # Try structured sets first
    # Sets with small numbers tend to have more overlap
    # Sets with spread out numbers tend to have more distinct sums
    
    # Let's try: one set with small consecutive, other with larger spread
    # Or both with prime-like properties
    
    # Generate candidate sets
    all_vals = list(range(1, 129))
    
    # Try some heuristic sets first
    heuristic_sets = [
        [1, 2, 3], [1, 2, 4], [1, 3, 5], [1, 4, 7], [1, 5, 9],
        [2, 3, 5], [2, 5, 7], [3, 5, 7], [3, 7, 11], [5, 7, 11],
        [1, 2, 100], [1, 50, 100], [10, 50, 100], [20, 60, 100],
        [1, 10, 100], [2, 10, 100], [1, 20, 100], [5, 15, 105],
    ]
    
    # Add random sets
    for _ in range(200):
        heuristic_sets.append(random.sample(all_vals, 3))
    
    # For each pair of heuristic sets, evaluate
    for A in heuristic_sets:
        if time.time() - t_start > TIME_LIMIT - 10:
            break
        for B in heuristic_sets:
            if time.time() - t_start > TIME_LIMIT - 10:
                break
            count, coprime = score_pair(A, B)
            if coprime:
                if count > best_count:
                    best_count = count
                    best_result = [sorted(A), sorted(B)]
    
    # Phase 2: Local search around best result
    if best_count < 9:
        A_curr = list(best_result[0])
        B_curr = list(best_result[1])
        
        for iteration in range(10000):
            if time.time() - t_start > TIME_LIMIT - 5:
                break
            
            # Try modifying one element in A or B
            improved = False
            
            for set_idx in [0, 1]:
                if set_idx == 0:
                    curr_set = A_curr
                    other_set = B_curr
                else:
                    curr_set = B_curr
                    other_set = A_curr
                
                for pos in range(3):
                    old_val = curr_set[pos]
                    for new_val in all_vals:
                        if new_val == old_val:
                            continue
                        if new_val in curr_set:
                            continue
                        
                        new_set = list(curr_set)
                        new_set[pos] = new_val
                        
                        if set_idx == 0:
                            count, coprime = score_pair(new_set, B_curr)
                        else:
                            count, coprime = score_pair(A_curr, new_set)
                        
                        if coprime and count > best_count:
                            best_count = count
                            if set_idx == 0:
                                best_result = [sorted(new_set), sorted(B_curr)]
                            else:
                                best_result = [sorted(A_curr), sorted(new_set)]
                            A_curr = list(best_result[0])
                            B_curr = list(best_result[1])
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved and iteration > 100:
                # Random restart
                A_curr = random.sample(all_vals, 3)
                B_curr = random.sample(all_vals, 3)
                count, coprime = score_pair(A_curr, B_curr)
                if coprime and count > best_count:
                    best_count = count
                    best_result = [sorted(A_curr), sorted(B_curr)]
    
    return best_result
