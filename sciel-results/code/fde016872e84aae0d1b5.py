import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Seed solution
    best_set = [0, 1, 3]
    best_score = -1.0
    
    def calc_score(U):
        if not U or 0 not in U:
            return -1.0
        U = sorted(set(U))
        if U[0] != 0:
            return -1.0
        max_u = U[-1]
        if max_u == 0:
            return 1.0
        denom = math.log2(2 * max_u + 1)
        if denom == 0:
            return 1.0
        
        sum_set = set()
        diff_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
                diff_set.add(a - b)
        
        ratio = len(diff_set) / len(sum_set)
        if ratio <= 1:
            return 1.0
        
        score = 1 + math.log2(ratio) / denom
        return score

    current_score = calc_score(best_set)
    best_score = current_score
    
    # Heuristic Search
    # Try to improve by adding elements
    # We limit the size to keep computation fast
    max_size = 15
    
    # Initialize with seed
    U = list(best_set)
    
    # Greedy expansion
    while len(U) < max_size:
        if time.time() > deadline - 10:
            break
        
        best_candidate = None
        best_candidate_score = best_score
        
        # Try adding numbers from a range
        # The next number should be larger than current max
        max_u = max(U)
        
        # Search range for new element
        # Usually, to keep sumset small, we add large gaps.
        # To keep diffset large relative to sumset, we want unique differences.
        # Let's try a range of potential new elements
        search_range = range(max_u + 1, max_u * 3 + 10)
        
        # To speed up, we can sample or use a step
        # But N is small, so full search is okay for N<15
        for new_val in search_range:
            candidate = U + [new_val]
            score = calc_score(candidate)
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = new_val
        
        if best_candidate is not None:
            U.append(best_candidate)
            best_score = best_candidate_score
            best_set = list(U)
        else:
            # If no improvement in this range, try a wider range or break
            # Try a much larger jump
            jump = max_u * 2 + 1
            candidate = U + [jump]
            score = calc_score(candidate)
            if score > best_score:
                U.append(jump)
                best_score = score
                best_set = list(U)
            else:
                break
                
    # Local Search: Try swapping or removing elements
    # This is complex, so we stick to the greedy result if it's better
    
    # Final check
    final_score = calc_score(best_set)
    if final_score > best_score:
        best_score = final_score
        
    return best_set

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()
# EVOLVE-BLOCK-END

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Seed solution
    best_set = [0, 1, 3]
    
    def calc_score(U):
        if not U or 0 not in U:
            return -1.0
        U = sorted(set(U))
        if U[0] != 0:
            return -1.0
        max_u = U[-1]
        if max_u == 0:
            return 1.0
        denom = math.log2(2 * max_u + 1)
        if denom == 0:
            return 1.0
        
        sum_set = set()
        diff_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
                diff_set.add(a - b)
        
        ratio = len(diff_set) / len(sum_set)
        if ratio <= 1:
            return 1.0
        
        score = 1 + math.log2(ratio) / denom
        return score

    best_score = calc_score(best_set)
    
    # Heuristic Search
    max_size = 12 # Keep small to ensure speed and high ratio
    
    U = list(best_set)
    
    # Greedy expansion
    while len(U) < max_size:
        if time.time() > deadline - 10:
            break
        
        best_candidate = None
        best_candidate_score = best_score
        
        max_u = max(U)
        
        # Search for the next element
        # We want to maximize the score.
        # Adding a large number increases the denominator significantly.
        # It also increases |U-U| and |U+U|.
        # If we add a number $x > 2 \max U$, then:
        # New sums: old sums + x, old sums + 2x, x, 2x.
        # New diffs: old diffs, x - U, U - x, 2x.
        # If x is large enough, the new sums and diffs are disjoint from old ones.
        # |U+U| new = |U+U| old + |U| + 2
        # |U-U| new = |U-U| old + 2|U| + 1
        
        # Let $S = |U+U|, D = |U-U|, N = |U|$.
        # New Ratio $R' = (D + 2N + 1) / (S + N + 2)$.
        # We want $R' > R$.
        
        # We iterate over possible x.
        # To keep denominator low, we want small x?
        # But x must be > max U.
        
        search_start = max_u + 1
        search_end = max_u + 50 # Limited range for speed
        
        for new_val in range(search_start, search_end):
            candidate = U + [new_val]
            score = calc_score(candidate)
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = new_val
        
        if best_candidate is not None:
            U.append(best_candidate)
            best_score = best_candidate_score
            best_set = list(U)
        else:
            break
                
    return best_set
