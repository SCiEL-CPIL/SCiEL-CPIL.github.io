"""Program for maximizing 3-AP differences in a 32-element set within 1..512."""

import random
import time
import sys

def count_aps(s):
    """
    Calculate the number of distinct differences d > 0 such that
    a, a+d, a+2d are all in the set s.
    """
    s_set = set(s)
    diffs = set()
    for i in range(len(s)):
        a = s[i]
        for j in range(len(s)):
            if i == j: continue
            b = s[j]
            if b <= a: continue # a must be smallest
            d = b - a
            c = a + 2 * d
            if c in s_set:
                diffs.add(d)
    return len(diffs)

def greedy_construct():
    """
    Greedy construction: start with an empty set and add elements one by one,
    choosing the element that maximizes the number of new 3-AP differences created.
    Then refine with local search.
    """
    N = 32
    MAX_VAL = 512
    
    # Start with a structured initial set
    # Use arithmetic progressions as a base
    current_set = set()
    
    # Add elements greedily
    for _ in range(N):
        best_elem = None
        best_score = -1
        # Try a sample of candidate positions
        candidates = random.sample(range(1, MAX_VAL + 1), min(200, MAX_VAL))
        # Also try some structured positions
        for base in range(1, MAX_VAL, 16):
            for offset in range(0, 16):
                candidates.append(base + offset)
        candidates = list(set(candidates))
        
        for elem in candidates:
            if elem in current_set:
                continue
            test_set = current_set | {elem}
            score = count_aps(list(test_set))
            if score > best_score:
                best_score = score
                best_elem = elem
        
        if best_elem is not None:
            current_set.add(best_elem)
    
    return list(current_set)

def local_search(initial_set, time_limit=110.0):
    """
    Local search: try swapping elements to improve the score.
    """
    N = 32
    MAX_VAL = 512
    start_time = time.time()
    
    current_set = set(initial_set)
    current_score = count_aps(list(current_set))
    best_set = list(current_set)
    best_score = current_score
    
    no_improvement_count = 0
    max_no_improvement = 500
    
    while time.time() - start_time < time_limit and no_improvement_count < max_no_improvement:
        improved = False
        
        # Try all possible swaps
        for i in range(N):
            for j in range(i + 1, N):
                old_i = list(current_set)[i]
                old_j = list(current_set)[j]
                
                # Try replacing old_i with a new value
                new_vals = [random.randint(1, MAX_VAL) for _ in range(5)]
                # Also try values near existing elements
                for ref in current_set:
                    new_vals.append(ref + random.randint(-10, 10))
                    new_vals.append(ref - random.randint(-10, 10))
                
                for new_val in new_vals:
                    if new_val < 1 or new_val > MAX_VAL or new_val in current_set:
                        continue
                    
                    # Try swapping old_i with new_val
                    current_set.remove(old_i)
                    current_set.add(new_val)
                    new_score = count_aps(list(current_set))
                    
                    if new_score > current_score:
                        current_score = new_score
                        if new_score > best_score:
                            best_score = new_score
                            best_set = list(current_set)
                        improved = True
                        no_improvement_count = 0
                        break
                    else:
                        current_set.remove(new_val)
                        current_set.add(old_i)
                
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            no_improvement_count += 1
    
    return best_set

def construct():
    """
    Use greedy construction followed by local search to find a 32-element set
    in [1, 512] that maximizes the number of distinct 3-AP differences.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Try multiple greedy constructions with different random seeds
    best_overall = None
    best_overall_score = -1
    
    # Phase 1: Greedy construction
    for attempt in range(3):
        if time.time() - start_time > time_limit - 30:
            break
        greedy_set = greedy_construct()
        score = count_aps(greedy_set)
        if score > best_overall_score:
            best_overall_score = score
            best_overall = greedy_set
    
    if best_overall is None:
        # Fallback: random set
        best_overall = random.sample(range(1, 513), 32)
        best_overall_score = count_aps(best_overall)
    
    # Phase 2: Local search refinement
    if time.time() - start_time < time_limit - 5:
        refined_set = local_search(best_overall, time_limit=min(30.0, time_limit - (time.time() - start_time)))
        refined_score = count_aps(refined_set)
        if refined_score > best_overall_score:
            best_overall = refined_set
            best_overall_score = refined_score
    
    # Phase 3: More local search if time permits
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        refined_set2 = local_search(best_overall, time_limit=remaining_time - 5)
        refined_score2 = count_aps(refined_set2)
        if refined_score2 > best_overall_score:
            best_overall = refined_set2
            best_overall_score = refined_score2
    
    return best_overall
