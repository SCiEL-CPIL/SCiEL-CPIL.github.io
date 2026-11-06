import time
import random
import math
import numpy as np

def _score_with_diffs(s):
    """
    Returns (score, set_of_diffs) for a set of integers s.
    Score is the number of distinct d > 0 such that a, a+d, a+2d are in s.
    """
    s_set = set(s)
    diffs = set()
    elements = sorted(s_set)
    n = len(elements)
    # O(n^2) is fine for n=32
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            b = elements[j]
            d = b - a
            c = b + d
            if c in s_set:
                diffs.add(d)
    return len(diffs), diffs

def _greedy_coverage_init(time_limit):
    """
    Construct an initial set using a greedy heuristic that maximizes new difference coverage.
    """
    current_set = set()
    current_diffs = set()
    remaining = list(range(1, 513))
    random.shuffle(remaining)
    
    # We need 32 elements.
    # Start with a few random ones to break symmetry, or just start empty.
    # Starting empty is safer for pure greedy logic.
    
    for _ in range(32):
        if time.time() > time_limit - 10:
            break
            
        best_elem = None
        best_new_count = -1
        best_diffs = None
        
        # To speed up, we might sample candidates if the set is large, 
        # but 512 is small enough to check all if we are careful, 
        # or we can sample a subset. Let's check all for the first few, 
        # then sample for speed if needed.
        
        # Optimization: Pre-compute potential diffs? 
        # Actually, checking 512 candidates * 32 steps * O(32^2) score calc is ~ 512*32*1000 = 16M ops. 
        # In Python, this might be slow. Let's sample candidates.
        
        if len(remaining) > 100:
            candidates = random.sample(remaining, 100)
        else:
            candidates = remaining
            
        for elem in candidates:
            trial_set = current_set | {elem}
            trial_score, trial_diffs = _score_with_diffs(trial_set)
            new_diffs = len(trial_diffs - current_diffs)
            
            if new_diffs > best_new_count:
                best_new_count = new_diffs
                best_elem = elem
                best_diffs = trial_diffs
                # Tie-breaking: prefer smaller elements? Or just first found.
                # If new_diffs is 0, we still need to pick something to fill the set.
        
        if best_elem is None:
            # Fallback if no candidate improved coverage (or all 0)
            best_elem = random.choice(remaining)
            trial_set = current_set | {best_elem}
            _, best_diffs = _score_with_diffs(trial_set)
            
        current_set.add(best_elem)
        current_diffs = best_diffs
        remaining.remove(best_elem)
        
    return current_set, len(current_diffs)

def _local_search(current_set, current_score, current_diffs, time_limit, T_init=5.0):
    """
    Simulated Annealing local search with explicit difference tracking.
    """
    best_set = set(current_set)
    best_score = current_score
    best_diffs = set(current_diffs)
    
    T = T_init
    cooling = 0.9995
    min_T = 0.01
    stuck = 0
    max_stuck = 2000
    
    elements_list = list(current_set)
    
    while time.time() < time_limit - 2:
        # Pick an element to remove
        idx_out = random.randint(0, 31)
        elem_out = elements_list[idx_out]
        
        # Pick an element to add
        # To make moves more targeted, we can bias towards numbers that are "close" 
        # to existing ones to form small d, or random for large d.
        # Let's stick to random for diversity, but maybe try a few candidates.
        
        best_move = None
        best_move_score = -1
        best_move_diffs = None
        best_elem_in = None
        
        # Sample candidates for the new element
        # Avoid picking an element already in the set
        available = [x for x in range(1, 513) if x not in current_set]
        
        # Limit candidates for speed
        if len(available) > 50:
            candidates = random.sample(available, 50)
        else:
            candidates = available
            
        for elem_in in candidates:
            # Create new set
            new_set = set(current_set)
            new_set.remove(elem_out)
            new_set.add(elem_in)
            
            new_score, new_diffs = _score_with_diffs(new_set)
            
            if new_score > best_move_score:
                best_move_score = new_score
                best_move_diffs = new_diffs
                best_elem_in = elem_in
                best_move = new_set
        
        if best_move is None:
            stuck += 1
            if stuck > max_stuck:
                break
            continue
            
        delta = best_move_score - current_score
        
        # Acceptance criterion
        accept = False
        if delta > 0:
            accept = True
        else:
            # Avoid division by zero or overflow issues
            if T > 0:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    accept = True
            
        if accept:
            current_set = best_move
            current_score = best_move_score
            current_diffs = best_move_diffs
            elements_list = list(current_set)
            stuck = 0
            
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
                best_diffs = set(current_diffs)
        else:
            stuck += 1
            
        T *= cooling
        if T < min_T:
            T = min_T
            
    return best_set, best_score, best_diffs

def construct():
    start_time = time.time()
    # Evaluator kills at 150s. We use 145s as a hard stop to be safe.
    deadline = start_time + 145.0
    
    best_global_set = None
    best_global_score = 0
    
    # Phase 1: Generate diverse high-quality initial sets
    # We will run a few different initialization strategies in parallel (sequentially, but distinct)
    
    # Strategy 1: Greedy Coverage
    if time.time() < deadline - 50:
        t_limit = time.time() + 5
        s1, sc1 = _greedy_coverage_init(t_limit)
        if sc1 > best_global_score:
            best_global_score = sc1
            best_global_set = s1
    
    # Strategy 2: Random + Hill Climb (from current program's spirit)
    if time.time() < deadline - 40:
        s2 = set(random.sample(range(1, 513), 32))
        sc2, _ = _score_with_diffs(s2)
        if sc2 > best_global_score:
            best_global_score = sc2
            best_global_set = s2
            
    # Strategy 3: Structured Initializations (Intervals, Clusters)
    # Interval
    s3 = set(range(1, 33))
    sc3, _ = _score_with_diffs(s3)
    if sc3 > best_global_score:
        best_global_score = sc3
        best_global_set = s3
        
    # Two Clusters
    c1 = random.randint(1, 100)
    c2 = random.randint(300, 400)
    s4 = set(c1 + i for i in range(16)) | set(c2 + i for i in range(16))
    sc4, _ = _score_with_diffs(s4)
    if sc4 > best_global_score:
        best_global_score = sc4
        best_global_set = s4

    # Phase 2: Iterative Refinement
    # We will take the best found so far and run local search.
    # Then we will try to restart from slightly perturbed versions of the best.
    
    if best_global_set is None:
        best_global_set = set(range(1, 33))
        best_global_score, _ = _score_with_diffs(best_global_set)
        
    current_best_set = set(best_global_set)
    current_best_score, current_best_diffs = _score_with_diffs(current_best_set)
    
    # Run local search on the best initial
    t_limit = time.time() + 20
    refined_set, refined_score, refined_diffs = _local_search(current_best_set, current_best_score, current_best_diffs, t_limit)
    
    if refined_score > best_global_score:
        best_global_score = refined_score
        best_global_set = refined_set
        current_best_diffs = refined_diffs

    # Phase 3: Multi-start from the best found
    # We will perform several rounds of:
    # 1. Take best_global_set
    # 2. Perturb it (swap 2-3 elements)
    # 3. Run local search
    # 4. Keep the best
    
    while time.time() < deadline - 10:
        # Create a perturbed version
        perturbed_set = set(best_global_set)
        num_swaps = random.randint(2, 5)
        
        for _ in range(num_swaps):
            elem_out = random.choice(list(perturbed_set))
            elem_in = random.randint(1, 512)
            if elem_in not in perturbed_set:
                perturbed_set.remove(elem_out)
                perturbed_set.add(elem_in)
                
        p_score, p_diffs = _score_with_diffs(perturbed_set)
        
        # Run local search on the perturbed set
        # Time budget per run: 5 seconds
        t_limit = time.time() + 5
        if time.time() >= deadline - 10:
            break
            
        res_set, res_score, res_diffs = _local_search(perturbed_set, p_score, p_diffs, t_limit, T_init=2.0)
        
        if res_score > best_global_score:
            best_global_score = res_score
            best_global_set = res_set
            current_best_diffs = res_diffs

    # Final Safety Check
    if best_global_set is None:
        return list(range(1, 33))
        
    # Ensure exactly 32 distinct integers in 1..512
    final_list = list(best_global_set)
    if len(final_list) != 32:
        # Fallback if something went wrong
        return list(range(1, 33))
        
    for x in final_list:
        if x < 1 or x > 512:
            return list(range(1, 33))
            
    return final_list

# EVOLVE-BLOCK-END
