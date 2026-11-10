import time
import itertools

def calc_score(A):
    """
    Compute the score for a set A: max(|A+A|, |A*A|).
    """
    if not A:
        return 0
    A = list(A)
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    return max(len(sumset), len(prodset))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s/90s limit
    
    # Initial candidates to seed the greedy search
    # 1. Arithmetic Progression
    # 2. Small integers
    # 3. Powers of 2 (small product set, large sum set)
    # 4. Smooth numbers
    
    initial_sets = [
        list(range(1, 21)),
        list(range(1, 11)), # Start smaller, expand
        [2**i for i in range(5)], # Start with powers of 2
    ]
    
    best_set = None
    best_score = float('inf')
    
    # We will perform a greedy expansion from a few different seeds
    # To save time, we'll limit the search space for new elements to a reasonable range
    # and check candidates in a structured way.
    
    def greedy_expand(seed_set, max_iterations=100):
        A = list(seed_set)
        A.sort()
        current_score = calc_score(A)
        
        # If we already have 20 elements, just return
        if len(A) >= 20:
            return A, current_score
            
        # Candidates to consider for addition
        # We look at small integers and numbers related to existing ones
        # To be efficient, we don't check all 1..10000 every time.
        # We check a window of small numbers and multiples.
        
        while len(A) < 20:
            if time.time() > deadline:
                break
                
            best_candidate = None
            best_candidate_score = float('inf')
            
            # Generate candidate pool
            # 1. Small integers 1..50
            # 2. Multiples of existing elements by small factors
            candidates = set(range(1, 51))
            for a in A:
                for f in [2, 3, 4, 5, 6]:
                    val = a * f
                    if val <= 10000:
                        candidates.add(val)
            # Also try adding the next few integers after max(A)
            for i in range(max(A) + 1, max(A) + 10):
                candidates.add(i)
                
            # Filter out existing
            A_set = set(A)
            candidates = [c for c in candidates if c not in A_set]
            
            if not candidates:
                # If no candidates found in pool, expand range
                for i in range(51, 100):
                    if i not in A_set:
                        candidates.append(i)
                        
            for c in candidates:
                # Quick prune: if c is too large, product set might explode
                # but sum set might stay small.
                new_A = A + [c]
                # Calculate score
                # Optimization: We can update sets incrementally, but for n=20, 
                # full recalculation is O(400) which is fast.
                s_set = set(a + b for a in new_A for b in new_A)
                p_set = set(a * b for a in new_A for b in new_A)
                score = max(len(s_set), len(p_set))
                
                if score < best_candidate_score:
                    best_candidate_score = score
                    best_candidate = c
            
            if best_candidate is None:
                break
                
            A.append(best_candidate)
            A.sort()
            current_score = best_candidate_score
            
        return A, current_score

    # Run greedy expansion from different seeds
    for seed in initial_sets:
        if time.time() > deadline:
            break
            
        # Ensure seed is valid and size < 20
        if len(seed) > 20:
            seed = seed[:20]
            
        # Remove duplicates and sort
        seed = sorted(list(set(seed)))
        
        expanded_set, score = greedy_expand(seed)
        
        if score < best_score:
            best_score = score
            best_set = expanded_set
            
    # If we ran out of time or didn't get a good result, ensure we have a valid set of 20
    if not best_set or len(best_set) != 20:
        # Fallback to a known decent heuristic: 
        # Try to find a set with low max sum/prod
        # A common good structure is small integers.
        best_set = list(range(1, 21))
        
    # Final check: ensure exactly 20 distinct positive integers <= 10000
    best_set = sorted(list(set(best_set)))
    if len(best_set) < 20:
        # Pad with large numbers that don't increase sumset too much?
        # Actually, just use 1..20 if we failed
        best_set = list(range(1, 21))
    elif len(best_set) > 20:
        best_set = best_set[:20]
        
    return best_set
