import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _init_counts(s):
    """Calculate counts of APs for each difference d for a given set s."""
    counts = [0] * 256
    mask = 0
    for x in s:
        mask |= (1 << x)
    
    # For each d, count APs
    # A faster way to init:
    # For each pair (a, c) in S with c > a, let d = (c-a)/2.
    # If c-a is even, d is integer. Check if a+d in S.
    # This is O(N^2) = 32^2 = 1024 ops. Very fast.
    for i in range(len(s)):
        for j in range(i+1, len(s)):
            a = s[i]
            c = s[j]
            diff = c - a
            if diff % 2 == 0:
                d = diff // 2
                b = a + d
                if b in s:
                    counts[d] += 1
    return counts

def _score_from_counts(counts):
    """Calculate score from counts array."""
    score = 0
    for d in range(1, 256):
        if counts[d] > 0:
            score += 1
    return score

def _update_counts_remove(s, mask, i, counts, score):
    """Remove element i from set s, update counts and score. Returns new score."""
    # i is in s
    # Find all APs containing i
    for d in range(1, 256):
        # Case 1: i is the start (a)
        # a=i, b=i+d, c=i+2d
        if (i + d) in s and (i + 2 * d) in s:
            counts[d] -= 1
            if counts[d] == 0:
                score -= 1
        # Case 2: i is the middle (b)
        # a=i-d, b=i, c=i+d
        if (i - d) in s and (i + d) in s:
            counts[d] -= 1
            if counts[d] == 0:
                score -= 1
        # Case 3: i is the end (c)
        # a=i-2d, b=i-d, c=i
        if (i - 2 * d) in s and (i - d) in s:
            counts[d] -= 1
            if counts[d] == 0:
                score -= 1
    return score

def _update_counts_add(s, mask, j, counts, score):
    """Add element j to set s, update counts and score. Returns new score."""
    # j is NOT in s initially, but we assume s has been updated to include j? 
    # No, better to check against s BEFORE adding j? 
    # The APs formed by j must have the other two elements in s (which currently does NOT contain j).
    for d in range(1, 256):
        # Case 1: j is the start (a)
        # a=j, b=j+d, c=j+2d
        if (j + d) in s and (j + 2 * d) in s:
            counts[d] += 1
            if counts[d] == 1:
                score += 1
        # Case 2: j is the middle (b)
        # a=j-d, b=j, c=j+d
        if (j - d) in s and (j + d) in s:
            counts[d] += 1
            if counts[d] == 1:
                score += 1
        # Case 3: j is the end (c)
        # a=j-2d, b=j-d, c=j
        if (j - 2 * d) in s and (j - d) in s:
            counts[d] += 1
            if counts[d] == 1:
                score += 1
    return score

def _local_search(s, counts, score, deadline):
    """
    Perform local search with swaps.
    """
    current_s = set(s)
    current_counts = list(counts)
    current_score = score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all swaps
        # To speed up, we can randomize the order of i and j
        # Or just iterate all. 32 * 480 ~ 15k swaps per pass.
        # Each swap takes ~255 * 3 checks = ~750 ops.
        # 15k * 750 = 11M ops per pass.
        # Python might do 10M ops/sec? So ~1 sec per pass.
        # We have ~140s. So ~140 passes.
        
        # Optimization: Only try promising swaps?
        # For now, full scan.
        
        # Shuffle elements to avoid bias
        elems = list(current_s)
        random.shuffle(elems)
        
        for i in elems:
            if time.time() > deadline:
                break
            # Candidate j's are all numbers not in current_s
            # We can iterate 1..512 and skip if in current_s
            # Or precompute list of non-members
            # Non-members: 512 - 32 = 480
            
            # We can break early if we find an improvement?
            # Yes, first improvement hill climbing is fast.
            
            # To make it more robust, we can accept any improvement.
            # Let's just find the best swap in one pass? No, too slow.
            # First improvement is good.
            
            # Randomize j selection?
            # If we always pick j=1, we might bias towards low numbers.
            # Let's shuffle the range of j's?
            # Or just iterate 1..512.
            
            # Actually, iterating 1..512 for each i is 32*512 = 16k checks.
            # Let's just do it.
            
            for j in range(1, 513):
                if j in current_s:
                    continue
                
                # Simulate swap i -> j
                # Remove i
                temp_score = _update_counts_remove(current_s, 0, i, current_counts, current_score)
                
                # Note: _update_counts_remove assumes i is in s. It is.
                # But it doesn't actually remove i from s or mask.
                # We need to be careful. The checks `(i+d) in s` use the OLD s.
                # This is correct for removal: we check if other elements are in s.
                
                # Now add j
                # The checks `(j+d) in s` use the OLD s (which still contains i).
                # This is WRONG. The APs formed by j should only involve elements in S \ {i}.
                # So we must temporarily remove i from s, or pass a modified set.
                
                # This incremental approach is tricky with in-place sets.
                # Let's modify the functions to take a set and a "removed" element?
                # Or just do the full recalculation if the incremental is too complex/error-prone.
                
                # Given the small size, full recalculation of score from mask is ~255 ops.
                # 16k swaps * 255 ops = 4M ops per pass.
                # This is very fast in Python if optimized?
                # Let's stick to the mask-based scoring for simplicity and correctness, 
                # but optimize the loop.
                
                pass # We will abandon this incremental path for now due to complexity of state management in the inner loop.
        
        # Let's go back to mask-based scoring but with a smarter search.
        # The bottleneck is the Python loop in _score_mask.
        # Can we vectorize?
        # No, d changes the shift amount.
        
        # However, we can use the fact that we only need to check d up to 255.
        # And we can break early if we find many? No, we need the count.
        
        # Let's try a different strategy: 
        # 1. Generate a good initial set using greedy.
        # 2. Use Simulated Annealing with Swap moves.
        # 3. Score calculation:
        #    Precompute masks for shifts? No.
        #    Just use the bit trick.
        
    return current_s, current_score

def _fast_score(mask):
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _sa_search(s, initial_score, deadline):
    """
    Simulated Annealing with Swap moves.
    """
    current_s = set(s)
    current_mask = 0
    for x in current_s:
        current_mask |= (1 << x)
    current_score = initial_score
    
    best_s = set(current_s)
    best_score = current_score
    
    T_start = 10.0
    T_end = 0.01
    alpha = 0.999
    
    T = T_start
    step = 0
    while T > T_end and time.time() < deadline:
        # Pick random i in S
        i = random.choice(list(current_s))
        
        # Pick random j not in S
        # To avoid infinite loop, pick from 1..512
        j = random.randint(1, 512)
        if j in current_s:
            continue
        
        # Perform swap
        new_s = set(current_s)
        new_s.remove(i)
        new_s.add(j)
        
        new_mask = 0
        for x in new_s:
            new_mask |= (1 << x)
        
        new_score = _fast_score(new_mask)
        
        delta = new_score - current_score
        
        if delta > 0:
            current_s = new_s
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_s = set(current_s)
                best_score = current_score
        else:
            prob = np.exp(delta / T)
            if random.random() < prob:
                current_s = new_s
                current_mask = new_mask
                current_score = new_score
        
        T *= alpha
        step += 1
        
    return best_s, best_score

def _greedy_init():
    """
    Greedy construction.
    """
    s = set()
    mask = 0
    best_score = 0
    candidates = list(range(1, 513))
    random.shuffle(candidates)
    
    for _ in range(32):
        best_elem = None
        best_new_score = best_score
        for c in candidates:
            if c in s:
                continue
            new_mask = mask | (1 << c)
            new_score = _fast_score(new_mask)
            if new_score > best_new_score:
                best_new_score = new_score
                best_elem = c
        if best_elem is not None:
            s.add(best_elem)
            mask |= (1 << best_elem)
            best_score = best_new_score
        else:
            if candidates:
                c = candidates.pop()
                s.add(c)
                mask |= (1 << c)
                best_score = _fast_score(mask)
    return s, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    random.seed(42)
    np.random.seed(42)
    
    # Phase 1: Greedy Init
    s, score = _greedy_init()
    best_s, best_score = set(s), score
    
    # Phase 2: SA
    if time.time() < deadline - 10:
        s, score = _sa_search(best_s, best_score, deadline - 10)
        if score > best_score:
            best_s, best_score = s, score
            
    return sorted(list(best_s))

# EVOLVE-BLOCK-END
