<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 55  # 60s limit, keep 5s margin
    
    # Phase 1: Greedy removal from dense upper pool
    best = _greedy_removal(120, 192)
    best_size = len(best)
    
    # Phase 2: Try different pool starts if time permits
    if time.time() < deadline - 20:
        for ps in [100, 80, 60, 140]:
            if time.time() >= deadline - 20:
                break
            candidate = _greedy_removal(ps, 192)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(candidate)
    
    # Phase 3: Local search to add more elements
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        best = _local_search(best, remaining)
        best_size = len(best)
        
    # Final validation
    if not _is_valid_set(best):
        best = _greedy_removal(144, 192)
        
    return best
=======
def construct():
    start_time = time.time()
    deadline = start_time + 55  # 60s limit, keep 5s margin
    
    # Phase 1: Greedy construction from dense upper pool (descending order favors larger numbers)
    best = _greedy_construction(120, 192, 'desc')
    best_size = len(best)
    
    # Also try ascending order? Large numbers are safer, so desc is likely better.
    # But maybe mixing is good.
    
    # Phase 2: Try different pool starts if time permits
    if time.time() < deadline - 20:
        for ps in [100, 80, 60, 140, 150]:
            if time.time() >= deadline - 20:
                break
            candidate = _greedy_construction(ps, 192, 'desc')
            if len(candidate) > best_size:
                best = candidate
                best_size = len(candidate)
    
    # Phase 3: Local search to add more elements
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        best = _local_search(best, remaining)
        best_size = len(best)
        
    # Final validation
    if not _is_valid_set(best):
        # Fallback: try a very small safe set
        best = _greedy_construction(180, 192, 'desc')
        
    return best
>>>>>>> REPLACE

I should also make `_local_search` slightly more robust. The current one is okay, but I'll ensure it doesn't get stuck in bad swaps. The current implementation looks reasonable.

One more thing: `_valid_addition_fast` creates a meshgrid for every call. This is expensive.
I can optimize `_valid_addition_fast` by precomputing triple indices or just using a simpler loop if N is small. But N is up to 48.
Meshgrid of 48x48x48 is ~100k elements. That's fast.
The pair check is 48x48 ~ 2k elements.
So `_valid_addition_fast` is fast enough.

The main gain is replacing the slow Python-loop `_greedy_removal` with the NumPy-backed `_greedy_construction`.

Let's double check the `_valid_addition_fast` logic.
Condition 1: x divides sum of 3 others.
Condition 2: a divides x + u + v.

This is correct.

I will also add a small optimization to `_local_search` to prioritize adding candidates that are "large" or "safe". The current candidate list is sorted reverse, which is good.

One potential issue: `_greedy_construction` iterates through candidates. If it fails to add many, it's fine.
The `construct` function tries multiple pools.

Let's refine the pool starts. 120-192 is 73 elements.
100-192 is 93 elements.
80-192 is 113 elements.
60-192 is 133 elements.
Smaller numbers are more dangerous (more likely to divide sums).
So starting with larger numbers is better.
Maybe 150-192 first?
Let's try 150, 120, 100, 80.

I'll update the pool starts in `construct`.

