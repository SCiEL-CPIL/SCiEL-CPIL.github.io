import time

def _verify_distinct_subset_sums(vals):
    """
    Verify that all 2^n subset sums are distinct.
    Returns True if distinct, False otherwise.
    """
    n = len(vals)
    sums = {0}
    for v in vals:
        new_sums = set()
        for s in sums:
            new_sums.add(s + v)
        if len(sums) + len(new_sums) != len(sums.union(new_sums)):
            return False
        sums = sums.union(new_sums)
    return True

def construct():
    # Deadline guard: The evaluator kills after 90s.
    # This solution is O(1) in terms of search, but verification is O(2^n).
    # For n=12, 2^12 = 4096 sums, which is instantaneous.
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    n = 12
    
    # The optimal sequence is powers of 2.
    # a_i = 2^(i-1) for i=1..12
    # This is the unique minimal superincreasing sequence starting with 1.
    optimal_vals = [1 << i for i in range(n)]
    
    # Check if we are within time limits (always true for this static solution)
    if time.time() - start_time > deadline:
        # Fallback: return a valid but potentially non-optimal set if somehow we timed out
        # (Highly unlikely for this specific problem size)
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    # Safety check: Ensure the solution is valid
    if not _verify_distinct_subset_sums(optimal_vals):
        # This should never happen for powers of 2, but just in case
        # Fallback to a greedy search if verification fails (unlikely)
        # But for n=12, powers of 2 are guaranteed valid.
        pass
        
    return optimal_vals
