<<<<<<< SEARCH
def construct():
    """Return the longest verified run of consecutive integers with distinct totients.
    
    Returns a dict with 'start' and 'length' keys.
    """
    best_start = 1
    best_length = 1
    
    # Verify each known run from the table, keeping the longest valid one.
    # We iterate in order of increasing length so the last valid one wins.
    for start, length in _KNOWN_RUNS:
        if start + length - 1 > 10**12:
            continue
        if _verify_run(start, length):
            if length > best_length:
                best_length = length
                best_start = start
    
    # Fallback: if no known run verified (shouldn't happen), return a trivial run.
    if best_length < 1:
        best_start = 1
        best_length = 1
    
    return {'start': best_start, 'length': best_length}
=======
def construct():
    """Return the longest verified run of consecutive integers with distinct totients.
    
    Returns a dict with 'start' and 'length' keys.
    """
    best_start = 1
    best_length = 1
    
    # Search for the longest run in a reasonable range.
    # We search up to 1,000,000. This is fast enough (< 2s).
    # If a longer run exists at higher numbers, we might miss it, 
    # but small ranges often contain the record holders for small k.
    limit = 1000000
    
    for s in range(1, limit + 1):
        # We only need to check if we can beat the current best_length.
        # If we find a collision before reaching best_length, we stop.
        # If we reach best_length, we continue to see if it's longer.
        
        seen = set()
        k = 0
        # We check up to best_length + 10 to allow for some extension
        # But strictly, we just need to find the max for this s.
        # Optimization: if s + best_length > limit, we can't have a run 
        # longer than limit - s. But we are finding max in range.
        
        # We iterate until collision or we are sure we can't beat best_length?
        # No, we need the exact length to update best.
        # But if k < best_length and we hit collision, we can stop.
        # If k == best_length and we hit collision, we stop (length is best_length, not >).
        # If k > best_length, we update and continue.
        
        for i in range(best_length + 50): # Buffer
            n = s + i
            if n > 10**12:
                break
            v = _totient(n)
            if v in seen:
                break
            seen.add(v)
            k += 1
            
            if k > best_length:
                best_length = k
                best_start = s
                
        # Early termination if we can't possibly beat best_length?
        # If s + best_length > limit, we can't find a run longer than limit-s.
        # But we are searching for global max.
        # If limit is 1e6, and best_length is 10, we check 10 numbers.
        
    return {'start': best_start, 'length': best_length}
>>>>>>> REPLACE
