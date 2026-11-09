"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _check_tails(values, sums, x):
    """Check if adding x to values creates any duplicate block sum.
    Returns (valid, new_sums_list) where new_sums_list is the list of new tail sums."""
    n = len(values)
    total = x
    tails = [x]
    for i in range(n - 1, -1, -1):
        total += values[i]
        tails.append(total)
    # Check for conflicts
    for s in tails:
        if s in sums:
            return False, None
    return True, tails

def _greedy_fill(values, sums, start_val, deadline):
    """Continue greedy fill from current state, trying values from start_val upward."""
    for x in range(start_val, 257):
        if time.time() > deadline:
            break
        valid, tails = _check_tails(values, sums, x)
        if valid:
            values.append(x)
            sums.update(tails)
    return values, sums

def _try_local_search(seq, sums, deadline, max_iter=50):
    """Try to improve sequence by removing one element and re-filling."""
    best_seq = seq[:]
    best_sums = sums.copy()
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        n = len(best_seq)
        
        # Try removing each element
        for i in range(n):
            if time.time() > deadline:
                break
            
            removed_val = best_seq[i]
            base = best_seq[:i] + best_seq[i+1:]
            
            # Recompute sums for base
            base_sums = set()
            for j in range(len(base)):
                s = 0
                for k in range(j, len(base)):
                    s += base[k]
                    base_sums.add(s)
            
            # Re-fill from the point after the removed element
            # We need to try adding values that were after removed_val
            # Actually, let's just re-greedy the entire range, but keep base elements
            # Simpler: try to add any value from 1..256 not in base
            
            refill = base[:]
            refill_sums = base_sums.copy()
            
            # Try adding values in increasing order
            for x in range(1, 257):
                if time.time() > deadline:
                    break
                if x in refill:
                    continue
                valid, tails = _check_tails(refill, refill_sums, x)
                if valid:
                    refill.append(x)
                    refill_sums.update(tails)
            
            if len(refill) > len(best_seq):
                best_seq = refill
                best_sums = refill_sums
                improved = True
                break  # Restart the loop with the new best
    
    return best_seq, best_sums

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Strategy 1: Greedy smallest
    def greedy_smallest():
        values, sums = [], set()
        for x in range(1, 257):
            if time.time() > deadline:
                break
            valid, tails = _check_tails(values, sums, x)
            if valid:
                values.append(x)
                sums.update(tails)
        return values
    
    # Strategy 2: Greedy with lookahead
    def greedy_lookahead():
        values, sums = [], set()
        remaining = list(range(1, 257))
        while remaining and time.time() < deadline:
            best_candidate = None
            best_ext = None
            best_ext_sums = None
            best_length = 0
            
            for x in remaining:
                valid, tails = _check_tails(values, sums, x)
                if not valid:
                    continue
                
                new_sums = sums | set(tails)
                new_values = values + [x]
                
                # Quick greedy extension to estimate potential
                ext, ext_sums = _greedy_fill(new_values[:], new_sums.copy(), x + 1, deadline)
                
                if len(ext) > best_length:
                    best_length = len(ext)
                    best_candidate = x
                    best_ext = ext
                    best_ext_sums = ext_sums
            
            if best_candidate is None:
                break
            
            values = best_ext
            sums = best_ext_sums
            remaining = [c for c in remaining if c > values[-1]]
        
        return values
    
    # Strategy 3: Local search on greedy result
    def local_search_strategy():
        # Start with greedy smallest
        seq = greedy_smallest()
        if time.time() > deadline - 5:
            return seq
        
        # Compute sums
        sums = set()
        for i in range(len(seq)):
            s = 0
            for j in range(i, len(seq)):
                s += seq[j]
                sums.add(s)
        
        # Try to improve
        improved_seq, improved_sums = _try_local_search(seq, sums, deadline)
        return improved_seq
    
    # Run strategies, keep the best
    best_result = []
    
    strategies = [greedy_smallest, greedy_lookahead, local_search_strategy]
    for strategy in strategies:
        if time.time() > deadline - 5:
            break
        try:
            result = strategy()
            if len(result) > len(best_result):
                best_result = result
        except Exception:
            pass
    
    # If still not good enough, try random restarts
    if len(best_result) < 10 and time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            
            # Randomized greedy
            values, sums = [], set()
            remaining = list(range(1, 257))
            random.shuffle(remaining)
            
            for x in remaining:
                if time.time() > deadline - 5:
                    break
                if values and x <= values[-1]:
                    continue
                valid, tails = _check_tails(values, sums, x)
                if valid:
                    values.append(x)
                    sums.update(tails)
            
            if len(values) > len(best_result):
                best_result = values
    
    return best_result if best_result else greedy_smallest()
# EVOLVE-BLOCK-END
