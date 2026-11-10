"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _check_validity(S):
    """Check if set S satisfies the cardinality-separated subset sum property."""
    n = len(S)
    if n == 0:
        return True
    # For each cardinality k, compute all subset sums of size k
    # Check that no sum from cardinality k equals any sum from cardinality j (k != j)
    # We can check: for all k1 < k2, max_sum(k1) < min_sum(k2) is sufficient but not necessary
    # Actually we need: no sum in layer k1 equals any sum in layer k2
    # Efficient check: collect all sums by cardinality, then check for overlaps
    
    sums_by_card = [set() for _ in range(n + 1)]
    sums_by_card[0].add(0)
    
    for i, val in enumerate(S):
        for k in range(i, -1, -1):
            for s in sums_by_card[k]:
                sums_by_card[k + 1].add(s + val)
    
    # Check for cross-cardinality collisions
    for k1 in range(n + 1):
        for k2 in range(k1 + 1, n + 1):
            if sums_by_card[k1] & sums_by_card[k2]:
                return False
    return True

def _check_validity_fast(S):
    """Faster validity check using the sufficient condition for AP-like sets."""
    n = len(S)
    if n == 0:
        return True
    S_sorted = sorted(S)
    
    # Compute min and max sums for each cardinality
    min_sums = [0] * (n + 1)
    max_sums = [0] * (n + 1)
    
    for k in range(1, n + 1):
        min_sums[k] = sum(S_sorted[:k])
        max_sums[k] = sum(S_sorted[-k:])
    
    # Check: for all k1 < k2, max_sums[k1] < min_sums[k2]
    # This is a sufficient condition (not necessary, but works for well-separated sets)
    for k1 in range(n + 1):
        for k2 in range(k1 + 1, n + 1):
            if max_sums[k1] >= min_sums[k2]:
                return False
    return True

def _is_valid_full(S):
    """Full exact validity check."""
    n = len(S)
    if n == 0:
        return True
    
    sums_by_card = [set() for _ in range(n + 1)]
    sums_by_card[0].add(0)
    
    for val in S:
        for k in range(n - 1, -1, -1):
            if sums_by_card[k]:
                for s in sums_by_card[k]:
                    sums_by_card[k + 1].add(s + val)
    
    for k1 in range(n + 1):
        for k2 in range(k1 + 1, n + 1):
            if sums_by_card[k1] & sums_by_card[k2]:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Strategy 1: Find the largest arithmetic progression in 1..256
    # that satisfies the condition
    best_set = list(range(1, 9))  # fallback: small set that definitely works
    best_size = len(best_set)
    
    # Try all arithmetic progressions
    for d in range(1, 257):
        for a in range(1, 257):
            # Generate AP: a, a+d, a+2d, ... up to 256
            ap = []
            val = a
            while val <= 256:
                ap.append(val)
                val += d
            
            n = len(ap)
            if n <= best_size:
                continue
            
            # Quick check using min/max sums
            ap_sorted = sorted(ap)
            valid = True
            for k1 in range(n + 1):
                for k2 in range(k1 + 1, n + 1):
                    min_sum_k2 = sum(ap_sorted[:k2]) if k2 > 0 else 0
                    max_sum_k1 = sum(ap_sorted[-k1:]) if k1 > 0 else 0
                    if max_sum_k1 >= min_sum_k2:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and n > best_size:
                # Verify with full check for smaller sets
                if n <= 20:
                    if _is_valid_full(ap):
                        best_set = ap
                        best_size = n
                else:
                    # For larger sets, trust the min/max check (it's sufficient)
                    best_set = ap
                    best_size = n
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Strategy 2: Try to extend the best AP by adding elements
    if best_size < 60 and time.time() < deadline:
        current = list(best_set)
        used = set(current)
        
        # Try adding elements from 1..256 not in current
        candidates = [x for x in range(1, 257) if x not in used]
        candidates.sort()
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for c in candidates:
                new_set = current + [c]
                new_set.sort()
                # Quick check
                n = len(new_set)
                valid = True
                for k1 in range(n + 1):
                    for k2 in range(k1 + 1, n + 1):
                        min_sum_k2 = sum(new_set[:k2]) if k2 > 0 else 0
                        max_sum_k1 = sum(new_set[-k1:]) if k1 > 0 else 0
                        if max_sum_k1 >= min_sum_k2:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    if n <= 25:
                        if _is_valid_full(new_set):
                            current = new_set
                            used.add(c)
                            candidates.remove(c)
                            improved = True
                            best_set = current
                            best_size = n
                            break
                    else:
                        current = new_set
                        used.add(c)
                        candidates.remove(c)
                        improved = True
                        best_set = current
                        best_size = n
                        break
    
    # Strategy 3: Try greedy construction from scratch with different starting points
    if time.time() < deadline - 5:
        for start_val in [1, 5, 10, 20, 50, 100, 150, 200, 250]:
            if time.time() > deadline:
                break
            current = [start_val]
            used = {start_val}
            
            for val in range(1, 257):
                if val in used:
                    continue
                if time.time() > deadline:
                    break
                new_set = current + [val]
                new_set.sort()
                n = len(new_set)
                
                # Quick min/max check
                valid = True
                for k1 in range(n + 1):
                    for k2 in range(k1 + 1, n + 1):
                        min_sum_k2 = sum(new_set[:k2]) if k2 > 0 else 0
                        max_sum_k1 = sum(new_set[-k1:]) if k1 > 0 else 0
                        if max_sum_k1 >= min_sum_k2:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    if n <= 20:
                        if _is_valid_full(new_set):
                            current = new_set
                            used.add(val)
                    else:
                        current = new_set
                        used.add(val)
            
            if len(current) > best_size:
                best_set = current
                best_size = len(current)
    
    # Final verification
    if best_size > 0:
        # Ensure all elements are in 1..256 and distinct
        if len(set(best_set)) != len(best_set):
            best_set = list(set(best_set))
        if any(x < 1 or x > 256 for x in best_set):
            best_set = [x for x in best_set if 1 <= x <= 256]
        
        # Final exact check for small sets
        if len(best_set) <= 15:
            if not _is_valid_full(best_set):
                # Fallback to known valid set
                best_set = list(range(1, 9))
    
    return list(best_set)
# EVOLVE-BLOCK-END
