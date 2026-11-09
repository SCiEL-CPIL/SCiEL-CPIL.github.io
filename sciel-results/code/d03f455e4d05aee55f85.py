import time
import sys

START_TIME = time.time()
DEADLINE = 60.0  # seconds, safety margin before 90s kill

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def check_validity(A):
    """Check if set A satisfies: no sum is achieved by subsets of different cardinalities."""
    n = len(A)
    if n == 0:
        return True
    # sums_by_k[k] = set of achievable sums using exactly k elements
    # Use a list of sets; sums_by_k[0] = {0}
    max_sum = sum(A)
    # For efficiency, use a list of bytearrays or sets
    # sums_by_k[k] is a set of ints
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0] = {0}
    
    for a in A:
        # Update in reverse to avoid using a twice
        for k in range(n - 1, -1, -1):
            if sums_by_k[k]:
                for s in sums_by_k[k]:
                    sums_by_k[k + 1].add(s + a)
    
    # Check for conflicts: no sum appears in two different cardinality sets
    seen = {}  # sum -> cardinality
    for k in range(n + 1):
        for s in sums_by_k[k]:
            if s in seen and seen[s] != k:
                return False
            seen[s] = k
    return True

def check_validity_fast(A):
    """Faster check using the incremental property."""
    n = len(A)
    if n <= 1:
        return True
    # Build sums_by_k incrementally and check conflicts as we go
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0] = {0}
    current_max_k = 0
    
    for a in A:
        # New sums for cardinality k+1 come from sums_by_k[k] + a
        # We need to check if any of these conflict with existing sums in other cardinalities
        # Build new entries
        new_entries = {}  # k -> set of new sums
        for k in range(current_max_k, -1, -1):
            if sums_by_k[k]:
                new_sums = set()
                for s in sums_by_k[k]:
                    ns = s + a
                    new_sums.add(ns)
                new_entries[k + 1] = new_sums
        
        # Check conflicts: for each new sum in cardinality k, ensure it's not in any other cardinality's existing sums
        for k, new_sums in new_entries.items():
            for ns in new_sums:
                # Check if ns is already in sums_by_k[j] for any j != k
                for j in range(n + 1):
                    if j != k and ns in sums_by_k[j]:
                        return False
        # Also check within the same new set - but same cardinality is fine
        
        # Now add the new sums to sums_by_k
        for k, new_sums in new_entries.items():
            sums_by_k[k].update(new_sums)
            if k > current_max_k:
                current_max_k = k
    
    return True

def construct():
    """Construct a set of distinct integers in 1..256 with cardinality-separated subset sums."""
    if is_done():
        return [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Start with powers of 2 as a known valid set
    base = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Try to add more elements greedily
    # Candidate elements: all ints in 1..256 not in base
    candidates = [x for x in range(1, 257) if x not in set(base)]
    
    # Sort candidates by some heuristic - maybe try smaller ones first, or larger ones
    # Larger elements are less likely to cause conflicts
    candidates.sort(reverse=True)
    
    current = list(base)
    current_set = set(base)
    
    # Try adding candidates one by one
    for c in candidates:
        if is_done():
            break
        if c in current_set:
            continue
        test = current + [c]
        if check_validity_fast(test):
            current = test
            current_set.add(c)
    
    # If we have time, try a second pass with remaining candidates
    if not is_done():
        remaining = [x for x in range(1, 257) if x not in current_set]
        remaining.sort(reverse=True)
        for c in remaining:
            if is_done():
                break
            test = current + [c]
            if check_validity_fast(test):
                current = test
                current_set.add(c)
    
    # Ensure we return a list of Python ints
    result = [int(x) for x in current]
    return result
