"""Search for large sets with divisibility-incomparable subset sums."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _is_valid_subset_sums(sums_set):
    """Check that no two distinct sums in the set divide one another."""
    s = sorted(sums_set)
    n = len(s)
    # For each pair, check divisibility. Since sums are positive ints.
    for i in range(n):
        a = s[i]
        # Only need to check multiples of a in the set
        # For efficiency, check if any s[j] is a multiple of a (j > i)
        for j in range(i + 1, n):
            if s[j] % a == 0:
                return False
    return True


def _subset_sums(values):
    """Compute all non-empty subset sums of the list."""
    sums = set()
    n = len(values)
    # Use iterative method
    for v in values:
        new_sums = set()
        for s in sums:
            new_sums.add(s + v)
        new_sums.add(v)
        sums |= new_sums
    return sums


def _validate_set(vals):
    """Full validation: all subset sums unique (no two subsets same sum) and no two divide."""
    # First check: no two distinct subsets have the same sum.
    # This is equivalent to: all subset sums are distinct for distinct subsets.
    # We can check by generating all subset sums with a count.
    n = len(vals)
    # Use a dict: sum -> count
    sum_count = {}
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += vals[i]
        sum_count[s] = sum_count.get(s, 0) + 1
    # If any sum appears more than once, invalid
    for s, c in sum_count.items():
        if c > 1:
            return False
    # Now check divisibility among distinct sums
    sums_set = set(sum_count.keys())
    return _is_valid_subset_sums(sums_set)


def _validate_set_fast(vals):
    """Faster validation using incremental subset sum generation."""
    n = len(vals)
    # Generate all subset sums, tracking if any duplicate
    sums = set()
    # We'll use a list of current sums and add one element at a time
    # But we need to detect duplicates.
    # Alternative: use a dict sum->count, but that's memory heavy for large n.
    # For n up to ~15, 2^n is manageable.
    
    # For small n, just do the mask approach but with early exit
    # Actually, let's use the incremental approach with a set and check for duplicates
    current_sums = set()
    for v in vals:
        new_sums = set()
        for s in current_sums:
            ns = s + v
            if ns in current_sums or ns in new_sums:
                # Duplicate sum found -> invalid
                return False
            new_sums.add(ns)
        if v in current_sums or v in new_sums:
            return False
        new_sums.add(v)
        current_sums |= new_sums
    
    # Now check divisibility
    return _is_valid_subset_sums(current_sums)


def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Strategy: Start with a known good set and try to grow it.
    # Use greedy + local search.
    
    # Candidate pool: numbers in 1..256
    # We want to find a set of size up to 30.
    # Key insight: if we use powers of 2, subset sums are all distinct (binary representation).
    # But powers of 2 divide each other: 2^i divides 2^j for i<j.
    # So we need numbers that don't create divisible subset sums.
    
    # A good approach: use numbers that are "large" and "incomparable" in terms of divisibility.
    # For example, primes near 256 won't divide each other, but their sums might.
    
    # Let's try a different approach: use numbers that are pairwise "sum-incomparable".
    # Actually, the constraint is on ALL subset sums, not just the elements.
    
    # Heuristic: start with a set of numbers that are all in a range where sums don't divide.
    # If all numbers are > 128, then any subset sum is > 128, and the maximum sum is at most 30*256=7680.
    # But smaller sums could divide larger ones.
    
    # Better heuristic: use numbers that are all close to each other, so sums are in a narrow range.
    # If all numbers are in [a, b], then subset sums of k elements are in [k*a, k*b].
    # For two subset sums to divide, we need one to be a multiple of the other.
    # If the range is narrow, it's harder for one sum to be a multiple of another.
    
    # Let's try: pick numbers from a narrow range, say [100, 256].
    # Start with a greedy construction.
    
    # First, let's find a good base set.
    # Try random restarts with greedy insertion.
    
    best_set = []
    best_size = 0
    
    random.seed(42)
    
    # Candidate numbers: 1..256
    candidates = list(range(1, 257))
    
    # Try multiple random orderings
    for trial in range(200):
        if time.time() > deadline:
            break
        
        random.shuffle(candidates)
        current = []
        current_sums = set()
        
        for v in candidates:
            if time.time() > deadline:
                break
            
            # Try adding v
            # Check if v alone causes issues (it shouldn't if v not in current_sums)
            # Check new sums
            new_sums = set()
            valid = True
            for s in current_sums:
                ns = s + v
                if ns in current_sums or ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            if not valid:
                continue
            if v in current_sums or v in new_sums:
                continue
            new_sums.add(v)
            
            # Check divisibility among all sums (current_sums | new_sums)
            all_sums = current_sums | new_sums
            if not _is_valid_subset_sums(all_sums):
                continue
            
            current.append(v)
            current_sums = all_sums
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
        
        # Early termination if we hit 30
        if best_size >= 30:
            break
    
    # Try to improve with local search
    if time.time() < deadline - 5.0:
        # Try swapping elements
        for iteration in range(50):
            if time.time() > deadline:
                break
            
            improved = False
            # Try removing one element and adding another
            for i in range(len(best_set)):
                if time.time() > deadline:
                    break
                for v in range(1, 257):
                    if v in best_set:
                        continue
                    # Try replacing best_set[i] with v
                    new_set = best_set[:i] + [v] + best_set[i+1:]
                    if _validate_set_fast(new_set):
                        best_set = new_set
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    # Ensure we return a valid set
    if not best_set:
        best_set = [31, 32, 34]
    
    # Final validation
    if not _validate_set(best_set):
        # Fallback to a known valid set
        best_set = [31, 32, 34]
    
    return list(best_set)

# EVOLVE-BLOCK-END
