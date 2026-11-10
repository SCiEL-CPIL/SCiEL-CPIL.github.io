# EVOLVE-BLOCK-START
import time
import random
import math
from functools import reduce
from fractions import Fraction
from collections import defaultdict

def _is_dissociated_fast(elems):
    """
    Check if all subset sums of reciprocals are distinct using meet-in-the-middle.
    Two subset sums collide iff there exist two different subsets with the same sum.
    This is equivalent to: there exist a non-trivial signed sum (coeffs in {-1,0,1})
    that equals zero.
    
    Split into two halves. For each half, compute all possible signed sums.
    Then check if any signed sum from half1 equals the negative of a signed sum from half2,
    where not both are the all-zero vector.
    """
    if not elems:
        return True
    if len(elems) == 1:
        return True
    
    n = len(elems)
    L = reduce(math.lcm, elems)
    weights = [L // e for e in elems]
    
    mid = n // 2
    half1 = weights[:mid]
    half2 = weights[mid:]
    
    def all_signed_sums(ws):
        """Generate all signed sums for a list of weights (each coeff in {-1, 0, 1})."""
        sums = [0]
        for w in ws:
            new_sums = []
            for s in sums:
                new_sums.append(s - w)
                new_sums.append(s)
                new_sums.append(s + w)
            sums = new_sums
        return sums
    
    sums1 = all_signed_sums(half1)
    sums2 = all_signed_sums(half2)
    
    # Build a frequency map for sums2
    freq2 = defaultdict(int)
    for s in sums2:
        freq2[s] += 1
    
    # For each sum in sums1, check if -sum exists in sums2
    # A collision exists if there's a pair (s1, s2) with s1 + s2 = 0
    # and not both s1 and s2 correspond to the all-zero assignment
    # The all-zero assignment gives sum 0 in both halves
    # So we need: count of pairs (s1, s2) with s1+s2=0 > 1 (since (0,0) is always one)
    
    total_pairs = 0
    for s1 in sums1:
        total_pairs += freq2[-s1]
    
    # total_pairs counts all pairs (s1, s2) with s1 + s2 = 0
    # The trivial pair is (0, 0) from all-zero assignments
    # But wait: multiple assignments can give sum 0 in a half
    # Actually, if any non-trivial signed sum equals 0, we have a collision
    # So: if total_pairs > 1, there's a collision (since (0,0) from all-zero is always present)
    # But actually, we need to be more careful.
    
    # A dissociated set means: the only signed sum that equals 0 is the all-zero one.
    # So we check: is there any non-zero signed combination that sums to 0?
    # Using MITM: for each s1 in sums1, for each s2 in sums2 with s1+s2=0,
    # the combined assignment is non-trivial unless both s1 and s2 come from all-zero.
    
    # Simpler: just check if 0 appears in the "difference" set
    # Actually, the cleanest way: the set is dissociated iff the only solution to
    # sum(c_i * w_i) = 0 with c_i in {-1,0,1} is c_i = 0 for all i.
    
    # So we just need: does there exist a non-trivial signed sum equal to 0?
    # Using MITM: for each s1 in sums1, check if -s1 is in sums2.
    # If s1=0 and -s1=0, that's the trivial case. We need to exclude it.
    
    # Let's count: how many ways can we get total sum 0?
    # It's sum over s1 of (count of s1 in sums1) * (count of -s1 in sums2)
    # The trivial all-zero gives one way. If total > 1, not dissociated.
    
    # But we already computed total_pairs above which is exactly this count.
    # Wait, no. total_pairs = sum_{s1 in sums1} freq2[-s1]
    # This counts each pair (assignment1, assignment2) such that sum1 + sum2 = 0.
    # The number of such pairs should be exactly 1 if dissociated (only all-zero).
    
    # Actually, let me reconsider. If the set is dissociated, then the only signed sum
    # that equals 0 is the all-zero one. So there should be exactly 1 pair (the all-zero
    # from half1 and all-zero from half2) that gives total 0.
    
    # But wait: within a single half, multiple assignments could give the same sum.
    # E.g., if half1 has weights [1, 2, 3], then 3-1-2=0, so sum 0 can be achieved
    # by all-zero AND by (1,2,3) with coeffs (-1,-1,1). That would mean the set is NOT
    # dissociated.
    
    # So the check is: total_pairs should equal 1 if and only if the set is dissociated.
    # Because total_pairs counts all (a, b) where a is a signed sum from half1, b is a
    # signed sum from half2, and a + b = 0. Each such pair corresponds to a signed
    # assignment of the full set that sums to 0. If dissociated, only the all-zero
    # assignment works, so total_pairs = 1.
    
    return total_pairs == 1

def _is_dissociated(elems):
    """Check if all subset sums of reciprocals are distinct using integer arithmetic."""
    return _is_dissociated_fast(elems)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit, keep safety margin before 90s kill
    
    universe = list(range(1, 23))
    
    # Greedy function with a given order
    def greedy(order):
        cur = []
        for e in order:
            if time.time() > deadline - 5:
                break
            test_set = cur + [e]
            if _is_dissociated_fast(test_set):
                cur = test_set
        return cur
    
    best_set = []
    
    # 1. Try deterministic orders
    orders = [
        sorted(universe, reverse=True),
        sorted(universe),
        list(universe),
        sorted(universe, key=lambda x: -x.bit_length()),
    ]
    
    for order in orders:
        if time.time() > deadline - 15:
            break
        result = greedy(order)
        if len(result) > len(best_set):
            best_set = list(result)
    
    # 2. Local search: try to add any missing element
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        remaining = [e for e in universe if e not in best_set]
        random.shuffle(remaining)
        for e in remaining:
            if time.time() > deadline - 10:
                break
            test_set = best_set + [e]
            if _is_dissociated_fast(test_set):
                best_set = test_set
                improved = True
                break
    
    # 3. Random restarts with greedy + local search
    while time.time() < deadline - 15:
        random.shuffle(universe)
        result = greedy(universe)
        if len(result) > len(best_set):
            best_set = list(result)
            # Try to improve this new best
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                remaining = [e for e in range(1, 23) if e not in best_set]
                random.shuffle(remaining)
                for e in remaining:
                    if time.time() > deadline - 10:
                        break
                    if _is_dissociated_fast(best_set + [e]):
                        best_set.append(e)
                        improved = True
                        break
        
        # If we found a very large set, stop early
        if len(best_set) >= 18:
            break
    
    # 4. Try to grow best_set further with more aggressive local search
    if time.time() < deadline - 10:
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            remaining = [e for e in universe if e not in best_set]
            random.shuffle(remaining)
            added = False
            for e in remaining:
                test_set = best_set + [e]
                if _is_dissociated_fast(test_set):
                    best_set = test_set
                    added = True
                    break
            if not added:
                break
    
    # Final verification
    if best_set and _is_dissociated_fast(best_set):
        return sorted(best_set)
    
    # Fallback to a known valid set
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
