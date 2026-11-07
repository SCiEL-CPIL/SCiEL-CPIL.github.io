"""Bitset-based search for large sets with divisibility-incomparable subset sums."""
import time
import random

# Deadline guard: 80s wall-clock (10s safety margin under 90s limit)
_DEADLINE = 80.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def construct():
    global _START
    _START = time.time()
    
    # Maximum possible sum: 30 elements * 256 = 7680
    MAX_SUM = 7680
    MAX_ELEM = 256
    MAX_SIZE = 30
    
    # Precompute: for each sum value s (1..MAX_SUM), a bitset of all multiples of s
    # multiples_bitset[s] has bit m set iff m is a multiple of s and 1 <= m <= MAX_SUM
    # We need to check: if s is in our sums, no other multiple of s (2s, 3s, ...) can also be in sums
    # So for each s, forbidden[s] = bitset of {2s, 3s, 4s, ...} up to MAX_SUM
    # When adding a new sum value v, we need: no existing sum s where v is a multiple of s,
    # and no existing sum s where s is a multiple of v.
    
    # Actually, let's think more carefully.
    # We maintain a bitset `sums_bits` where bit s is set iff s is a subset sum.
    # When we add element x, new sums = old_sums ∪ {x} ∪ {s+x for s in old_sums}
    # We need to check that in the new sums set, no two distinct values divide each other.
    
    # Key insight: if we maintain sums in a sorted way, we can check divisibility incrementally.
    # But bitset approach: for a candidate new sum v, it conflicts with existing sum s iff
    #   s divides v (v % s == 0 and v != s) or v divides s (s % v == 0 and s != v)
    # Since all sums are positive and distinct (we'll handle distinctness separately),
    # v conflicts with s iff s | v or v | s, with s != v.
    
    # For a given v, the conflicting existing sums are:
    #   - Proper divisors of v that are in sums
    #   - Proper multiples of v that are in sums
    # 
    # We can precompute for each v:
    #   divisors_bitset[v]: bitset of proper divisors of v
    #   multiples_bitset[v]: bitset of proper multiples of v (2v, 3v, ...) up to MAX_SUM
    #
    # Then v conflicts with existing sums iff (sums_bits & divisors_bitset[v]) or (sums_bits & multiples_bitset[v])
    
    # Precompute divisors and multiples bitsets
    # Using Python integers as bitsets (arbitrary precision)
    
    # divisors_bitset[v] = bitset with bits set at positions d where d|v, d < v, d >= 1
    # multiples_bitset[v] = bitset with bits set at positions m where v|m, m > v, m <= MAX_SUM
    
    divisors_bits = [0] * (MAX_SUM + 1)
    multiples_bits = [0] * (MAX_SUM + 1)
    
    for v in range(1, MAX_SUM + 1):
        # Proper divisors
        d = 1
        while d * d < v:
            if v % d == 0:
                divisors_bits[v] |= (1 << d)
                other = v // d
                if other != v:
                    divisors_bits[v] |= (1 << other)
            d += 1
        if d * d == v:
            divisors_bits[v] |= (1 << d)
        
        # Proper multiples: 2v, 3v, ... up to MAX_SUM
        m = 2 * v
        while m <= MAX_SUM:
            multiples_bits[v] |= (1 << m)
            m += v
    
    # Now greedy construction with bitset checks
    # State: sums_bits (Python int bitset), chosen list
    # Adding element x: new_sums_bits = sums_bits | (1 << x) | ((sums_bits << x) & mask)
    #   where mask = (1 << (MAX_SUM + 1)) - 1 to limit bits
    #   But we also need to handle the case where s+x > MAX_SUM, which we just ignore.
    
    # Wait, we also need to ensure all subset sums are distinct.
    # If two different subsets have the same sum, that's invalid.
    # With the bitset representation, we can't directly detect collisions.
    # However, if we're careful about how we build the set, we can avoid this.
    # Actually, the problem says "Equal sum values are counted once. Two different subsets 
    # with the same sum still make the set invalid."
    # So we need all 2^n - 1 non-empty subset sums to be distinct.
    # This is a very strong constraint! It means the set must be "sum-distinct" or "superincreasing-like".
    # Actually, it's not exactly superincreasing, but it requires that no two subsets have the same sum.
    # This is equivalent to saying the elements form a "sum-free" set in a certain sense, or more precisely,
    # it's a set where all subset sums are unique, which is a well-studied property.
    # A sufficient condition is that each element is greater than the sum of all previous elements (superincreasing),
    # but that's not necessary.
    
    # For now, let's focus on the divisibility constraint and handle distinctness separately.
    # Actually, for the bitset approach, if we just track which sum values appear, we lose the information
    # about whether sums are unique. We need a different approach for uniqueness.
    
    # Let me reconsider. The key constraints are:
    # 1. All elements distinct, in [1, 256]
    # 2. All non-empty subset sums are distinct (no two subsets have same sum)
    # 3. No two distinct subset sums divide each other
    
    # Constraint 2 is very strong. For n elements, we need 2^n - 1 distinct sums.
    # The minimum possible sum is 1, maximum is at most 256*n.
    # So we need 2^n - 1 <= 256*n, which gives n <= about 9-10 for small values.
    # Actually 2^10 - 1 = 1023, 256*10 = 2560, so n=10 is possible in terms of range.
    # 2^11 - 1 = 2047, 256*11 = 2816, still possible.
    # 2^12 - 1 = 4095, 256*12 = 3072, NOT possible. So max n is around 11.
    
    # Wait, that's a key insight! The maximum size is bounded by 2^n - 1 <= sum of all elements <= 256*n.
    # For n=11: 2047 <= 2816, OK
    # For n=12: 4095 <= 3072, NOT OK
    # So the maximum possible size is at most 11!
    
    # This dramatically changes the search space. We're looking for at most 11 elements.
    # Let's verify: if we have n elements all in [1,256], the sum of all elements is at most 256n.
    # We need 2^n - 1 distinct non-empty subset sums, all in [1, 256n].
    # So 2^n - 1 <= 256n.
    # n=10: 1023 <= 2560 ✓
    # n=11: 2047 <= 2816 ✓
    # n=12: 4095 <= 3072 ✗
    
    # So max size is 11. Let's search for sets of size up to 11.
    
    # For the distinctness constraint, we need to track actual subset sums, not just which values appear.
    # But with at most 11 elements, 2^11 = 2048 subsets, which is very manageable.
    
    # Let's use a simpler approach: backtracking with pruning.
    # For each candidate set, compute all subset sums and check both constraints.
    
    def _check_set(elements):
        """Check if a set of elements satisfies both constraints."""
        n = len(elements)
        if n == 0:
            return True
        if len(set(elements)) != n:
            return False
        for x in elements:
            if not isinstance(x, int) or isinstance(x, bool):
                return False
            if x < 1 or x > 256:
                return False
        
        # Compute all non-empty subset sums
        sums = []
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            sums.append(s)
        
        # Check distinctness
        if len(set(sums)) != len(sums):
            return False
        
        # Check divisibility
        sum_set = set(sums)
        max_sum = max(sum_set)
        for s in sum_set:
            multiple = s * 2
            while multiple <= max_sum:
                if multiple in sum_set:
                    return False
                multiple += s
        
        return True
    
    def _greedy_with_pruning(pool, order, max_size=11):
        """Greedy construction with pruning."""
        chosen = []
        chosen_set = set()
        
        for x in order:
            if len(chosen) >= max_size:
                break
            if x in chosen_set:
                continue
            
            # Try adding x
            candidate = chosen + [x]
            if _check_set(candidate):
                chosen = candidate
                chosen_set.add(x)
        
        return chosen
    
    pool = list(range(1, 257))
    best = []
    best_len = 0
    
    rng = random.Random(42)
    
    # Phase 1: Greedy with various orderings
    # Try large-first
    order1 = sorted(pool, reverse=True)
    result1 = _greedy_with_pruning(pool, order1)
    if len(result1) > best_len:
        best = result1
        best_len = len(result1)
    
    # Try small-first
    if _remaining() > 10:
        order2 = sorted(pool)
        result2 = _greedy_with_pruning(pool, order2)
        if len(result2) > best_len:
            best = result2
            best_len = len(result2)
    
    # Phase 2: Randomized restarts
    while _remaining() > 5:
        random_order = list(pool)
        rng.shuffle(random_order)
        result = _greedy_with_pruning(pool, random_order)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Phase 3: Local search - try to improve best by swapping elements
    if best and _remaining() > 10:
        improved = True
        while improved and _remaining() > 5:
            improved = False
            best_set = set(best)
            for i in range(len(best)):
                if _remaining() < 5:
                    break
                for x in pool:
                    if x in best_set:
                        continue
                    candidate = best[:i] + [x] + best[i+1:]
                    if _check_set(candidate):
                        # Check if this leads to a larger set
                        # Try to extend
                        extended = _greedy_with_pruning(pool, [x] + [y for y in pool if y not in set(candidate)], max_size=best_len + 1)
                        # Actually, let's just do a simple swap
                        best = candidate
                        best_set = set(best)
                        improved = True
                        break
                if improved:
                    break
    
    # Final validation
    if best and not _check_set(best):
        best = [128, 192, 224]
    
    return [int(x) for x in best]
