"""Rewritten program for reciprocal dissociated set task."""
# EVOLVE-BLOCK-START
import time
from math import lcm as _lcm

def construct():
    start_time = time.time()
    timeout = 55.0

    N = 22
    candidates = list(range(1, N + 1))

    # Precompute LCM of any subset quickly using incremental lcm
    # For a given set S, compute L = lcm(S), then vals = [L//x for x in S]
    # Then check all 2^k subset sums of vals are distinct.

    def check_dissociated(vals):
        """Check if all subset sums of vals are distinct. vals is a list of positive ints.
        Returns True if dissociated, False otherwise. Uses bitmask enumeration."""
        k = len(vals)
        if k == 0:
            return True
        # Generate all 2^k subset sums
        # Use iterative doubling
        sums = [0]
        for v in vals:
            # Add v to each existing sum
            sums += [s + v for s in sums]
            # Early check: if len(sums) < 2^(i+1), collision already
            if len(sums) < (1 << (len(vals))):
                # We can check for duplicates periodically
                pass
        # Check for duplicates
        if len(set(sums)) != len(sums):
            return False
        return True

    def check_dissociated_fast(S):
        """Fast check: compute LCM, convert to integer vals, check subset sums distinct."""
        k = len(S)
        if k == 0:
            return True
        L = 1
        for x in S:
            L = _lcm(L, x)
            # If L gets too large, we might have issues, but for [1..22] it's fine
        vals = [L // x for x in S]
        # Sort vals to potentially catch collisions earlier (smaller values first)
        # Actually, order doesn't matter for the set of sums
        # Generate all subset sums
        sums = [0]
        for v in vals:
            sums += [s + v for s in sums]
        # Check uniqueness
        return len(set(sums)) == (1 << k)

    def check_dissociated_with_early_exit(S):
        """Check with early exit: build sums incrementally, check for dups at each step."""
        k = len(S)
        if k == 0:
            return True
        L = 1
        for x in S:
            L = _lcm(L, x)
        vals = [L // x for x in S]
        # Build subset sums incrementally
        sums = [0]
        for i, v in enumerate(vals):
            new_sums = [s + v for s in sums]
            sums = sums + new_sums
            # Check for duplicates: expected size is 2^(i+1)
            if len(sums) != (1 << (i + 1)):
                return False
        return len(set(sums)) == (1 << k)

    # Better: use a set-based approach that can early-exit
    def is_dissociated_set(S):
        """Use a set to track sums; early exit on collision."""
        k = len(S)
        if k == 0:
            return True
        L = 1
        for x in S:
            L = _lcm(L, x)
        vals = [L // x for x in S]
        sums = {0}
        for v in vals:
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums = sums | new_sums
        return True

    # Precompute: for each pair (a, b), 1/a == 1/b is impossible since a != b.
    # But 1/a == 1/b + 1/c means a*b*c = b*c + a*c + a*b... 
    # Actually, the key constraint is: no subset sum of reciprocals equals another.
    # A cheap necessary condition: for any two distinct elements a, b in S,
    # we need 1/a != 1/b (always true) and 1/a != 1/b + 1/c for any c in S, etc.
    # The simplest cheap check: for any triple (a,b,c) in S, 1/a != 1/b + 1/c
    # i.e., b*c != a*(b+c) i.e., b*c - a*b - a*c != 0 i.e., (a+b)(a+c) != a^2 + a*b + a*c + b*c... 
    # Let's just check: a*(b+c) != b*c, i.e., a*b + a*c != b*c
    # This is a necessary condition (if violated, the set is not dissociated).
    
    def cheap_necessary_check(S):
        """Check necessary conditions that are fast to verify.
        For any distinct a, b, c in S: 1/a != 1/b + 1/c, etc.
        More generally, for any partition of S into two non-empty parts A, B:
        sum_{a in A} 1/a != sum_{b in B} 1/b.
        The cheapest non-trivial check: triples where one element's reciprocal
        equals sum of two others."""
        k = len(S)
        if k <= 2:
            return True
        # Check all triples: for distinct a, b, c in S, 1/a != 1/b + 1/c
        # This means a*(b+c) != b*c
        S_set = set(S)
        for i in range(k):
            a = S[i]
            for j in range(i+1, k):
                b = S[j]
                for m in range(j+1, k):
                    c = S[m]
                    # Check 1/a == 1/b + 1/c => a*(b+c) == b*c
                    if a * (b + c) == b * c:
                        return False
                    # Check 1/b == 1/a + 1/c => b*(a+c) == a*c
                    if b * (a + c) == a * c:
                        return False
                    # Check 1/c == 1/a + 1/b => c*(a+b) == a*b
                    if c * (a + b) == a * b:
                        return False
        return True

    # Backtracking search
    def backtrack(current_set, available, k_target, time_limit):
        if time.time() - start_time > time_limit:
            return None
        if len(current_set) == k_target:
            if is_dissociated_set(current_set):
                return list(current_set)
            return None
        
        # Pruning
        if len(current_set) + len(available) < k_target:
            return None
            
        for i, x in enumerate(available):
            new_set = current_set + [x]
            # Cheap check first
            if not cheap_necessary_check(new_set):
                continue
            # Full check
            if is_dissociated_set(new_set):
                result = backtrack(new_set, available[i+1:], k_target, time_limit)
                if result is not None:
                    return result
            # If new_set is not dissociated, skip x for this branch
        return None

    # Try sizes from 22 down to 1
    # Sort candidates in descending order (larger numbers = smaller reciprocals, 
    # less likely to cause collisions)
    for k in range(N, 0, -1):
        available = sorted(candidates, reverse=True)
        result = backtrack([], available, k, timeout)
        if result is not None:
            return result

    # Fallback: known valid set
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]

# EVOLVE-BLOCK-END
