"""Rewritten program: systematic + randomized search with prime-mask pruning."""
import itertools
import random
import time

def _prime_factor_mask(n):
    """Return a bitmask of odd prime factors of n (for odd n)."""
    mask = 0
    d = 3
    while d * d <= n:
        while n % d == 0:
            mask |= (1 << d)
            n //= d
        d += 2
    if n > 1:
        mask |= (1 << n)
    return mask

def _is_pairwise_coprime(values):
    """Check if all distinct values in the list are pairwise coprime."""
    vals = list(set(values))
    masks = [_prime_factor_mask(v) for v in vals]
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if masks[i] & masks[j]:
                return False
    return True

def _score(A, B):
    """Return (num_distinct, is_valid) for given A, B."""
    sums = [a + b for a in A for b in B]
    distinct = set(sums)
    if len(distinct) < 2:
        return 0, False
    if not _is_pairwise_coprime(list(distinct)):
        return 0, False
    return len(distinct), True

def construct():
    # Tighter deadline to ensure we return before 60s
    deadline = time.time() + 50
    
    # Parity constraint: either A all odd & B all even, or A all even & B all odd
    odds = list(range(1, 129, 2))   # odd numbers in [1,128]
    evens = list(range(2, 129, 2))  # even numbers in [1,128]
    
    best_count = 0
    best_AB = None
    
    # Precompute masks for all odd numbers 3..255
    max_sum = 255
    masks = {}
    for v in range(3, max_sum + 1, 2):  # odd sums only
        masks[v] = _prime_factor_mask(v)
    
    def check_valid(A, B):
        """Fast check using precomputed masks."""
        sums = [a + b for a in A for b in B]
        distinct = set(sums)
        if len(distinct) < 2:
            return 0, False
        # Check pairwise coprime using masks
        dlist = list(distinct)
        for i in range(len(dlist)):
            mi = masks.get(dlist[i], 0)
            for j in range(i + 1, len(dlist)):
                mj = masks.get(dlist[j], 0)
                if mi & mj:
                    return 0, False
        return len(distinct), True

    # Known good solution (9 distinct sums) to ensure we have a baseline
    # A=[1, 7, 13], B=[4, 10, 16] -> sums: 5, 11, 17, 11, 17, 23, 17, 23, 29
    # Distinct: {5, 11, 17, 23, 29} -> 5 distinct. 
    # Let's find a better one or just rely on search.
    # The previous run found 9 distinct. Let's try to find it quickly.
    
    random.seed(42)
    
    # Phase 1: Randomized search
    # We can do many trials. Each trial is very fast (3x3 sums, set, mask check).
    # Estimate: 100,000 trials * 10us/trial = 1s. So we can do millions.
    
    for trial in range(1000000):
        if time.time() > deadline:
            break
        # Randomly pick 3 odds and 3 evens
        A = list(random.sample(odds, 3))
        B = list(random.sample(evens, 3))
        count, valid = check_valid(A, B)
        if valid and count > best_count:
            best_count = count
            best_AB = (A, B)
            if best_count == 9: # Max possible is 9
                # We found a perfect solution, can stop early or continue to see if we can verify
                # But 9 is the max, so we can break
                break
        # Also try the swap (A even, B odd)
        A2 = list(random.sample(evens, 3))
        B2 = list(random.sample(odds, 3))
        count2, valid2 = check_valid(A2, B2)
        if valid2 and count2 > best_count:
            best_count = count2
            best_AB = (A2, B2)
            if best_count == 9:
                break

    # Phase 2: Local improvement if we didn't find 9
    if best_AB and best_count < 9 and time.time() < deadline:
        A_best, B_best = best_AB
        # Try perturbing one element at a time
        for _ in range(20000):
            if time.time() > deadline:
                break
            # Decide which set to perturb
            if random.random() < 0.5:
                # Perturb A (odd set)
                A_new = list(A_best)
                idx = random.randint(0, 2)
                old_val = A_new[idx]
                new_val = random.choice([v for v in odds if v != old_val])
                A_new[idx] = new_val
                if len(set(A_new)) != 3:
                    continue
                count, valid = check_valid(A_new, B_best)
                if valid and count > best_count:
                    best_count = count
                    best_AB = (A_new, list(B_best))
                    if best_count == 9:
                        break
            else:
                # Perturb B (even set)
                B_new = list(B_best)
                idx = random.randint(0, 2)
                old_val = B_new[idx]
                new_val = random.choice([v for v in evens if v != old_val])
                B_new[idx] = new_val
                if len(set(B_new)) != 3:
                    continue
                count, valid = check_valid(A_best, B_new)
                if valid and count > best_count:
                    best_count = count
                    best_AB = (list(A_best), B_new)
                    if best_count == 9:
                        break

    if best_AB is None:
        # Fallback: known valid solution
        return [[1, 7, 13], [4, 10, 16]]
    
    return [best_AB[0], best_AB[1]]

# EVOLVE-BLOCK-END
