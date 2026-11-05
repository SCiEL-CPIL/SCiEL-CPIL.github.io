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
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    
    # Parity constraint: either A all odd & B all even, or A all even & B all odd
    odds = list(range(1, 129, 2))   # odd numbers in [1,128]
    evens = list(range(2, 129, 2))  # even numbers in [1,128]
    
    best_count = 0
    best_AB = None
    
    # Systematic search: try all combinations of 3 odds for A and 3 evens for B
    # That's C(64,3) * C(64,3) which is too large. Use smarter approach.
    
    # Precompute masks for all odd numbers 1..255 (sums of odd+even are odd, max 128+128=256, but odd sums max 255)
    # Actually sums range from 1+2=3 to 127+128=255
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
    
    # Phase 1: Systematic search over structured subsets
    # Try small ranges first
    for start_a in range(0, len(odds), 2):
        if time.time() > deadline:
            break
        a_pool = odds[start_a:start_a + 12]
        for start_b in range(0, len(evens), 2):
            if time.time() > deadline:
                break
            b_pool = evens[start_b:start_b + 12]
            for A in itertools.combinations(a_pool, 3):
                if time.time() > deadline:
                    break
                for B in itertools.combinations(b_pool, 3):
                    count, valid = check_valid(A, B)
                    if valid and count > best_count:
                        best_count = count
                        best_AB = (list(A), list(B))
    
    # Phase 2: Randomized search with larger pools
    if time.time() < deadline:
        random.seed(42)
        for trial in range(200000):
            if time.time() > deadline:
                break
            # Randomly pick 3 odds and 3 evens
            A = list(random.sample(odds, 3))
            B = list(random.sample(evens, 3))
            count, valid = check_valid(A, B)
            if valid and count > best_count:
                best_count = count
                best_AB = (A, B)
                # Also try the swap (A even, B odd)
                A2 = list(random.sample(evens, 3))
                B2 = list(random.sample(odds, 3))
                count2, valid2 = check_valid(A2, B2)
                if valid2 and count2 > best_count:
                    best_count = count2
                    best_AB = (A2, B2)
    
    # Phase 3: Local improvement around best found
    if best_AB and time.time() < deadline:
        A_best, B_best = best_AB
        # Try perturbing one element at a time
        for _ in range(5000):
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
    
    if best_AB is None:
        # Fallback: known valid solution
        return [[1, 7, 13], [4, 10, 16]]
    
    return [best_AB[0], best_AB[1]]

# EVOLVE-BLOCK-END
