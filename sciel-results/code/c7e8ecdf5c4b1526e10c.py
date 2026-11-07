import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    N = 128
    # Base set: {65, ..., 128}
    # This is sum-free internally (min sum 65+66=131 > 128)
    base = set(range(65, N + 1))
    
    # Bitmask for O(1) membership tests
    def make_mask(S):
        m = 0
        for x in S:
            m |= (1 << x)
        return m

    def can_add(mask, S, candidate):
        """
        Check if adding candidate to S (with membership mask) maintains validity.
        We only need to check triples where candidate is one of the distinct members a,b,c.
        If candidate is not in the triple, the triple was already valid or invalid before.
        """
        elems = list(S)
        n = len(elems)
        
        # Case 1: candidate is the largest element c in the triple (a, b, candidate)
        # Violation if a+b, a+candidate, b+candidate are all in S_new.
        # Since candidate is new, a+candidate and b+candidate must be in S (old) or be candidate (impossible as sums > candidate).
        # So a+candidate and b+candidate must be in S.
        # And a+b must be in S_new. Since a,b < candidate, a+b < 2*candidate. 
        # If a+b == candidate, it's in S_new. Otherwise it must be in S.
        for i in range(n):
            a = elems[i]
            ac = candidate + a
            if ac > N:
                continue
            # ac must be in S (old) or ac == candidate (impossible)
            if not ((mask >> ac) & 1):
                continue
            for j in range(i + 1, n):
                b = elems[j]
                bc = candidate + b
                if bc > N:
                    continue
                if not ((mask >> bc) & 1):
                    continue
                ab = a + b
                if ab > N:
                    continue
                # ab must be in S_new. So ab in S or ab == candidate.
                if ((mask >> ab) & 1) or ab == candidate:
                    return False

        # Case 2: candidate is the middle element b in the triple (a, candidate, c)
        # Violation if a+candidate, a+c, candidate+c are all in S_new.
        # a+candidate and candidate+c must be in S (old) or be candidate (impossible).
        # a+c must be in S_new.
        for i in range(n):
            a = elems[i]
            if a >= candidate:
                break
            ac = candidate + a
            if ac > N:
                continue
            if not ((mask >> ac) & 1):
                continue
            for k in range(n):
                c = elems[k]
                if c <= candidate:
                    continue
                cc = candidate + c
                if cc > N:
                    continue
                if not ((mask >> cc) & 1):
                    continue
                ac_old = a + c
                if ac_old > N:
                    continue
                # ac_old must be in S_new. So ac_old in S or ac_old == candidate.
                if ((mask >> ac_old) & 1) or ac_old == candidate:
                    return False

        # Case 3: candidate is the smallest element a in the triple (candidate, b, c)
        # Violation if candidate+b, candidate+c, b+c are all in S_new.
        # candidate+b and candidate+c must be in S (old) or be candidate (impossible).
        # b+c must be in S_new.
        for j in range(n):
            b = elems[j]
            if b <= candidate:
                continue
            cb = candidate + b
            if cb > N:
                continue
            if not ((mask >> cb) & 1):
                continue
            for k in range(j + 1, n):
                c = elems[k]
                cc = candidate + c
                if cc > N:
                    continue
                if not ((mask >> cc) & 1):
                    continue
                bc = b + c
                if bc > N:
                    continue
                # bc must be in S_new. So bc in S or bc == candidate.
                if ((mask >> bc) & 1) or bc == candidate:
                    return False

        return True

    # Greedy: try to add elements from 1 to 64 in increasing order
    S_best = set(base)
    mask_best = make_mask(S_best)
    
    S_curr = set(base)
    mask_curr = make_mask(S_curr)
    for x in range(1, 65):
        if time.time() > deadline:
            break
        if can_add(mask_curr, S_curr, x):
            S_curr.add(x)
            mask_curr |= (1 << x)
    
    if len(S_curr) > len(S_best):
        S_best = set(S_curr)
        mask_best = make_mask(S_best)

    # Random restarts: try different orderings of 1..64
    rng = random.Random(42)
    small_elements = list(range(1, 65))
    
    # Run as many restarts as possible within time budget
    while time.time() < deadline - 2.0:
        S_try = set(base)
        mask_try = make_mask(S_try)
        order = small_elements[:]
        rng.shuffle(order)
        
        for x in order:
            if time.time() > deadline:
                break
            if can_add(mask_try, S_try, x):
                S_try.add(x)
                mask_try |= (1 << x)
        
        if len(S_try) > len(S_best):
            S_best = set(S_try)
            mask_best = make_mask(S_best)
            # Optional: try to add more elements if we found a better set
            # But the greedy already tried all 1..64 in some order.
            # Actually, the greedy in the restart tries all 1..64 in shuffled order.
            # So S_try is maximal for that ordering.

    # Final check and return
    # Ensure we return a list of ints
    result = sorted(S_best)
    result = [int(x) for x in result]
    
    # Safety check: if we somehow exceeded time, return the base set
    if time.time() > deadline:
        return [int(x) for x in sorted(base)]

    return result

if __name__ == "__main__":
    construct()
