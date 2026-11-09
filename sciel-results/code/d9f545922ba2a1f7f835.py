import itertools
import time

def construct():
    start_time = time.time()
    deadline = 55.0  # 55 seconds limit to be safe
    
    # 1. Generate smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    smooth_set = set(smooth)
    
    # Precompute antichain validity for pairs and triples?
    # Since S is small (~38), we can just check on the fly.
    
    # Helper to check if a sorted tuple of terms forms an antichain
    def is_antichain_sorted(terms):
        # terms must be sorted
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                if terms[j] % terms[i] == 0:
                    return False
        return True

    answers = [None] * 129  # Index 0 unused, 1..128

    for n in range(1, 129):
        # Check deadline
        if time.time() > deadline:
            # If we run out of time, we must return something.
            # But for n<=128, this shouldn't happen.
            # If it does, we might have Nones.
            # Let's try to fill remaining with a greedy approach or just return what we have?
            # The task says "All targets require valid representations".
            # So we must ensure we finish.
            # If we hit deadline, we can break and try to fix missing ones?
            # No, better to just let it run if it's fast.
            # The 3-term search is O(S^2) ~ 1444 ops. 128 * 1444 ~ 180k ops. Very fast.
            pass

        found = False
        # Try 1 term
        if n in smooth_set:
            answers[n] = [n]
            found = True
        else:
            # Try 2 terms
            # x + y = n, x, y in smooth, x != y, x not divide y
            for x in smooth:
                if x >= n:
                    break
                y = n - x
                if y in smooth_set and y != x:
                    # Check antichain
                    if x < y:
                        if y % x != 0:
                            answers[n] = [x, y]
                            found = True
                            break
                    else: # y < x
                        if x % y != 0:
                            answers[n] = [x, y]
                            found = True
                            break
            if found:
                continue

            # Try 3 terms
            # x + y + z = n
            # To ensure we find the first valid one (which might not be minimal if we don't search all, 
            # but we are searching for existence of size 3, so any valid one is fine for count=3)
            # We just need ANY valid antichain of size 3.
            for x in smooth:
                if x >= n:
                    break
                for y in smooth:
                    if y >= n - x:
                        break
                    z = n - x - y
                    if z in smooth_set and z != x and z != y and x != y:
                        # Check distinctness: z != x, z != y, x != y.
                        # Smooth numbers are unique, so if values are distinct, terms are distinct.
                        # Check antichain
                        # Sort for efficient check
                        t = sorted([x, y, z])
                        if is_antichain_sorted(t):
                            answers[n] = [x, y, z]
                            found = True
                            break
                if found:
                    break
            if found:
                continue

            # Try 4 terms
            # If not found in 3, try 4.
            # This is rarer.
            candidates = [s for s in smooth if s < n]
            # Use combinations
            for terms in itertools.combinations(candidates, 4):
                if sum(terms) == n:
                    t = sorted(terms)
                    if is_antichain_sorted(t):
                        answers[n] = list(terms)
                        found = True
                        break
            if found:
                continue
            
            # Try 5 terms
            for terms in itertools.combinations(candidates, 5):
                if sum(terms) == n:
                    t = sorted(terms)
                    if is_antichain_sorted(t):
                        answers[n] = list(terms)
                        found = True
                        break
            if found:
                continue

            # If still not found, try higher k or raise error
            # For n<=128, 3 terms should always work.
            # But just in case, let's do a general search up to 10.
            for k in range(6, 11):
                if len(candidates) < k:
                    break
                for terms in itertools.combinations(candidates, k):
                    if sum(terms) == n:
                        t = sorted(terms)
                        if is_antichain_sorted(t):
                            answers[n] = list(terms)
                            found = True
                            break
                if found:
                    break

        if not found:
            # This should not happen
            # Fallback: just pick any smooth numbers that sum to n? 
            # If antichain constraint fails, we might need to adjust.
            # But for now, assume found.
            pass

    return answers[1:]
