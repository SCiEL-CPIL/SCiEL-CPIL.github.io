"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_valid(S):
    """Check that no three distinct a,b,c in S have all three pair sums in S."""
    s = set(S)
    for i, a in enumerate(S):
        for j in range(i+1, len(S)):
            b = S[j]
            ab = a + b
            if ab not in s:
                continue
            for k in range(j+1, len(S)):
                c = S[k]
                if a + c in s and b + c in s:
                    return False
    return True

def _violation_count(S):
    """Count triples (a,b,c) with all three pair sums in S. 0 means valid."""
    s = set(S)
    cnt = 0
    for i, a in enumerate(S):
        for j in range(i+1, len(S)):
            b = S[j]
            ab = a + b
            if ab not in s:
                continue
            for k in range(j+1, len(S)):
                c = S[k]
                if a + c in s and b + c in s:
                    cnt += 1
    return cnt

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    best = list(range(17, 33)) + list(range(65, 129))  # seed: 16 + 64 = 80
    best_size = len(best)

    def try_add(S, x):
        s = set(S)
        if x in s:
            return S
        # Adding x creates a bad triple iff there exist a,b in S with
        # a+b in S and a+x in S and b+x in S.
        # Equivalently: exists a in S such that a+x in S and (a+x - a) = x... 
        # Let's check directly but efficiently.
        # For each pair (a,b) in S with a+b in S, check if a+x in S and b+x in S.
        # But that's O(n^2). Alternative: for each a in S where a+x in S,
        # we need b in S such that a+b in S and b+x in S.
        # b+x in S means b in {y - x : y in S}. So b in S ∩ (S - x).
        # And a+b in S means b in S - a.
        # So for each a in S with a+x in S, check if (S ∩ (S-x) ∩ (S-a)) is non-empty.
        S_set = set(S)
        S_minus_x = {y - x for y in S_set if y - x >= 1}
        for a in S_set:
            if a + x in S_set:
                # need b in S ∩ (S_minus_x) ∩ (S - a)
                # S - a = {y - a for y in S}
                S_minus_a = {y - a for y in S_set if y - a >= 1}
                # intersection of three sets
                # iterate over smallest
                candidates = S_set & S_minus_x & S_minus_a
                if candidates:
                    return None  # adding x creates a violation
        return S + [x]

    # Phase 1: Greedy add
    S = list(best)
    S_set = set(S)
    for x in range(1, 129):
        if x in S_set:
            continue
        if time.time() > deadline:
            break
        new_S = try_add(S, x)
        if new_S is not None:
            S = new_S
            S_set = set(S)

    if len(S) > best_size:
        best = S
        best_size = len(S)

    # Phase 2: Local search - try removing one and adding another
    for _ in range(2000):
        if time.time() > deadline:
            break
        S = list(best)
        S_set = set(S)
        n = len(S)
        # Try removing a random element and adding a random missing one
        idx = random.randint(0, n - 1)
        removed = S[idx]
        S2 = S[:idx] + S[idx+1:]
        S2_set = set(S2)
        # Try to add back removed or add a new element
        for x in [removed] + [random.randint(1, 128) for _ in range(10)]:
            if x in S2_set:
                continue
            new_S = try_add(S2, x)
            if new_S is not None and len(new_S) > best_size:
                best = new_S
                best_size = len(new_S)
                break
            elif new_S is not None and len(new_S) == len(S2) + 1:
                # same size but different composition; keep for diversity
                pass

    # Phase 3: Try to add more elements to best
    S = list(best)
    S_set = set(S)
    for x in range(1, 129):
        if x in S_set:
            continue
        if time.time() > deadline:
            break
        new_S = try_add(S, x)
        if new_S is not None:
            S = new_S
            S_set = set(S)

    if len(S) > best_size:
        best = S
        best_size = len(S)

    # Final verification
    if not _is_valid(best):
        # Fallback to known valid set
        best = list(range(17, 33)) + list(range(65, 129))

    # Ensure all are Python ints
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
