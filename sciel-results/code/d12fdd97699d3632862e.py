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
    deadline = time.time() + 75.0  # safety margin before 90s kill
    best = list(range(17, 33)) + list(range(65, 129))  # seed: 16 + 64 = 80
    best_size = len(best)

    def try_add(S, x, S_set=None):
        """Try to add x to S. Returns new list or None if invalid."""
        if S_set is None:
            S_set = set(S)
        if x in S_set:
            return None
        # Adding x creates a bad triple iff there exist distinct a,b in S with
        # a+b in S, a+x in S, and b+x in S.
        # For each a in S such that a+x in S, check if there exists b in S
        # such that b+x in S and a+b in S.
        # b+x in S => b in {y-x : y in S_set}
        # a+b in S => b in {y-a : y in S_set}
        S_minus_x = {y - x for y in S_set if 1 <= y - x <= 128}
        for a in S_set:
            if a + x in S_set:
                S_minus_a = {y - a for y in S_set if 1 <= y - a <= 128}
                # Need b in S_set ∩ S_minus_x ∩ S_minus_a
                # Use the smallest set for iteration
                if len(S_minus_x) < len(S_minus_a):
                    for b in S_minus_x:
                        if b in S_set and b in S_minus_a:
                            return None
                else:
                    for b in S_minus_a:
                        if b in S_set and b in S_minus_x:
                            return None
        return S + [x]

    def greedy_add(S, S_set, order=None):
        """Greedy add elements in given order."""
        if order is None:
            order = range(1, 129)
        for x in order:
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            new_S = try_add(S, x, S_set)
            if new_S is not None:
                S = new_S
                S_set = set(S)
        return S, S_set

    # Phase 1: Greedy add in natural order
    S = list(best)
    S_set = set(S)
    S, S_set = greedy_add(S, S_set)
    if len(S) > best_size:
        best = S
        best_size = len(S)

    # Phase 2: Multiple random restarts with greedy
    for restart in range(20):
        if time.time() > deadline:
            break
        # Start from a random subset of best or a random seed
        if restart == 0:
            S = list(best)
        else:
            # Random seed: take random elements from 1..128
            S = []
            S_set = set()
            for x in random.sample(range(1, 129), min(40, len(best))):
                if x not in S_set:
                    S.append(x)
                    S_set.add(x)
        S, S_set = greedy_add(S, S_set)
        if len(S) > best_size:
            best = S
            best_size = len(S)

    # Phase 3: Local search - remove one, try to add multiple
    for iteration in range(5000):
        if time.time() > deadline:
            break
        S = list(best)
        S_set = set(S)
        n = len(S)
        if n == 0:
            break
        # Try removing a random element
        idx = random.randint(0, n - 1)
        removed = S[idx]
        S2 = S[:idx] + S[idx+1:]
        S2_set = set(S2)
        
        # Try to add back removed and other elements
        candidates = [removed] + [x for x in random.sample(range(1, 129), min(15, 128 - len(S2_set))) if x not in S2_set]
        best_S2 = None
        best_S2_size = len(S2)
        for x in candidates:
            if x in S2_set:
                continue
            new_S = try_add(S2, x, S2_set)
            if new_S is not None:
                if len(new_S) > best_S2_size:
                    best_S2 = new_S
                    best_S2_size = len(new_S)
        
        if best_S2 is not None and len(best_S2) > best_size:
            best = best_S2
            best_size = len(best)

    # Phase 4: Try to add more elements to best
    S = list(best)
    S_set = set(S)
    S, S_set = greedy_add(S, S_set)
    if len(S) > best_size:
        best = S
        best_size = len(S)

    # Phase 5: More aggressive local search
    for iteration in range(3000):
        if time.time() > deadline:
            break
        S = list(best)
        S_set = set(S)
        n = len(S)
        if n < 2:
            break
        # Remove two random elements
        idx1 = random.randint(0, n - 1)
        idx2 = random.randint(0, n - 2)
        if idx2 >= idx1:
            idx2 += 1
        removed1 = S[idx1]
        removed2 = S[idx2]
        S2 = S[:idx2] + S[idx2+1:idx1] + S[idx1+1:] if idx2 < idx1 else S[:idx1] + S[idx1+1:idx2] + S[idx2+1:]
        S2_set = set(S2)
        
        # Try to add back removed elements and new ones
        candidates = [removed1, removed2] + [x for x in random.sample(range(1, 129), min(20, 128 - len(S2_set))) if x not in S2_set]
        best_S2 = None
        best_S2_size = len(S2)
        for x in candidates:
            if x in S2_set:
                continue
            new_S = try_add(S2, x, S2_set)
            if new_S is not None:
                if len(new_S) > best_S2_size:
                    best_S2 = new_S
                    best_S2_size = len(new_S)
        
        if best_S2 is not None and len(best_S2) > best_size:
            best = best_S2
            best_size = len(best)

    # Final verification
    if not _is_valid(best):
        # Fallback to known valid set
        best = list(range(17, 33)) + list(range(65, 129))

    # Ensure all are Python ints
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
