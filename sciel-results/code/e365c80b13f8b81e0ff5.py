"""Fresh design: hypergraph greedy + local-search polish for no-triple pair-sum set."""
import time
import random
import sys

N = 128

def _precompute_hyperedges():
    """Build hyperedges: for each triple (a,b,c) with a<b<c in 1..N,
    if a+b, a+c, b+c are all in 1..N, then {a,b,c,a+b,a+c,b+c} is a forbidden 6-set.
    Return list of frozensets (as bitmasks) of the 6 elements."""
    edges = []
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            for c in range(b+1, N+1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 <= N and s2 <= N and s3 <= N:
                    # The 6 values: a, b, c, s1, s2, s3 (some may coincide)
                    vals = {a, b, c, s1, s2, s3}
                    if len(vals) >= 3:  # at least a,b,c distinct
                        edges.append(frozenset(vals))
    return edges

def _can_add(bitmask, elem, forbidden_sets):
    """Check if adding elem to current set (bitmask) creates a violation.
    A violation exists if there is a forbidden 6-set F such that
    F \ {elem} is a subset of the current set."""
    bit = 1 << (elem - 1)
    new_mask = bitmask | bit
    for fs in forbidden_sets:
        fs_mask = 0
        for v in fs:
            fs_mask |= 1 << (v - 1)
        # violation if (fs_mask & ~bit) is subset of bitmask, i.e., all other 5 in current
        if (fs_mask & ~bit) & ~bitmask == 0 and (fs_mask & bitmask) != 0:
            return False
    return True

def _validate(result):
    """Check the result has no forbidden triple."""
    s = set(result)
    for i, a in enumerate(result):
        for b in result[i+1:]:
            for c in result[i+1:]:
                pass
    # More efficient: check all triples
    arr = sorted(result)
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            for k in range(j+1, len(arr)):
                a, b, c = arr[i], arr[j], arr[k]
                if a+b in s and a+c in s and b+c in s:
                    return False
    return True

def construct():
    deadline = time.time() + 85.0
    rng = random.Random(42)

    # Precompute forbidden sets as bitmasks for speed
    forbidden_masks = []
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            for c in range(b+1, N+1):
                s1, s2, s3 = a+b, a+c, b+c
                if s1 <= N and s2 <= N and s3 <= N:
                    vals = {a, b, c, s1, s2, s3}
                    m = 0
                    for v in vals:
                        m |= 1 << (v - 1)
                    forbidden_masks.append(m)

    best = list(range(65, 129))  # seed: upper half, size 64
    best_size = len(best)

    def _is_valid_set(lst):
        s = set(lst)
        arr = sorted(lst)
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                for k in range(j+1, len(arr)):
                    a, b, c = arr[i], arr[j], arr[k]
                    if a+b in s and a+c in s and b+c in s:
                        return False
        return True

    # Phase 1: Greedy construction with randomized order
    for trial in range(200):
        if time.time() > deadline - 10:
            break
        order = list(range(1, N+1))
        rng.shuffle(order)
        current = set()
        for elem in order:
            # Check if adding elem violates any forbidden set
            ok = True
            bit = 1 << (elem - 1)
            for fm in forbidden_masks:
                # If all other elements of fm are already in current, we can't add elem
                others = fm & ~bit
                # Check if others is subset of current
                cur_mask = 0
                for v in current:
                    cur_mask |= 1 << (v - 1)
                if (others & ~cur_mask) == 0 and others != 0:
                    ok = False
                    break
            if ok:
                current.add(elem)
        if len(current) > best_size:
            best = sorted(current)
            best_size = len(current)

    # Phase 2: Local search polish on best
    if time.time() < deadline - 5:
        current = set(best)
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            # Try adding each missing element
            for elem in range(1, N+1):
                if elem in current:
                    continue
                ok = True
                bit = 1 << (elem - 1)
                cur_mask = 0
                for v in current:
                    cur_mask |= 1 << (v - 1)
                for fm in forbidden_masks:
                    others = fm & ~bit
                    if (others & ~cur_mask) == 0 and others != 0:
                        ok = False
                        break
                if ok:
                    current.add(elem)
                    improved = True
                    break
            # Try swaps: remove one, add another
            if not improved:
                for rem in list(current):
                    if time.time() > deadline - 3:
                        break
                    cur_mask = 0
                    for v in current:
                        cur_mask |= 1 << (v - 1)
                    for add in range(1, N+1):
                        if add in current:
                            continue
                        new_set = current - {rem} | {add}
                        # Validate new_set
                        s = set(new_set)
                        arr = sorted(new_set)
                        valid = True
                        for i in range(len(arr)):
                            for j in range(i+1, len(arr)):
                                for k in range(j+1, len(arr)):
                                    a, b, c = arr[i], arr[j], arr[k]
                                    if a+b in s and a+c in s and b+c in s:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if valid:
                            current = new_set
                            improved = True
                            break
                    if improved:
                        break

        if len(current) > best_size:
            best = sorted(current)
            best_size = len(current)

    # Final validation
    if not _is_valid_set(best):
        # Fallback to safe seed
        best = list(range(65, 129))

    return list(best)
