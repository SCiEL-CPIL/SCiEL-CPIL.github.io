"""
Maximize cardinality of S ⊆ {2..256} such that for all distinct a,b,c ∈ S: a ∤ b*c.

Key insight: numbers > 128 are trivially valid (since b*c > 256 ≥ a). 
Start with all 128 numbers > 128, then greedily/local-search to add smaller ones.
"""

import time
import random

_START = time.time()
_DEADLINE = 85.0  # safety margin before 90s kill

def _elapsed():
    return time.time() - _START


def _precompute(n=256):
    """Precompute divisor lists up to n*n for fast divisibility checks."""
    max_prod = n * n
    # divs[m] = list of d in [2..n] that divide m
    divs = [[] for _ in range(max_prod + 1)]
    for d in range(2, n + 1):
        for m in range(d, max_prod + 1, d):
            divs[m].append(d)
    return divs


def _can_add(x, S, divs):
    """Check if x can be added to set S (list) without violating the constraint.
    For all distinct a, b in S: need a ∤ x*b, b ∤ x*a, and x ∤ a*b."""
    L = len(S)
    for i in range(L):
        a = S[i]
        for j in range(i + 1, L):
            b = S[j]
            if a in divs[x * b]:
                return False
            if b in divs[x * a]:
                return False
            if x in divs[a * b]:
                return False
    return True


def _is_valid(S, divs):
    """Verify the entire set S is valid."""
    L = len(S)
    for i in range(L):
        a = S[i]
        for j in range(i + 1, L):
            b = S[j]
            for k in range(j + 1, L):
                c = S[k]
                if a in divs[b * c]:
                    return False
                if b in divs[a * c]:
                    return False
                if c in divs[a * b]:
                    return False
    return True


def construct():
    n = 256
    divs = _precompute(n)
    uni = list(range(2, n + 1))

    # Phase 1: Start with all numbers > 128 (trivially valid since b*c >= 129*129 > 256 >= a)
    large = list(range(129, 257))  # 128 elements
    S = large[:]
    S_set = set(S)

    # Phase 2: Greedily add smaller numbers
    # Try to add numbers from 2 to 128
    small_candidates = list(range(2, 129))

    # Sort by "least restrictive" - numbers with fewer divisors are easier to add
    def num_divisors(x):
        cnt = 0
        for d in divs[x]:
            cnt += 1
        return cnt

    # Try adding in order of fewest conflicts
    # For each candidate x, count how many pairs (a,b) in S would be violated
    def conflict_count(x, S):
        L = len(S)
        cnt = 0
        for i in range(L):
            a = S[i]
            for j in range(i + 1, L):
                b = S[j]
                if a in divs[x * b]:
                    cnt += 1
                if b in divs[x * a]:
                    cnt += 1
                if x in divs[a * b]:
                    cnt += 1
        return cnt

    # Greedy: repeatedly add the candidate with fewest conflicts
    remaining = small_candidates[:]
    while remaining and _elapsed() < _DEADLINE - 10:
        best_x = None
        best_conf = float('inf')
        for x in remaining:
            c = conflict_count(x, S)
            if c == 0:
                best_x = x
                best_conf = 0
                break
            if c < best_conf:
                best_conf = c
                best_x = x
        if best_x is not None and best_conf == 0:
            S.append(best_x)
            S_set.add(best_x)
            remaining.remove(best_x)
        else:
            break

    # Phase 3: Local search with swaps to improve
    # Try 1-remove-2-add moves and 2-remove-3-add moves
    rng = random.Random(42)

    def try_add(x, S, S_set, divs):
        if x in S_set:
            return False
        if _can_add(x, S, divs):
            S.append(x)
            S_set.add(x)
            return True
        return False

    # Local search: try to add any remaining small number
    improved = True
    while improved and _elapsed() < _DEADLINE - 10:
        improved = False
        for x in small_candidates:
            if x not in S_set:
                if _can_add(x, S, divs):
                    S.append(x)
                    S_set.add(x)
                    improved = True

    # Try swap moves: remove one element, try to add two others
    for _ in range(500):
        if _elapsed() > _DEADLINE - 10:
            break
        if len(S) < 3:
            break
        # Pick a random element to potentially remove
        idx = rng.randrange(len(S))
        rem = S[idx]
        # Try to find two elements that can replace it
        S_no_rem = S[:idx] + S[idx+1:]
        S_no_rem_set = set(S_no_rem)

        # Find candidates that can be added
        addable = []
        for x in small_candidates:
            if x not in S_no_rem_set and x != rem:
                if _can_add(x, S_no_rem, divs):
                    addable.append(x)

        if len(addable) >= 2:
            # Try pairs
            found = False
            for i in range(min(len(addable), 20)):
                for j in range(i + 1, min(len(addable), 20)):
                    a1, a2 = addable[i], addable[j]
                    # Check both can be added together
                    S_test = S_no_rem + [a1]
                    S_test_set = S_no_rem_set | {a1}
                    if _can_add(a2, S_test, divs):
                        S = S_test + [a2]
                        S_set = S_test_set | {a2}
                        found = True
                        break
                if found:
                    break
            if found:
                improved = True

    # Phase 4: More aggressive local search with multiple restarts
    best_S = S[:]
    best_set = S_set.copy()

    for restart in range(100):
        if _elapsed() > _DEADLINE - 5:
            break

        # Start from best
        cur = best_S[:]
        cur_set = set(best_S)

        # Perturbation: remove 1-3 random elements
        if len(cur) > 10:
            for _ in range(rng.randint(1, 3)):
                if cur:
                    r = rng.choice(cur)
                    cur.remove(r)
                    cur_set.discard(r)

        # Greedily add back
        for x in uni:
            if _elapsed() > _DEADLINE - 5:
                break
            if x not in cur_set and _can_add(x, cur, divs):
                cur.append(x)
                cur_set.add(x)

        if len(cur) > len(best_S):
            best_S = cur[:]
            best_set = cur_set.copy()

    # Phase 5: Final greedy pass on best solution
    cur = best_S[:]
    cur_set = set(best_S)
    for x in uni:
        if _elapsed() > _DEADLINE - 2:
            break
        if x not in cur_set and _can_add(x, cur, divs):
            cur.append(x)
            cur_set.add(x)

    if len(cur) > len(best_S):
        best_S = cur[:]

    # Verify
    result = best_S
    if not _is_valid(result, divs):
        # Fallback: rebuild from scratch
        result = large[:]
        res_set = set(result)
        for x in small_candidates:
            if _can_add(x, result, divs):
                result.append(x)
                res_set.add(x)

    # Ensure we have at least one element
    if not result:
        result = [256]

    # Convert to plain Python ints
    return [int(x) for x in result]
