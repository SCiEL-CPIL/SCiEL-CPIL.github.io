"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_START = time.time()
_DEADLINE = 150.0 - 5.0  # safety margin

def construct():
    # Branch-and-bound backtracking to minimize the largest element (diameter)
    # of a Sidon set containing 0 that realizes all differences 1..30.
    #
    # Representation:
    #   - We build a sorted list `a` of distinct integers, always with a[0] = 0.
    #   - At each step we try to append a new smallest possible element,
    #     maintaining the Sidon property (all positive differences distinct).
    #   - We track which differences 1..30 have been realized.
    #   - We prune when the current max element >= best found, or when the
    #     remaining differences cannot possibly be realized.
    #
    # The search is ordered by smallest-element-first, which is known to find
    # tight Golomb rulers efficiently.

    NEED = set(range(1, 31))  # differences that must be realized

    best = [None]  # best solution found (list of ints)
    best_val = [float('inf')]

    def elapsed_ok():
        return (time.time() - _START) < _DEADLINE

    # Precompute nothing heavy; do it inline for speed.

    def solve(a, covered, diff_set):
        # a: sorted list of chosen elements (a[0]==0)
        # covered: set of differences from 1..30 already realized
        # diff_set: set of ALL positive differences realized (for Sidon check)
        #
        # Prune 1: current max >= best
        cur_max = a[-1]
        if cur_max >= best_val[0]:
            return

        # Prune 2: can remaining differences be realized?
        # A difference d can be realized by a pair (x, x+d) where x is in the
        # current set OR a future element. The maximum possible difference
        # with the current max is limited, but future elements can extend it.
        # However, we must have at least |NEED - covered| more differences,
        # and each new element can create at most len(a) new differences
        # (one with each existing element). We also need enough "room".
        remaining = NEED - covered
        if not remaining:
            # All differences realized; this is a valid solution.
            if cur_max < best_val[0]:
                best_val[0] = cur_max
                best[0] = list(a)
            return

        # Feasibility: each remaining difference d must be <= the final diameter.
        # The final diameter is at least max(remaining) and at least cur_max.
        # But we can't exceed best_val[0], so we need max(remaining) < best_val[0].
        if max(remaining) >= best_val[0]:
            return

        # We need to add at least len(remaining) more differences.
        # Each new element added at position p creates (len(a)) new differences
        # (with each existing element). We need enough new elements.
        # Minimum new elements needed: ceil(len(remaining) / len(a))
        # Each new element must be > cur_max, so the final max is at least
        # cur_max + 1 (if we add one). But we also need to realize specific diffs.
        #
        # Stronger bound: to realize a difference d, we need two elements
        # differing by d. If d > cur_max, we need a new element at position
        # x+d where x is existing, or x and x+d both new. The smallest such
        # new element is at least d (if 0 is in set, x=0 gives x+d=d).
        # So the final max must be at least max(remaining).
        # We already checked that. Additionally, to realize ALL remaining
        # differences, we need the final set to be large enough.
        #
        # Let's use a simple bound: we need at least len(remaining) new
        # differences. Each new element gives at most len(a) new diffs.
        # So we need at least k = ceil(len(remaining) / len(a)) new elements.
        # The k-th new element (smallest possible) would be at least cur_max + k
        # (if we pack them tightly). But they must also satisfy Sidon.
        # This bound is weak; rely on the search.

        # Try to add the next element. Candidate positions:
        # We want to add the smallest possible new element > cur_max.
        # But we also want to realize specific differences.
        #
        # Strategy: try candidates in increasing order. For each candidate x,
        # check if adding x to a preserves the Sidon property and doesn't
        # exceed best_val[0]. Then recurse.
        #
        # To make this efficient, we generate candidates smartly:
        #   - x = cur_max + 1, cur_max + 2, ... up to best_val[0] - 1
        #   - Also, x that would realize a specific remaining difference d:
        #     if there exists y in a such that y + d is a valid candidate,
        #     or x - d is in a.
        #
        # For now, try a bounded range of candidates.

        if not elapsed_ok():
            return

        # Determine candidate range. We want to try small candidates first.
        # But also candidates that realize specific differences.
        # Let's try candidates from cur_max+1 up to min(best_val[0]-1, cur_max + 200)
        # and also specific candidates that realize remaining differences.

        candidates = set()
        # Small increments
        for delta in range(1, 50):
            c = cur_max + delta
            if c < best_val[0]:
                candidates.add(c)
        # Candidates that realize specific differences with existing elements
        for d in remaining:
            for y in a:
                c = y + d
                if c > cur_max and c < best_val[0]:
                    candidates.add(c)
                c2 = y - d
                if c2 >= 0 and c2 > cur_max and c2 < best_val[0]:
                    candidates.add(c2)

        # Sort candidates to try smallest first (tighter rulers)
        for x in sorted(candidates):
            if not elapsed_ok():
                return
            # Check Sidon: for each existing y, the difference |x - y| must
            # not already be in diff_set.
            ok = True
            new_diffs = set()
            for y in a:
                d = abs(x - y)
                if d in diff_set:
                    ok = False
                    break
                new_diffs.add(d)
            if not ok:
                continue
            # Also check that x is not already in a (it isn't, x > cur_max)
            new_a = a + [x]
            new_covered = covered | (new_diffs & NEED)
            new_diff_set = diff_set | new_diffs
            solve(new_a, new_covered, new_diff_set)
            if not elapsed_ok():
                return
            # If we found a solution with max == best_val[0], we can stop
            # (no better possible since we try increasing candidates)
            if best_val[0] <= cur_max + 1:
                return

    # Initial call: a = [0], covered = empty, diff_set = empty
    solve([0], set(), set())

    if best[0] is not None:
        return best[0]

    # Fallback: return the greedy solution (shouldn't happen, but safe)
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a
# EVOLVE-BLOCK-END
