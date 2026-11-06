# EVOLVE-BLOCK-START
import time
import random

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 135.0   # safety margin before 150s kill
    NEEDED = set(range(1, 31))

    best_set = None
    best_max = 10**6

    # A valid seed set (the original greedy) as a fallback / starting point
    def greedy():
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i+1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            placed = False
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    placed = True
                    break
            if not placed:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a

    seed = greedy()
    if all(d in (y - x for i, x in enumerate(seed) for y in seed[i+1:]) for d in NEEDED):
        best_set = seed
        best_max = max(seed)

    def valid_full(s):
        ds = set()
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                d = s[j] - s[i]
                if d in ds:
                    return False
                ds.add(d)
        return NEEDED.issubset(ds) and len(s) <= 32

    # ---- DFS building elements in increasing order ----
    # placed: list (increasing). used_diff: set of differences already used.
    # remaining_needed: set of needed diffs not yet covered.
    # We stop (success) when remaining_needed is empty (and len<=32).
    # Forward check: a needed diff d can still be covered only if there exists
    #   a placed a_i with a_i + d <= max, OR we can add two new elements.
    # We allow adding up to MAXN elements total.

    MAXN = 32
    rng = random.Random(12345)

    def solve_for_max(mx, max_nodes=2000000, seed_order=None):
        nonlocal best_set, best_max
        if mx < 30:  # need diff 30
            return None
        used_diff = set()
        remaining = set(NEEDED)
        placed = [0]
        # We'll choose candidate next values in increasing order from placed's max+1 .. mx.
        # To keep search tractable, candidate pool = values that create at least one new
        # needed difference, or are forced. We generate candidates on the fly.

        nodes = 0
        result = [None]

        def candidates():
            # values v > placed[-1] with v <= mx such that adding v keeps Sidon and
            # covers at least one remaining needed diff (v - a_i in remaining) OR
            # is otherwise useful. We'll just test all v in (placed[-1]+1 .. mx) but
            # that's too many. Instead generate from remaining diffs:
            cands = set()
            last = placed[-1]
            for a_i in placed:
                for d in remaining:
                    v = a_i + d
                    if last < v <= mx:
                        cands.add(v)
            return sorted(cands)

        def dfs():
            nonlocal nodes
            nodes += 1
            if nodes % 50000 == 0:
                if time.time() > DEADLINE - 3:
                    return
            if not remaining:
                result[0] = list(placed)
                return True
            if len(placed) >= MAXN:
                return False
            # lower bound: need at least k more elements to cover |remaining| diffs
            # each new element covers at most len(placed) new diffs (to existing) ... 
            # actually a new element x covers diffs x - a_i for all a_i in placed.
            # So with m more elements, max new diffs = sum over each of m of current+...
            # Simple bound: each new element adds at most (len(placed) + (m-1)) diffs.
            # We'll skip tight bound; rely on candidate pruning.
            cands = candidates()
            if not cands:
                return False
            for v in cands:
                # check Sidon with placed
                ok = True
                new_diffs = []
                for a_i in placed:
                    d = v - a_i
                    if d <= 0:
                        continue
                    if d in used_diff:
                        ok = False
                        break
                    new_diffs.append(d)
                if not ok:
                    continue
                # apply
                for d in new_diffs:
                    used_diff.add(d)
                    remaining.discard(d)
                placed.append(v)
                if dfs():
                    return True
                placed.pop()
                for d in new_diffs:
                    used_diff.discard(d)
                    if d in NEEDED:
                        remaining.add(d)
            return False

        ok = dfs()
        if ok and result[0] is not None:
            s = result[0]
            if valid_full(s):
                if max(s) < best_max:
                    best_max = max(s)
                    best_set = s
                return s
        return None

    # Iterative deepening over max, starting near the lower bound.
    # Lower bound on max given n elements: n(n-1)/2. We don't know n, but a set
    # covering 30 diffs needs at least 8 elements (8*7/2=28 <30, so 9: 9*8/2=36>=30).
    # So max >= 30 (diff 30) and >= 36 if 9 elements. Try a range.
    for mx in range(36, 600):
        if time.time() > DEADLINE - 10:
            break
        if mx < best_max:
            break
        s = solve_for_max(mx)
        if s is not None:
            # found a valid set with this max; try a few more to see if smaller works
            # but we iterate increasing, so first found is minimal in this sweep.
            pass

    # Final: if we have a valid best_set, return it; else seed.
    if best_set is not None and valid_full(best_set):
        return best_set
    return seed

# EVOLVE-BLOCK-END
