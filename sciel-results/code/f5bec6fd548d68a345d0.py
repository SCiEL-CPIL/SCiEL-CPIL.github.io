"""Small inclusion-maximal Sidon sets in [1,256]."""
import time
import random

N = 256
_DEADLINE = 55.0  # safety margin under 60s wall-clock


def _sums_of(s):
    """Return the set of all unordered pair sums a+b with a<=b in s."""
    sums = set()
    for i, a in enumerate(s):
        for b in s[i:]:
            sums.add(a + b)
    return sums


def _is_sidon(s):
    """Check all unordered pair sums a+b (a<=b) are distinct."""
    seen = set()
    for i, a in enumerate(s):
        for b in s[i:]:
            v = a + b
            if v in seen:
                return False
            seen.add(v)
    return True


def _is_maximal(s):
    """Check no missing element from [1,N] can be added without breaking Sidon."""
    sset = set(s)
    sums = _sums_of(s)
    for x in range(1, N + 1):
        if x in sset:
            continue
        # adding x: new sums are x+x and x+a for a in s
        if (2 * x) in sums:
            continue
        ok = True
        for a in s:
            if (x + a) in sums:
                ok = False
                break
        if ok:
            return False
    return True


def _valid(s):
    """A valid solution must be Sidon and inclusion-maximal."""
    if not s or len(set(s)) != len(s):
        return False
    if not _is_sidon(s):
        return False
    return _is_maximal(s)


def _greedy(order):
    """Ascending greedy saturation along a given order of [1..N]."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected


def _remove_if_sidon(s):
    """Try removing each element (in order) if the result stays Sidon.
    Returns a smaller Sidon set or the input."""
    s = list(s)
    changed = True
    while changed:
        changed = False
        for i in range(len(s)):
            trial = s[:i] + s[i + 1:]
            if _is_sidon(trial):
                s = trial
                changed = True
                break
    return s


def _try_add_back_to_maximize(s):
    """Given a Sidon set, greedily add back missing elements (in order)
    until no more can be added -> inclusion-maximal."""
    s = list(s)
    sset = set(s)
    sums = _sums_of(s)
    for x in range(1, N + 1):
        if x in sset:
            continue
        if (2 * x) in sums:
            continue
        ok = True
        for a in s:
            if (x + a) in sums:
                ok = False
                break
        if ok:
            s.append(x)
            sset.add(x)
            sums.add(2 * x)
            sums.update(x + a for a in s[:-1])
    return s


def construct():
    global _DEADLINE
    start = time.time()
    rng = random.Random(12345)

    best = None
    best_size = N + 1

    def consider(s):
        nonlocal best, best_size
        if not s:
            return
        s = list(dict.fromkeys(s))
        if _valid(s) and len(s) < best_size:
            best_size = len(s)
            best = s

    # 1) Baseline ascending greedy
    consider(_greedy(range(1, N + 1)))

    # 2) Diverse greedy initializations with different orders
    orders = []
    # reverse
    orders.append(list(range(N, 0, -1)))
    # random permutations
    for _ in range(6):
        o = list(range(1, N + 1))
        rng.shuffle(o)
        orders.append(o)
    # step patterns
    orders.append(list(range(1, N + 1, 2)) + list(range(2, N + 1, 2)))
    orders.append(list(range(2, N + 1, 3)) + list(range(1, N + 1, 3)))

    for o in orders:
        if time.time() > _DEADLINE:
            break
        consider(_greedy(o))

    # 3) Iterative removal + re-maximize from current best
    if best is not None:
        cur = list(best)
        for _ in range(40):
            if time.time() > _DEADLINE:
                break
            reduced = _remove_if_sidon(cur)
            maximal = _try_add_back_to_maximize(reduced)
            if _valid(maximal) and len(maximal) < best_size:
                best_size = len(maximal)
                best = maximal
                cur = list(best)
            else:
                # nudge: drop a random extra element to escape local minimum
                if len(reduced) < len(cur):
                    cur = reduced
                else:
                    break

    # 4) Local search: single-element swaps to shrink while staying maximal
    if best is not None:
        cur = list(best)
        no_improve = 0
        for _ in range(200):
            if time.time() > _DEADLINE:
                break
            improved = False
            for i in range(len(cur)):
                # try removing cur[i] then re-maximizing
                trial = cur[:i] + cur[i + 1:]
                maximal = _try_add_back_to_maximize(trial)
                if _valid(maximal) and len(maximal) < len(cur):
                    cur = maximal
                    improved = True
                    no_improve = 0
                    break
                elif _valid(maximal) and len(maximal) == len(cur) - 1:
                    pass
            if not improved:
                no_improve += 1
                if no_improve >= 5:
                    break
            if len(cur) < best_size:
                best_size = len(cur)
                best = cur

    if best is None:
        # ultimate fallback
        best = _greedy(range(1, N + 1))
    return best


# EVOLVE-BLOCK-END
