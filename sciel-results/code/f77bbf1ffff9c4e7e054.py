import time
import random

N = 256
DEADLINE = 80.0  # safety margin under the 90s kill

# Precompute all 4-APs (a, a+d, a+2d, a+3d) and their member indices
def _build_4aps():
    aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                aps.append((a, b, c, e))
    return aps

_4APS = _build_4aps()

# For each element, list of (ap_index, position_in_ap)
_ELEMENT_APS = [[] for _ in range(N + 1)]
for _i, _ap in enumerate(_4APS):
    for _pos, _val in enumerate(_ap):
        _ELEMENT_APS[_val].append((_i, _pos))

def _make_3ap_counter():
    """Return a function counting 3-APs (a,a+d,a+2d), d>0, in a set of 1..N."""
    pass

def _count_3aps(s):
    """Count distinct 3-term APs with positive difference in set s."""
    cnt = 0
    lst = sorted(s)
    sset = set(s)
    for i in range(len(lst)):
        a = lst[i]
        # a as first term
        # a, a+d, a+2d
        for d in range(1, (N - a) // 2 + 1):
            if (a + d) in sset and (a + 2 * d) in sset:
                cnt += 1
    return cnt

def _count_3aps_fast(s):
    """Faster 3-AP count using array membership."""
    cnt = 0
    for a in s:
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if b in s and c in s:
                cnt += 1
    return cnt

def _greedy_initial(rng):
    """Build a 4-AP-free set via randomized greedy, biased toward 3-AP density."""
    selected = set()
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    # Process in a mix: try each candidate, add if it doesn't create a 4-AP.
    # We'll do a forward pass in shuffled order.
    for x in candidates:
        # Check adding x doesn't complete any 4-AP
        ok = True
        for (ai, pos) in _ELEMENT_APS[x]:
            ap = _4APS[ai]
            # If all other 3 members are in selected, adding x completes it
            others = [ap[j] for j in range(4) if j != pos]
            if all(o in selected for o in others):
                ok = False
                break
        if ok:
            selected.add(x)
    return selected

def _remove_4ap_violators(s):
    """Remove elements to break all 4-APs (greedy hitting set)."""
    s = set(s)
    changed = True
    while changed:
        changed = False
        # Find a 4-AP fully in s
        for ap in _4APS:
            if all(x in s for x in ap):
                # Remove the element in this AP that is in the most 4-APs (to break many)
                # Actually remove one that minimally hurts; just remove last
                # Choose the element with highest 4-AP degree among the 4
                degs = [len(_ELEMENT_APS[x]) for x in ap]
                rem = ap[degs.index(max(degs))]
                s.remove(rem)
                changed = True
                break
    return s

def construct():
    start = time.time()
    rng = random.Random(12345)

    # Build a few diverse initial solutions
    inits = []
    # Greedy forward (original seed behavior) for reference
    seed_set = set()
    for x in range(1, N + 1):
        if all(not all(x - j * d in seed_set for j in (1, 2, 3)) for d in range(1, (x - 1) // 3 + 1)):
            seed_set.add(x)
    inits.append(seed_set)

    # Several randomized greedy
    for _ in range(6):
        inits.append(_greedy_initial(rng))

    # Evaluate, keep best
    def score(s):
        return _count_3aps_fast(s)

    best = max(inits, key=score)
    best_score = score(best)

    # Simulated annealing with add/remove/swap moves
    T = 25.0
    T_min = 0.1
    cur = set(best)
    cur_score = best_score

    def elapsed():
        return time.time() - start

    while elapsed() < DEADLINE:
        # Pick a move type
        move = rng.random()
        if move < 0.34:
            # Add a random element not in cur
            avail = [x for x in range(1, N + 1) if x not in cur]
            if not avail:
                continue
            x = rng.choice(avail)
            new = set(cur)
            new.add(x)
        elif move < 0.67:
            # Remove a random element
            if not cur:
                continue
            x = rng.choice(list(cur))
            new = set(cur)
            new.discard(x)
        else:
            # Swap: remove one, add one
            if not cur:
                continue
            x = rng.choice(list(cur))
            avail = [y for y in range(1, N + 1) if y not in cur]
            if not avail:
                continue
            y = rng.choice(avail)
            new = set(cur)
            new.discard(x)
            new.add(y)

        # Check 4-AP validity
        valid = True
        for ap in _4APS:
            if all(v in new for v in ap):
                valid = False
                break

        if not valid:
            # Reject invalid (or heavy penalty)
            T = max(T_min, T * 0.9999)
            continue

        new_score = score(new)
        delta = new_score - cur_score
        if delta > 0 or rng.random() < pow(2.718281828459045, delta / max(T, 1e-9)):
            cur = new
            cur_score = new_score
            if cur_score > best_score:
                best = set(cur)
                best_score = cur_score
        T = max(T_min, T * 0.9999)

    # Final local improvement pass: try adding any element that keeps 4-AP-free and helps
    # (greedy add)
    improved = True
    while improved and elapsed() < DEADLINE:
        improved = False
        for x in range(1, N + 1):
            if x in best:
                continue
            new = set(best)
            new.add(x)
            valid = True
            for ap in _4APS:
                if all(v in new for v in ap):
                    valid = False
                    break
            if valid:
                ns = score(new)
                if ns >= best_score:
                    best = new
                    best_score = ns
                    improved = True
        # try removing elements that don't reduce score
        for x in list(best):
            new = set(best)
            new.discard(x)
            ns = score(new)
            if ns == best_score and len(new) < len(best):
                # keep smaller only if same score; prefer larger, so skip
                pass

    # Ensure nonempty
    if not best:
        best = {1}

    return sorted(best)
