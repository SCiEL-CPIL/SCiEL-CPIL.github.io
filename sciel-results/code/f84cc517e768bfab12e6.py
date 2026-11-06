"""Fresh design: local-search / simulated annealing over maximal no-isosceles sets."""
N = 16
import itertools, random, time, math

DEADLINE = time.time() + 280.0  # safety margin well under 330s

def _dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx*dx + dy*dy

def _isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _is_valid_set(S):
    pts = list(S)
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            for k in range(j+1, len(pts)):
                if _isosceles(pts[i], pts[j], pts[k]):
                    return False
    return True

def _is_maximal(S):
    Sset = set(S)
    for p in itertools.product(range(N), repeat=2):
        if p in Sset:
            continue
        if all(not _isosceles(p, a, b) for a, b in itertools.combinations(S, 2)):
            return False
    return True

def _completion(S):
    """Greedily add any point that keeps the set no-isosceles until maximal."""
    S = list(S)
    changed = True
    while changed:
        changed = False
        for p in itertools.product(range(N), repeat=2):
            if p in S:
                continue
            if all(not _isosceles(p, a, b) for a, b in itertools.combinations(S, 2)):
                S.append(p)
                changed = True
    return S

def _greedy_maximal(order):
    """Build a maximal set by scanning in a given order, adding if possible."""
    chosen = []
    for p in order:
        if all(not _isosceles(p, a, b) for a, b in itertools.combinations(chosen, 2)):
            chosen.append(p)
    return chosen

def construct():
    random.seed(12345)
    all_points = list(itertools.product(range(N), repeat=2))

    # Initial candidate: greedy in a random order (often smaller than row-major).
    best = None
    best_len = None

    def consider(S):
        nonlocal best, best_len
        S = _completion(S)
        if _is_maximal(S) and _is_valid_set(S):
            L = len(S)
            if best is None or L < best_len:
                best = S
                best_len = L
            return S
        return S

    # A few random-order greedy starts.
    for _ in range(40):
        order = all_points[:]
        random.shuffle(order)
        consider(_greedy_maximal(order))

    # Local search: try removing a random point; if the set stays no-isosceles,
    # re-complete to maximal; keep if it shrinks or ties (prefer shorter).
    cur = best[:]
    cur_len = len(cur)
    T = 0.5
    t0 = time.time()
    step = 0
    while time.time() < DEADLINE - 5.0 and best_len is not None:
        step += 1
        if len(cur) <= 1:
            break
        # pick a random point to try removing
        idx = random.randrange(len(cur))
        cand = cur[:idx] + cur[idx+1:]
        if not _is_valid_set(cand):
            # still might be fine after re-completion? no, invalid stays invalid
            continue
        cand = _completion(cand)
        if not _is_maximal(cand) or not _is_valid_set(cand):
            continue
        new_len = len(cand)
        delta = new_len - cur_len
        # accept if better, or with probability based on temperature
        if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-6)):
            cur = cand
            cur_len = new_len
            if best is None or cur_len < best_len:
                best = cur[:]
                best_len = cur_len
        # cool slowly
        if step % 200 == 0:
            T *= 0.97
        if T < 0.02:
            T = 0.02

    return best if best is not None else _completion(_greedy_maximal(all_points))
