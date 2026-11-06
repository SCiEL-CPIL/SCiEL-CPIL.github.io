"""Snake-in-the-box Q_9: fast exclusion-set greedy + Warnsdorff lookahead + bidirectional splice."""
import time
import random

N = 9
SIZE = 1 << N  # 512
TIME_LIMIT = 115.0  # safety margin below 150s hard kill

# Precompute neighbors for every vertex (each has N neighbors)
NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(SIZE)]


def _valid(v, used, forbidden):
    return v not in used and v not in forbidden


def _greedy(rng, start, mode, deadline):
    """Build a snake by greedy extension from `start`.

    `mode` selects the candidate heuristic:
      0 = Warnsdorff (fewest onward options first)  -- classic snake heuristic
      1 = most-open (most onward options first)
      2 = random
    Maintains `used` (on-path) and `forbidden` (Hamming-distance-1 from any
    non-terminal vertex) so validity is an O(1) set membership test.
    """
    path = [start]
    used = {start}
    forbidden = set()

    while True:
        cur = path[-1]
        cands = [nb for nb in NEIGHBORS[cur] if _valid(nb, used, forbidden)]
        if not cands:
            break

        if mode == 2 or len(cands) == 1:
            nxt = cands[0] if len(cands) == 1 else rng.choice(cands)
        else:
            # Count onward options for each candidate (how many of its neighbors
            # remain available after we move to it).
            scored = []
            for c in cands:
                cnt = 0
                for nb in NEIGHBORS[c]:
                    if nb == cur:
                        continue
                    if _valid(nb, used, forbidden):
                        cnt += 1
                scored.append((cnt, c))
            if mode == 0:
                scored.sort(key=lambda t: (t[0], rng.random()))
            else:
                scored.sort(key=lambda t: (-t[0], rng.random()))
            nxt = scored[0][1]

        path.append(nxt)
        used.add(nxt)

        # cur becomes non-terminal: all its neighbors are now forbidden
        for nb in NEIGHBORS[cur]:
            forbidden.add(nb)

        if time.time() > deadline:
            break
    return path


def _try_splice(a, b, used_a, forbidden_a, rng, deadline):
    """Try to join snake `a` (head->tail) to snake `b` by reversing b and
    appending it, if the junction is a valid single-bit step and no new
    distance-1 conflict is introduced. Returns a longer path or None."""
    if len(a) + len(b) - 1 <= max(len(a), len(b)):
        return None
    # Junction: a[-1] to b[-1] must differ in exactly one bit.
    d = (a[-1] ^ b[-1]).bit_count()
    if d != 1:
        return None

    # Build merged path: a + reversed(b)
    merged = list(a) + list(reversed(b))
    used_m = set(merged)

    # Validate the induced property efficiently: walk and maintain forbidden.
    forbidden_m = set()
    ok = True
    for i in range(len(merged) - 1):
        cur = merged[i]
        nxt = merged[i + 1]
        if (cur ^ nxt).bit_count() != 1:
            ok = False
            break
        # nxt must not be forbidden (distance 1 from any earlier non-terminal)
        if nxt in forbidden_m:
            ok = False
            break
        for nb in NEIGHBORS[cur]:
            forbidden_m.add(nb)
    if not ok:
        return None
    return merged


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    rng = random.Random(12345)

    best = None
    best_len = 0

    def consider(p):
        nonlocal best, best_len
        if len(p) > best_len:
            best_len = len(p)
            best = p

    # 1) Deterministic Warnsdorff from 0 (fast, gives a solid baseline).
    p = _greedy(rng, 0, 0, deadline)
    consider(p)

    # 2) Randomized multi-start greedy across heuristics.
    modes = [0, 0, 0, 1, 2]  # weight toward Warnsdorff
    while time.time() < deadline - 4.0:
        start = rng.randrange(SIZE)
        mode = modes[rng.randrange(len(modes))]
        p = _greedy(rng, start, mode, deadline)
        consider(p)

    # 3) Bidirectional splicing: try to join pairs of good partial snakes.
    #    Collect a pool of recent candidate snakes for splicing.
    pool = []
    if best:
        pool.append(best)
    for _ in range(12):
        if time.time() > deadline - 2.0:
            break
        start = rng.randrange(SIZE)
        p = _greedy(rng, start, 0, deadline)
        pool.append(p)
        consider(p)

    # Attempt splices among pool members (small pool keeps this bounded).
    pool = sorted(pool, key=len, reverse=True)
    for i in range(len(pool)):
        if time.time() > deadline - 1.0:
            break
        for j in range(len(pool)):
            if i == j:
                continue
            m = _try_splice(pool[i], pool[j], None, None, rng, deadline)
            if m is not None:
                consider(m)

    if best is None:
        return [0]
    return best
