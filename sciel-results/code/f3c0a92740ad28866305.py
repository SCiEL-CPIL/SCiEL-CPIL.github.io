"""Snake-in-the-box Q_9: bitset backtracking with randomized restarts."""
import time

N = 9
SIZE = 1 << N  # 512
NEIGH = [0] * SIZE
for _v in range(SIZE):
    nb = 0
    for _b in range(N):
        nb |= 1 << (_v ^ (1 << _b))
    NEIGH[_v] = nb


def _extend(start, time_limit, rng):
    """Extend from an initial path (list) as long as possible using bitset state."""
    # Build bitset state from the starting path.
    allowed = (1 << SIZE) - 1
    used = 0
    for v in start:
        allowed &= ~(NEIGH[v] | (1 << v))
        used |= 1 << v
    path = list(start)
    cur = path[-1]
    while True:
        # candidates = neighbors of cur that are still allowed
        cand = NEIGH[cur] & allowed
        if cand == 0:
            break
        # Score each candidate: prefer those that keep the most future options.
        # A cheap proxy: number of allowed neighbors of the candidate (its "out-degree").
        # Pick best (tie broken randomly).
        best = -1
        best_score = -1
        best_rand = 0.0
        c = cand
        while c:
            low = c & (-c)
            c ^= low
            v = low.bit_length() - 1
            score = (NEIGH[v] & allowed).bit_count()
            # small random tie-break / exploration
            r = rng.random()
            if score > best_score or (score == best_score and r > best_rand):
                best_score = score
                best_rand = r
                best = v
        if best < 0:
            break
        # commit
        allowed &= ~(NEIGH[best] | (1 << best))
        used |= 1 << best
        path.append(best)
        cur = best
    return path


def construct():
    deadline = time.time() + 110.0  # safety margin under 120s
    import random
    best = [0]

    # A few deterministic seeds to establish a baseline quickly.
    # Start from 0 with a greedy low-bit preference (reproduces ~104).
    greedy = [0]
    allowed = (1 << SIZE) - 1
    allowed &= ~(NEIGH[0] | 1)
    cur = 0
    while True:
        cand = NEIGH[cur] & allowed
        if cand == 0:
            break
        # lowest bit set
        low = cand & (-cand)
        v = low.bit_length() - 1
        allowed &= ~(NEIGH[v] | (1 << v))
        greedy.append(v)
        cur = v
    if len(greedy) > len(best):
        best = greedy

    # Randomized restarts: randomize the first few moves to escape local structure.
    rng = random.Random(12345)
    while time.time() < deadline:
        # Build a random short prefix by random-walking a few steps.
        depth = rng.randint(3, 8)
        start = [0]
        allowed = (1 << SIZE) - 1
        allowed &= ~(NEIGH[0] | 1)
        cur = 0
        ok = True
        for _ in range(depth):
            cand = NEIGH[cur] & allowed
            if cand == 0:
                ok = False
                break
            # random choice among candidates
            c = cand
            n = c.bit_count()
            k = rng.randrange(n)
            for _ in range(k):
                low = c & (-c)
                c ^= low
            low = c & (-c)
            v = low.bit_length() - 1
            allowed &= ~(NEIGH[v] | (1 << v))
            start.append(v)
            cur = v
        if not ok:
            continue
        # Extend greedily with randomized scoring.
        p = _extend(start, deadline, rng)
        if len(p) > len(best):
            best = p
            if len(best) >= 200:
                # good enough; keep searching a bit more but this is strong
                pass
    return best
