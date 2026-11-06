"""Search for a 12-state binary synchronizing automaton with a large reset threshold."""
import time
import random
from collections import deque

N = 12
DEADLINE = 70.0  # safety margin before the 90s kill

def _threshold(a, b):
    """Exact shortest reset-word length via BFS on the subset automaton.
    Returns None if not synchronizing."""
    full = (1 << N) - 1
    if a[full] == full or b[full] == full:
        return None
    dist = {full: 0}
    q = deque([full])
    while q:
        s = q.popleft()
        d = dist[s]
        for nxt in (a[s], b[s]):
            if nxt == 1:
                return d + 1
            if nxt not in dist:
                dist[nxt] = d + 1
                q.append(nxt)
    return None

def construct():
    # Precompute all subset images for the two letters.
    a = [0] * (1 << N)
    b = [0] * (1 << N)

    def build(fa, fb):
        for s in range(1, 1 << N):
            m = s
            da = 0
            db = 0
            while m:
                i = (m & -m).bit_length() - 1
                da |= 1 << fa[i]
                db |= 1 << fb[i]
                m &= m - 1
            a[s] = da
            b[s] = db

    best = 12
    best_fa = list(range(N))
    best_fb = [0] * N

    deadline = time.time() + DEADLINE
    rng = random.Random(20240517)

    # Family 1: "cycle" letter (a permutation) + random second letter.
    # Family 2: two random complete maps.
    for it in range(2000000):
        if time.time() > deadline:
            break
        if it % 2 == 0:
            fa = list(range(N))
            rng.shuffle(fa)
            fb = [rng.randrange(N) for _ in range(N)]
        else:
            fa = [rng.randrange(N) for _ in range(N)]
            fb = [rng.randrange(N) for _ in range(N)]
        build(fa, fb)
        t = _threshold(a, b)
        if t is not None and t > best:
            best = t
            best_fa = fa[:]
            best_fb = fb[:]
            # If we reach a very high threshold, keep a bit longer but stay safe.
            if best >= 100 and time.time() > deadline - 15.0:
                break

    return best_fa, best_fb
