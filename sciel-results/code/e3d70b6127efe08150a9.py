"""Program: find A, B (3 distinct ints in [1,128]) maximizing |A+B| with pairwise-coprime distinct sums."""
import math
import time
import random

# EVOLVE-BLOCK-START
def _sieve(n):
    is_prime = [True] * (n + 1)
    if n >= 0:
        is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:n + 1:step] = [False] * len(range(start, n + 1, step))
    return is_prime

# Precomputed at import time (fast, ~microseconds)
_IS_PRIME = _sieve(256)
_GCD = math.gcd

def _score_of(A, B):
    """Cardinality of distinct A+B if pairwise coprime, else 0."""
    seen = set()
    for a in A:
        for b in B:
            seen.add(a + b)
    n = len(seen)
    # pairwise coprime check
    lst = list(seen)
    for i in range(n):
        xi = lst[i]
        for j in range(i + 1, n):
            if _GCD(xi, lst[j]) > 1:
                return 0
    return n

def _ap_search(deadline):
    """Exhaustive search over AP triples A={a,a+d1,a+2d1}, B={b,b+d2,b+2d2}.
    Returns (A,B) if a full 9-prime solution exists, else (None, None)."""
    max_d = 63
    ip = _IS_PRIME
    for d1 in range(1, max_d + 1):
        if time.time() > deadline:
            return None, None
        a_max = 128 - 2 * d1
        if a_max < 1:
            continue
        for d2 in range(1, max_d + 1):
            if _GCD(d1, d2) != 1:
                continue
            if time.time() > deadline:
                return None, None
            b_max = 128 - 2 * d2
            if b_max < 1:
                continue
            offsets = [i * d1 + j * d2 for i in range(3) for j in range(3)]
            for a in range(1, a_max + 1):
                if time.time() > deadline:
                    return None, None
                for b in range(1, b_max + 1):
                    base = a + b
                    ok = True
                    for off in offsets:
                        s = base + off
                        if not ip[s]:
                            ok = False
                            break
                    if ok:
                        return [a, a + d1, a + 2 * d1], [b, b + d2, b + 2 * d2]
    return None, None

def _structured_search(deadline):
    """Structured non-AP search: A={a, a+p, a+q}, B={b, b+r, b+s}.
    Returns (best_score, best_A, best_B)."""
    ip = _IS_PRIME
    # offsets within [0, 127]; keep sums <= 256
    max_off = 40
    offsets = [o for o in range(1, max_off + 1)]
    pairs = [(p, q) for p in offsets for q in offsets if p < q]

    best_score = 0
    best_A = None
    best_B = None
    gcd = _GCD
    t = time.time()

    for p in offsets:
        for q in offsets:
            if p > q:
                continue
            for r in offsets:
                for s in offsets:
                    if r > s:
                        continue
                    if time.time() > deadline:
                        return best_score, best_A, best_B
                    a_max = 128 - q
                    b_max = 128 - s
                    if a_max < 1 or b_max < 1:
                        continue
                    for a in range(1, a_max + 1):
                        A = (a, a + p, a + q)
                        if time.time() > deadline:
                            return best_score, best_A, best_B
                        for b in range(1, b_max + 1):
                            B = (b, b + r, b + s)
                            # compute 9 sums
                            s00 = a + b
                            s01 = a + b + r
                            s02 = a + b + s
                            s10 = a + p + b
                            s11 = a + p + b + r
                            s12 = a + p + b + s
                            s20 = a + q + b
                            s21 = a + q + b + r
                            s22 = a + q + b + s
                            # all <= 256 by construction (a<=128-q, b<=128-s)
                            # distinct set
                            # use a small set
                            d = {s00, s01, s02, s10, s11, s12, s20, s21, s22}
                            n = len(d)
                            if n <= best_score:
                                continue
                            # pairwise coprime
                            ok = True
                            dl = list(d)
                            for i in range(n):
                                xi = dl[i]
                                for j in range(i + 1, n):
                                    if gcd(xi, dl[j]) > 1:
                                        ok = False
                                        break
                                if not ok:
                                    break
                            if ok:
                                best_score = n
                                best_A = list(A)
                                best_B = list(B)
                                if n >= 9:
                                    return best_score, best_A, best_B
    return best_score, best_A, best_B

def construct():
    t_start = time.time()
    KILL = 85.0  # safety margin before 90s kill

    # Start with a known-good seed
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = _score_of(best_A, best_B)

    # Phase 1: exhaustive AP search for a perfect 9 (fast if it exists)
    A_ap, B_ap = _ap_search(t_start + 20.0)
    if A_ap is not None:
        return [A_ap, B_ap]

    # Phase 2: structured non-AP search (covers non-AP triples)
    s2, A2, B2 = _structured_search(t_start + 65.0)
    if A2 is not None and s2 > best_score:
        best_score = s2
        best_A = A2
        best_B = B2
        if best_score >= 9:
            return [best_A, best_B]

    # Phase 3: random sampling of arbitrary triples if time remains
    rng = random.Random(42)
    t = time.time()
    while t < KILL - 2.0:
        A = rng.sample(range(1, 129), 3)
        B = rng.sample(range(1, 129), 3)
        s = _score_of(A, B)
        if s > best_score:
            best_score = s
            best_A = A
            best_B = B
            if best_score >= 9:
                return [best_A, best_B]
        if (rng.randint(0, 999) == 0):
            t = time.time()

    return [best_A, best_B]
# EVOLVE-BLOCK-END
