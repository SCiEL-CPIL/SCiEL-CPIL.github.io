# EVOLVE-BLOCK-START
import time
import math
from itertools import permutations
from fractions import Fraction

def construct():
    t0 = time.monotonic()
    DEADLINE = t0 + 60.0   # hard kill guard; return best before this with margin

    N = 24
    # LCM(1..24) so every reciprocal 1/k is an exact integer
    L = 1
    for k in range(2, N + 1):
        L = L * k // math.gcd(L, k)

    # ---- Precompute every subset sum of the full universe [1..24] ----
    # A subset's sum, in units of 1/L, equals sum( L // k  for k in subset ).
    # Scaled value is integer in [0, L]; L ~ 5.35e9 fits in a signed int64.
    vals = [L // k for k in range(1, N + 1)]
    all_sums = [0]
    for k in range(N):
        base = len(all_sums)
        for i in range(base):
            all_sums.append(all_sums[i] + vals[k])
    all_sums_set = set(all_sums)
    inv = [L // k for k in range(1, N + 1)]          # scaled 1/k
    maskbit = [1 << (k - 1) for k in range(1, N + 1)]

    # Check: is target (a Fraction) a subset sum of elements indexed by `mask`?
    # True only if some subset of those elements sums to target.
    def is_subset_sum(target, mask):
        if target <= 0:
            return False
        tv = target.numerator * L // target.denominator  # exact (1/k divides L)
        # fast reject: subset can't reach target
        sm = 0
        m = mask
        while m:
            b = m & (-m)
            sm += vals[(b.bit_length() - 1)]
            m -= b
        if sm < tv:
            return False
        # exact test against precomputed table: target must be a sum of some
        # subset of the universe, and that subset must be contained in mask.
        for s in all_sums:
            if s == tv:
                if (s & ~mask) == 0:
                    return True
        return False

    FULL_MASK = (1 << N) - 1

    # Greedy over a fixed ordering: keep every element that stays valid.
    def greedy(order):
        S = []
        mask = 0
        for x in order:
            xb = maskbit[x - 1]
            xm = mask ^ xb
            # (a) 1/x must not be a sum of a subset of the current set
            if is_subset_sum(Fraction(1, x), mask):
                continue
            ok = True
            for y in S:
                # adding x would break y iff some subset of (S \ {y}) sums to 1/y - 1/x
                if is_subset_sum(Fraction(1, y) - Fraction(1, x), xm):
                    ok = False
                    break
            if ok:
                S.append(x)
                mask |= xb
        return S

    # Seed the best with the natural greedy (largest -> smallest).
    best = greedy(range(N, 0, -1))
    best_size = len(best)

    # Search across all 24! orderings; keep the largest valid set found.
    t = t0
    for order in permutations(range(N, 0, -1)):
        S = greedy(order)
        if len(S) > best_size:
            best = S
            best_size = len(S)
        t = time.monotonic()
        if t > DEADLINE:
            break

    # Safety margin: stop a little before the kill to leave room to return.
    return best
# EVOLVE-BLOCK-END
