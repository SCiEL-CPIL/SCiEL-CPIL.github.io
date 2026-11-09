"""Rewritten program: fast validity + stochastic local search (annealing) on the conflict structure."""
# EVOLVE-BLOCK-START
import time
import random

N = 256
TAIL_START = 171
DEADLINE = time.time() + 80.0  # safety margin before the 90s kill


def _build_masks():
    """Precompute, for each element a, the forbidden b+c multiples as bitmasks.
    For a to be the 'a' of a triple (a < b <= c, a | b+c), the pair (b,c) must
    sum to a multiple of a. We store for each a a list of bitmasks, one per
    forbidden sum s (multiple of a), where bit i is set iff i is a valid
    'c' value (i >= a+2, i <= N) and (s - i) is a valid 'b' value (a+1 <= s-i <= i).
    """
    masks = []
    for a in range(1, N + 1):
        lst = []
        for s in range(2 * (a + 1), 2 * N + 1, a):
            m = 0
            lo = max(a + 1, s - N)
            hi = min(N, (s - 1) // 2)
            for c in range(lo, hi + 1):
                b = s - c
                if b < a or b > c:
                    continue
                m |= 1 << c
            if m:
                lst.append(m)
        masks.append(lst)
    return masks


def _valid_full(S):
    S_list = sorted(S)
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i + 1, m):
            b = S_list[j]
            for k in range(j, m):
                c = S_list[k]
                if a < b <= c and (b + c) % a == 0:
                    return False
    return True


def construct():
    masks = _build_masks()
    tail = set(range(TAIL_START, N + 1))

    # --- Fast validity check using bitmasks ---
    def can_add(S, x):
        # x as 'a': no (b,c) in S with b>c>x and b+c a multiple of x
        for m in masks[x - 1]:
            if m & S:
                return False
        # x as 'b': no (a,c) in S with a<x<=c and a | (x+c)
        for a in S:
            if a >= x:
                continue
            if (x + c0) % a == 0:
                pass
            # need c in S, c>=x, (x+c)%a==0  ->  c ≡ -x (mod a)
            c = (-x) % a
            if c < x:
                c += a
            while c <= N:
                if c in S:
                    return False
                c += a
        # x as 'c': no (a,b) in S with a<b<=x and a | (b+x)
        for a in S:
            b = (-x) % a
            if b <= a:
                b += a
            while b <= x:
                if b in S:
                    return False
                b += a
        return True

    def can_remove(S, x):
        # After removing x, check all triples among remaining elements.
        T = S - {x}
        T_list = sorted(T)
        m = len(T_list)
        for i in range(m):
            a = T_list[i]
            for j in range(i + 1, m):
                b = T_list[j]
                for k in range(j, m):
                    c = T_list[k]
                    if a < b <= c and (b + c) % a == 0:
                        return False
        return True

    def full_check(S):
        # O(n^3) but only on ~100 elems; used sparingly.
        return _valid_full(S)

    # --- Seed: greedy ascending add over the tail ---
    current = set(tail)
    for x in range(1, TAIL_START):
        if can_add(current, x):
            current.add(x)
    best = set(current)
    best_size = len(current)
    if not full_check(best):
        best = set(tail)
        best_size = len(tail)

    # --- Simulated annealing: random add/remove moves ---
    rng = random.Random(12345)
    cur = set(best)
    cur_size = best_size
    T = 2.0
    T_min = 0.02
    cooling = 0.999
    it = 0
    while time.time() < DEADLINE:
        it += 1
        # pick a random move
        if rng.random() < 0.5:
            # try to add a random element not in cur
            cand = rng.randrange(1, N + 1)
            if cand in cur:
                continue
            if can_add(cur, cand):
                cur.add(cand)
                cur_size += 1
                if cur_size > best_size:
                    best = set(cur)
                    best_size = cur_size
            else:
                # rejected add: cool
                T = max(T_min, T * cooling)
        else:
            # try to remove a random element (only non-tail to preserve size mostly)
            pool = [v for v in cur if v < TAIL_START]
            if not pool:
                pool = list(cur)
            if not pool:
                continue
            cand = rng.choice(pool)
            if can_remove(cur, cand):
                cur.remove(cand)
                cur_size -= 1
                # accept removal only if it helps later; accept with prob based on T
                # (removal always reduces size, so accept with small prob to escape)
                if rng.random() < T:
                    pass  # keep removed
                else:
                    cur.add(cand)
                    cur_size += 1
        # occasional cooling
        if it % 50 == 0:
            T = max(T_min, T * cooling)
        if it % 500 == 0 and time.time() < DEADLINE - 3:
            if not full_check(cur):
                cur = set(best)
                cur_size = best_size

    # --- Final deterministic polish: greedy re-add pass on best ---
    cur = set(best)
    for x in range(1, N + 1):
        if x in cur:
            continue
        if can_add(cur, x):
            cur.add(x)
    if full_check(cur):
        if len(cur) > best_size:
            best = cur
            best_size = len(cur)

    if not full_check(best):
        best = set(tail)

    return sorted(list(best))
# EVOLVE-BLOCK-END
