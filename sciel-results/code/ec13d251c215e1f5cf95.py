"""Rewrite: clique-guided construction + full 4096 polish for 40 comparable subsets."""
# EVOLVE-BLOCK-START
import time

_START = time.time()
DEADLINE = _START + 83.0
N = 12
M = 40
TOTAL = 1 << N
FULL = TOTAL - 1


def _comparable(a, b):
    return (a & b) == a or (a & b) == b


def _build_subsets():
    # subsets[mask] = frozenset of set bits
    subs = [None] * TOTAL
    for m in range(TOTAL):
        s = set()
        x = m
        i = 0
        while x:
            if x & 1:
                s.add(i)
            x >>= 1
            i += 1
        subs[m] = frozenset(s)
    return subs


SUBSETS = _build_subsets()

# comp_bits[mask]: Python-int bitset of all masks comparable to `mask` (excluding itself)
COMP_BITS = [0] * TOTAL
for a in range(TOTAL):
    bits = 0
    for b in range(TOTAL):
        if a != b and _comparable(a, b):
            bits |= (1 << b)
    COMP_BITS[a] = bits


def _score(fam):
    c = 0
    m = len(fam)
    cb = COMP_BITS
    for i in range(m):
        ai = fam[i]
        for j in range(i + 1, m):
            if cb[ai] >> fam[j] & 1:
                c += 1
    return c


# ---------- Maximum chain via DP (chain is a clique in the comparability graph) ----------
def _max_chain(k, time_limit):
    # dp[s] = (best chain size, chain as list of masks) ending at subset s
    # Use size as primary; build chains greedily by size then by best comparability.
    # We want the largest chain up to size k (k <= M).
    best = {}  # mask -> (size, chain)
    for s in range(TOTAL):
        best[s] = (1, [s])
    # Process in increasing popcount so that subsets come before supersets
    order = list(range(TOTAL))
    order.sort(key=lambda x: (x.bit_count(), x))
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for s in order:
            size, chain = best[s]
            # try to extend: find any subset t of s (t != s) with best[t]
            # and prepend
            # To extend chain ending at s to a longer one, we need a mask t < s (subset)
            # Actually chain is increasing: t subset s means t comes before s.
            # So a chain ending at s can be extended to a longer chain by prepending best[t] for t subset s.
            # But we want the LONGEST chain, so process by size; a chain ending at s of size L
            # can be extended by any t subset s to size L + best[t][0].
            # We already computed best[t] for smaller t. So best[s] should be:
            # 1 + max over t subset s (t != s) of best[t][0], with chain = best[t].chain + [s]
            # Let's recompute properly:
            pass
    # Redo cleanly:
    best = {}
    for s in order:
        best_size = 1
        best_prev = None
        # iterate subsets of s
        t = (s - 1) & s
        while t != s:
            if t in best:
                if best[t][0] > best_size - 1:
                    best_size = best[t][0] + 1
                    best_prev = t
            t = (t - 1) & s
        if best_prev is not None:
            best[s] = (best_size, best[best_prev][1] + [s])
        else:
            best[s] = (1, [s])
    # Find the chain of length exactly k (or closest)
    # We want a chain of size k. Find the mask with best_size == k (or >= k) and take first k.
    target = k
    best_chain = None
    best_len = 0
    for s in order:
        sz, ch = best[s]
        if sz >= target:
            # take last target elements? No: chain is increasing, so first target is the prefix
            # Actually we want any k consecutive? No, we want a k-element chain.
            # Take the first k elements of ch (the smallest k in the chain).
            cand = ch[:target]
            if len(cand) == target:
                if best_chain is None or _score(cand) > _score(best_chain):
                    best_chain = cand
                    best_len = target
    if best_chain is not None:
        return best_chain
    # fallback: longest available
    for s in order:
        sz, ch = best[s]
        if sz > best_len:
            best_len = sz
            best_chain = ch
    return best_chain if best_chain else [0]


def _chain_initial():
    # Build a chain of size M using the DP
    chain = _max_chain(M, _START + 15.0)
    return chain[:M]


# ---------- Full 4096 polish (clique-growth move) ----------
def _full_polish(fam, time_limit):
    fam = fam[:]
    score = _score(fam)
    m = len(fam)
    # precompute per-position comparable count
    def comp_counts(f):
        cc = [0] * len(f)
        cb = COMP_BITS
        for i in range(len(f)):
            ai = f[i]
            for j in range(len(f)):
                if i != j and cb[ai] >> f[j] & 1:
                    cc[i] += 1
        return cc

    cc = comp_counts(fam)
    while time.time() < time_limit:
        improved = False
        for i in range(m):
            if time.time() > time_limit:
                break
            old = fam[i]
            # find best replacement
            best_new = old
            best_gain = 0
            # For each candidate new mask, compute comparable count to others
            used = set(fam)
            used.discard(old)
            for nm in range(TOTAL):
                if nm in used:
                    continue
                # comparable count of nm to fam[j] for j != i
                cnt = 0
                cbn = COMP_BITS[nm]
                for j in range(m):
                    if j != i:
                        if cbn >> fam[j] & 1:
                            cnt += 1
                gain = cnt - cc[i]
                if gain > best_gain:
                    best_gain = gain
                    best_new = nm
            if best_new != old and best_gain > 0:
                fam[i] = best_new
                score += best_gain
                cc = comp_counts(fam)
                improved = True
        if not improved:
            break
    return fam, score


# ---------- Random + local search for diversity ----------
def _random_local(fam, time_limit):
    import random
    fam = fam[:]
    score = _score(fam)
    m = len(fam)
    cb = COMP_BITS
    while time.time() < time_limit:
        i = random.randrange(m)
        nm = random.randrange(TOTAL)
        if nm == fam[i]:
            continue
        # check dup
        dup = False
        for k in range(m):
            if k != i and fam[k] == nm:
                dup = True
                break
        if dup:
            continue
        old = fam[i]
        old_cnt = 0
        for j in range(m):
            if j != i and cb[old] >> fam[j] & 1:
                old_cnt += 1
        new_cnt = 0
        for j in range(m):
            if j != i and cb[nm] >> fam[j] & 1:
                new_cnt += 1
        if new_cnt > old_cnt:
            fam[i] = nm
            score += (new_cnt - old_cnt)
    return fam, score


def construct():
    # Phase 1: chain-based initial (fast, strong)
    fam = _chain_initial()
    # ensure size M
    if len(fam) < M:
        used = set(fam)
        for x in range(TOTAL):
            if x not in used:
                fam.append(x)
                used.add(x)
            if len(fam) == M:
                break
    if len(fam) > M:
        fam = fam[:M]

    best_fam = fam[:]
    best_score = _score(fam)

    # Phase 2: full polish (clique-growth)
    t2 = min(DEADLINE - 20.0, _START + 35.0)
    if time.time() < t2:
        f2, s2 = _full_polish(best_fam, t2)
        if s2 > best_score:
            best_score = s2
            best_fam = f2

    # Phase 3: random local search for diversity
    t3 = min(DEADLINE - 15.0, _START + 55.0)
    if time.time() < t3:
        f3, s3 = _random_local(best_fam, t3)
        if s3 > best_score:
            best_score = s3
            best_fam = f3

    # Phase 4: final full polish
    t4 = DEADLINE - 4.0
    if time.time() < t4:
        f4, s4 = _full_polish(best_fam, t4)
        if s4 > best_score:
            best_score = s4
            best_fam = f4

    # Convert to list of lists of ints
    result = []
    for mask in best_fam:
        subset = []
        for b in range(N):
            if (mask >> b) & 1:
                subset.append(b)
        result.append(subset)
    for i in range(len(result)):
        result[i] = [int(x) for x in result[i]]
    return result

# EVOLVE-BLOCK-END
