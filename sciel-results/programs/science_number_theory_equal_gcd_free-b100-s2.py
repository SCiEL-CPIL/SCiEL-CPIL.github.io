"""Equal-gcd-triple-free subsets of [1,128]: structural base + multi-order greedy
+ fast ball-based add checks + bounded 1-opt/2-opt local search.

Constraint: no three distinct a,b,c in S with gcd(a,b)==gcd(a,c)==gcd(b,c).
Adding x to a valid S is forbidden iff exists g and pair (a,b) in S with
gcd(x,a)=gcd(x,b)=gcd(a,b)=g.  We precompute, per (x,g), the list of y with
gcd(x,y)=g, so the check is: for some g, |S ∩ B_g(x)| >= 2 AND some pair in
that intersection has gcd g.  This is cheap and frees budget for restarts.
"""
import time
import random
from math import gcd
from itertools import combinations

_DEADLINE = 60.0          # wall-clock safety margin under the 90 s kill
_START = None
_CAND = list(range(1, 129))
_PRE = None               # pairwise gcd matrix, indexed 0..128
_BALL = None              # BALL[x][g] = list of y in [1,128] with gcd(x,y)==g
_PRIMEP = None            # prime-power flag per value
_GCD1 = None              # GCD1[x] = list of y in [1,128] with gcd(x,y)==1


def _ensure_pre():
    global _PRE, _BALL, _PRIMEP, _GCD1
    if _PRE is not None:
        return
    n = 128
    PRE = [[1] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            PRE[i][j] = PRE[j][i] = gcd(i, j)
    _PRE = PRE

    # Balls: for each x and each gcd value g, list of y with gcd(x,y)==g.
    BALL = {}
    for x in range(1, n + 1):
        d = {}
        for y in range(1, n + 1):
            g = PRE[x][y]
            d.setdefault(g, []).append(y)
        BALL[x] = d
    _BALL = BALL
    _GCD1 = {x: BALL[x].get(1, []) for x in range(1, n + 1)}

    # prime-power flags
    flag = [False] * (n + 1)
    for x in range(2, n + 1):
        v = x
        f0 = None
        ok = True
        p = 2
        while p * p <= v:
            if v % p == 0:
                if f0 is None:
                    f0 = p
                else:
                    ok = False
                    break
                while v % p == 0:
                    v //= p
            p += 1
        if ok and (v == 1 or v == f0):
            flag[x] = True
    _PRIMEP = flag


def _num_divisors(n):
    c = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            c += 1
            if i * i != n:
                c += 1
        i += 1
    return c


def _bad_triples_in(S):
    """Count equal-gcd triples within S (each triple counted once)."""
    P = _PRE
    s = list(S)
    n = len(s)
    cnt = 0
    for i in range(n):
        a = s[i]
        for j in range(i + 1, n):
            b = s[j]
            g_ab = P[a][b]
            for k in range(j + 1, n):
                c = s[k]
                if P[a][c] == g_ab and P[b][c] == g_ab:
                    cnt += 1
    return cnt


def _can_add(Sset, x):
    """True if adding x to valid set Sset creates no equal-gcd triple."""
    P = _PRE
    B = _BALL[x]
    for g, lst in B.items():
        # gather S elements in this ball (excluding x itself, which is not in S)
        inball = [u for u in Sset if u in set(lst)]
        if len(inball) < 2:
            continue
        # need a pair in inball whose gcd is g
        m = len(inball)
        for i in range(m):
            a = inball[i]
            for j in range(i + 1, m):
                b = inball[j]
                if P[a][b] == g:
                    return False
    return True


def _can_add_list(Slist, x):
    """List-based fast check (Slist is a plain list, no set membership)."""
    P = _PRE
    B = _BALL[x]
    for g, lst in B.items():
        if len(lst) < 2:
            continue
        inball = []
        for u in Slist:
            if u == x:
                continue
            if P[x][u] == g:
                inball.append(u)
        if len(inball) < 2:
            continue
        m = len(inball)
        for i in range(m):
            a = inball[i]
            for j in range(i + 1, m):
                b = inball[j]
                if P[a][b] == g:
                    return False
    return True


def _swap_valid(Slist, r, x):
    """Check Slist - {r} + {x} is valid. Original set is valid, so only new
    triples involve x; check pairs (a,b) in Slist\\{r}."""
    P = _PRE
    B = _BALL[x]
    for g, lst in B.items():
        if len(lst) < 2:
            continue
        inball = []
        for u in Slist:
            if u == r or u == x:
                continue
            if P[x][u] == g:
                inball.append(u)
        if len(inball) < 2:
            continue
        m = len(inball)
        for i in range(m):
            a = inball[i]
            for j in range(i + 1, m):
                b = inball[j]
                if P[a][b] == g:
                    return False
    return True


def _greedy(order):
    S = []
    for x in order:
        if _can_add_list(S, x):
            S.append(x)
    return S


def _local_add(S):
    """Repeatedly add any excluded element that fits until none remain."""
    S = list(S)
    improved = True
    while improved:
        improved = False
        for x in _CAND:
            if x not in S and _can_add_list(S, x):
                S.append(x)
                improved = True
    return S


def _local_search(S):
    """Bounded local search: adds, then targeted swaps to enable more adds."""
    P = _PRE
    S = list(S)
    Sset = set(S)
    max_rounds = 50
    for _ in range(max_rounds):
        if time.time() - _START > _DEADLINE - 6:
            break
        # (a) Try all adds until no more fit
        added = True
        while added:
            added = False
            for x in _CAND:
                if x not in Sset and _can_add_list(S, x):
                    S.append(x)
                    Sset.add(x)
                    added = True
        if time.time() - _START > _DEADLINE - 6:
            break
        # (b) 1-opt swap: find a swap that enables at least one new add
        swap_found = False
        for r in list(S):
            if time.time() - _START > _DEADLINE - 6:
                break
            for x in _CAND:
                if x in Sset or x == r:
                    continue
                if _swap_valid(S, r, x):
                    # Tentatively swap and check if it enables any add
                    S.remove(r)
                    Sset.discard(r)
                    S.append(x)
                    Sset.add(x)
                    # Check if any new element can be added
                    can_add_new = False
                    for y in _CAND:
                        if y not in Sset and _can_add_list(S, y):
                            can_add_new = True
                            break
                    if can_add_new:
                        swap_found = True
                        break
                    else:
                        # Revert
                        S.remove(x)
                        Sset.discard(x)
                        S.append(r)
                        Sset.add(r)
            if swap_found:
                break
        if not swap_found:
            break
    return S


def _construct():
    global _START
    _ensure_pre()
    _START = time.time()
    P = _PRE
    PRIMEP = _PRIMEP

    prime_powers = [x for x in range(1, 129) if PRIMEP[x]]
    divs = {x: _num_divisors(x) for x in range(1, 129)}

    best = []
    best_len = -1

    # --- Multi-order greedy (inspiration 1) + local add ---
    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        sorted(range(1, 129), key=lambda x: divs[x]),
        sorted(range(1, 129), key=lambda x: divs[x], reverse=True),
        sorted(range(1, 129), key=lambda x: (0 if PRIMEP[x] else 1, x)),
        prime_powers + [x for x in range(1, 129) if not PRIMEP[x]],
        sorted(range(1, 129), key=lambda x: (divs[x], x)),
        sorted(range(1, 129), key=lambda x: (-divs[x], x)),
    ]
    for order in orders:
        if time.time() - _START > _DEADLINE - 8:
            break
        S = _greedy(order)
        S = _local_add(S)
        if len(S) > best_len:
            best = S
            best_len = len(S)

    # --- Local search on the best base ---
    if time.time() - _START < _DEADLINE - 10:
        S = _local_search(best)
        if len(S) > best_len:
            best = S
            best_len = len(S)

    # --- Random restarts: biased random orders, greedy + local search ---
    rng = random.Random(12345)
    while time.time() - _START < _DEADLINE - 10:
        # Mix of strategies: prime powers first, divisor-based, random
        strategy = rng.randint(0, 2)
        if strategy == 0:
            pp = list(prime_powers)
            rest = [x for x in range(1, 129) if not PRIMEP[x]]
            rng.shuffle(pp)
            rng.shuffle(rest)
            order = pp + rest
        elif strategy == 1:
            order = sorted(range(1, 129), key=lambda x: (divs[x], rng.random()))
        else:
            order = list(range(1, 129))
            rng.shuffle(order)
        S = _greedy(order)
        S = _local_add(S)
        if len(S) > best_len:
            best = S
            best_len = len(S)
        if time.time() - _START < _DEADLINE - 12:
            S = _local_search(S)
            if len(S) > best_len:
                best = S
                best_len = len(S)

    # --- Final safety: ensure validity and nonempty ---
    if _bad_triples_in(best) != 0:
        # fall back to guaranteed-valid greedy
        best = _greedy(list(range(1, 129)))
        best = _local_add(best)
    if not best:
        best = [2]
    return best


def construct():
    from itertools import combinations  # noqa: F401
    from math import gcd  # noqa: F401
    try:
        res = _construct()
    except Exception:
        _ensure_pre()
        P = _PRE
        res = list(range(1, 129))
        while _bad_triples_in(res) != 0 and len(res) > 1:
            worst = res[0]
            worst_cnt = -1
            for cand in res:
                cnt = 0
                for a, b in combinations([y for y in res if y != cand], 2):
                    if P[cand][a] == P[cand][b] == P[a][b]:
                        cnt += 1
                if cnt > worst_cnt:
                    worst_cnt = cnt
                    worst = cand
            res.remove(worst)
        if not res:
            res = [2]
    return res
