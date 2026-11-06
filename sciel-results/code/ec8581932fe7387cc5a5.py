"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

# --- Precomputation ---
N = 128
_PRE = [[1] * (N + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(i, N + 1):
        _PRE[i][j] = _PRE[j][i] = gcd(i, j)

# BALL[x][g] = list of y in [1,128] with gcd(x,y)==g
_BALL = {}
for x in range(1, N + 1):
    d = {}
    for y in range(1, N + 1):
        g = _PRE[x][y]
        if g not in d:
            d[g] = []
        d[g].append(y)
    _BALL[x] = d

# Prime-power flags
_PRIMEP = [False] * (N + 1)
for x in range(2, N + 1):
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
        _PRIMEP[x] = True
# 1 is a prime power by convention (or just include it)
_PRIMEP[1] = True

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

_DIVS = {x: _num_divisors(x) for x in range(1, N + 1)}
_CAND = list(range(1, N + 1))

def _can_add_list(Slist, x):
    """
    Fast check if x can be added to valid set Slist.
    Uses precomputed balls.
    """
    P = _PRE
    B = _BALL[x]
    for g, lst in B.items():
        # We need to check if there exist a, b in Slist such that:
        # 1. gcd(x, a) == g
        # 2. gcd(x, b) == g
        # 3. gcd(a, b) == g
        
        # Identify elements in Slist that have gcd g with x
        # Since Slist is small, we can just iterate.
        # Optimization: if the total number of candidates in the ball is small, iterate.
        # Here we iterate Slist and check P[x][u] == g.
        inball = []
        for u in Slist:
            if P[x][u] == g:
                inball.append(u)
        
        if len(inball) < 2:
            continue
        
        # Check if any pair in inball has gcd g
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
    Sset = set(S)
    improved = True
    while improved:
        improved = False
        for x in _CAND:
            if x not in Sset and _can_add_list(S, x):
                S.append(x)
                Sset.add(x)
                improved = True
    return S

def _swap_valid(Slist, r, x):
    """
    Check if Slist - {r} + {x} is valid.
    r is in Slist, x is not.
    """
    P = _PRE
    B = _BALL[x]
    for g, lst in B.items():
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

def _local_search(S):
    """
    Bounded local search: adds, then targeted swaps to enable more adds.
    """
    P = _PRE
    S = list(S)
    Sset = set(S)
    max_rounds = 50
    start_time = time.time()
    
    for _ in range(max_rounds):
        if time.time() - start_time > 40.0:
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
        
        if time.time() - start_time > 40.0:
            break
        
        # (b) 1-opt swap: find a swap that enables at least one new add
        swap_found = False
        # Iterate over a copy of S for removal
        for r in list(S):
            if time.time() - start_time > 40.0:
                break
            for x in _CAND:
                if x in Sset or x == r:
                    continue
                if _swap_valid(S, r, x):
                    # Tentatively swap
                    S.remove(r)
                    Sset.discard(r)
                    S.append(x)
                    Sset.add(x)
                    
                    # Check if any new element can be added
                    can_add_new = False
                    # We can be smarter here: check all candidates
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

def construct():
    deadline = time.time() + 80.0
    best = []
    
    # Generate diverse orders
    prime_powers = [x for x in range(1, N + 1) if _PRIMEP[x]]
    non_prime_powers = [x for x in range(1, N + 1) if not _PRIMEP[x]]
    
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: _DIVS[x]),
        sorted(range(1, N + 1), key=lambda x: _DIVS[x], reverse=True),
        prime_powers + non_prime_powers,
        non_prime_powers + prime_powers,
        sorted(range(1, N + 1), key=lambda x: (0 if _PRIMEP[x] else 1, x)),
    ]
    
    # Add some random orders
    import random
    rng = random.Random(42)
    for _ in range(10):
        order = list(range(1, N + 1))
        rng.shuffle(order)
        orders.append(order)
        
    # Run greedy + local search on each order
    for order in orders:
        if time.time() > deadline - 5.0:
            break
        s = _greedy(order)
        s = _local_add(s)
        s = _local_search(s)
        if len(s) > len(best):
            best = s
            
    return best

# EVOLVE-BLOCK-END
