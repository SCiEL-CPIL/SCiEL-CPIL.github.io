import time
import random
from math import gcd

_DEADLINE = 60.0
_START = None
_CAND = list(range(1, 129))
_PRE = None
_BALL_MASK = None  # BALL_MASK[x][g] is a bitmask of y in [1,128] with gcd(x,y)==g
_PRIMEP = None
_GCD_PAIRS = None # GCD_PAIRS[g] is a list of pairs (a,b) with gcd(a,b)==g

def _ensure_pre():
    global _PRE, _BALL_MASK, _PRIMEP, _GCD_PAIRS
    if _PRE is not None:
        return
    n = 128
    PRE = [[1] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            PRE[i][j] = PRE[j][i] = gcd(i, j)
    _PRE = PRE

    # Precompute prime power flags
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

    # Precompute GCD pairs for each g
    # g can range from 1 to 128
    GCD_PAIRS = {g: [] for g in range(1, n + 1)}
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            g = PRE[i][j]
            GCD_PAIRS[g].append((i, j))
    _GCD_PAIRS = GCD_PAIRS

    # Precompute Ball Masks
    # BALL_MASK[x][g] is an integer bitmask
    BALL_MASK = {}
    for x in range(1, n + 1):
        d = {}
        for y in range(1, n + 1):
            g = PRE[x][y]
            if g not in d:
                d[g] = 0
            d[g] |= (1 << y)
        BALL_MASK[x] = d
    _BALL_MASK = BALL_MASK


def _bad_triples_in(S):
    """Count equal-gcd triples within S."""
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


def _can_add_mask(S_mask, x):
    """
    Check if x can be added to the set represented by S_mask.
    S_mask is an integer bitmask where bit y is 1 if y is in S.
    """
    B = _BALL_MASK[x]
    P = _PRE
    GCD_PAIRS = _GCD_PAIRS
    
    for g, ball_mask in B.items():
        # Elements of S that have gcd(x, y) == g
        S_in_ball = S_mask & ball_mask
        
        # If fewer than 2 elements, no pair can be formed
        # Quick check: count bits
        if S_in_ball == 0 or (S_in_ball & (S_in_ball - 1)) == 0:
            continue
        
        # Check if any pair in S_in_ball has gcd == g
        # We can iterate through pairs in GCD_PAIRS[g] and check if both are in S_in_ball
        # Or iterate through bits in S_in_ball.
        # Given N=128, iterating through bits is fast.
        
        # Let's try checking against precomputed pairs for this g
        # This might be faster if the number of pairs is small or S_in_ball is dense.
        # Actually, iterating through bits of S_in_ball is O(k^2) where k is popcount.
        # Iterating through GCD_PAIRS[g] is O(P_g).
        # Let's stick to bit iteration as it's more predictable for small k.
        
        bits = S_in_ball
        # Extract elements
        elems = []
        while bits:
            lsb = bits & -bits
            idx = lsb.bit_length() - 1
            elems.append(idx)
            bits ^= lsb
            
        m = len(elems)
        if m < 2:
            continue
            
        # Check pairs
        for i in range(m):
            a = elems[i]
            for j in range(i + 1, m):
                b = elems[j]
                if P[a][b] == g:
                    return False
    return True


def _greedy(order, S_mask=0):
    S = []
    for x in order:
        if _can_add_mask(S_mask, x):
            S.append(x)
            S_mask |= (1 << x)
    return S, S_mask


def _local_add(S, S_mask):
    """Repeatedly add any excluded element that fits until none remain."""
    improved = True
    while improved:
        improved = False
        for x in _CAND:
            if not (S_mask & (1 << x)):
                if _can_add_mask(S_mask, x):
                    S.append(x)
                    S_mask |= (1 << x)
                    improved = True
    return S, S_mask


def _local_search(S, S_mask):
    """
    Local search:
    1. Add any possible elements.
    2. Try 1-swaps that enable adds.
    3. Try 2-swaps (limited) that enable adds.
    """
    P = _PRE
    max_rounds = 20
    for _ in range(max_rounds):
        if time.time() - _START > _DEADLINE - 5:
            break
        
        # 1. Adds
        added = True
        while added:
            added = False
            for x in _CAND:
                if not (S_mask & (1 << x)):
                    if _can_add_mask(S_mask, x):
                        S.append(x)
                        S_mask |= (1 << x)
                        added = True
        
        if time.time() - _START > _DEADLINE - 5:
            break
        
        # 2. 1-Swaps
        swap_found = False
        S_list = list(S)
        for r in S_list:
            if time.time() - _START > _DEADLINE - 5:
                break
            r_mask = 1 << r
            S_minus_r_mask = S_mask & ~r_mask
            S_minus_r = [x for x in S if x != r]
            
            for x in _CAND:
                if x == r or (S_mask & (1 << x)):
                    continue
                x_mask = 1 << x
                # Check if S - {r} + {x} is valid
                # This is equivalent to checking if x can be added to S - {r}
                if _can_add_mask(S_minus_r_mask, x):
                    # Tentative swap
                    new_mask = S_minus_r_mask | x_mask
                    # Check if this enables any add
                    can_add_new = False
                    for y in _CAND:
                        if not (new_mask & (1 << y)):
                            if _can_add_mask(new_mask, y):
                                can_add_new = True
                                break
                    
                    if can_add_new:
                        # Accept swap
                        S = S_minus_r + [x]
                        S_mask = new_mask
                        swap_found = True
                        break
            if swap_found:
                break
        
        if not swap_found:
            # Try 2-swaps if time permits? 
            # 2-swaps are O(N^4) which is 128^4 ~ 268M, too slow.
            # We'll skip 2-swaps in the inner loop to save time, relying on restarts.
            break
            
    return S, S_mask


def _construct():
    global _START
    _ensure_pre()
    _START = time.time()
    PRIMEP = _PRIMEP

    prime_powers = [x for x in range(1, 129) if PRIMEP[x]]
    others = [x for x in range(1, 129) if not PRIMEP[x]]
    
    best = []
    best_len = -1
    best_mask = 0

    # Fixed orderings
    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        sorted(range(1, 129), key=lambda x: (0 if PRIMEP[x] else 1, x)),
        prime_powers + others,
        others + prime_powers,
    ]
    
    for order in orders:
        if time.time() - _START > _DEADLINE - 10:
            break
        S, S_mask = _greedy(order)
        S, S_mask = _local_add(S, S_mask)
        if len(S) > best_len:
            best = S
            best_len = len(S)
            best_mask = S_mask

    # Local search on best
    if time.time() - _START < _DEADLINE - 10:
        S, S_mask = _local_search(best, best_mask)
        if len(S) > best_len:
            best = S
            best_len = len(S)
            best_mask = S_mask

    # Random restarts
    rng = random.Random(12345)
    while time.time() - _START < _DEADLINE - 10:
        # Biased random order: Prime powers first, then others
        pp = list(prime_powers)
        rest = list(others)
        rng.shuffle(pp)
        rng.shuffle(rest)
        order = pp + rest
        
        S, S_mask = _greedy(order)
        S, S_mask = _local_add(S, S_mask)
        if len(S) > best_len:
            best = S
            best_len = len(S)
            best_mask = S_mask
            
        # Local search on this random result
        if time.time() - _START < _DEADLINE - 12:
            S, S_mask = _local_search(S, S_mask)
            if len(S) > best_len:
                best = S
                best_len = len(S)
                best_mask = S_mask

    # Final validation
    if _bad_triples_in(best) != 0:
        # Fallback
        best, _ = _greedy(list(range(1, 129)))
        best, _ = _local_add(best, 0)
        
    if not best:
        best = [2]
        
    return best


def construct():
    try:
        return _construct()
    except Exception:
        _ensure_pre()
        # Simple fallback
        S = []
        for x in range(1, 129):
            valid = True
            for a in S:
                for b in S:
                    if a == b: continue
                    if gcd(x, a) == gcd(x, b) == gcd(a, b):
                        valid = False
                        break
                if not valid: break
            if valid:
                S.append(x)
        return S if S else [2]
