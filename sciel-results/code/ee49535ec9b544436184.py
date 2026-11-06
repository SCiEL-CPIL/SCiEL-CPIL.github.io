import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    n = 256

    # Precompute mod masks: mod_mask[x][r] = bitmask of v in [1,n] with v % x == r
    mod_mask = []
    for x in range(1, n + 1):
        masks = [0] * x
        for v in range(1, n + 1):
            masks[v % x] |= (1 << v)
        mod_mask.append(masks)  # index x -> list of length x

    # ge_mask[lo] = bitmask of values in [lo, n]; le_mask[hi] = bitmask of values in [1, hi]
    ge_mask = [0] * (n + 2)
    le_mask = [0] * (n + 2)
    for v in range(1, n + 1):
        bit = 1 << v
        for lo in range(1, v + 1):
            ge_mask[lo] |= bit
        for hi in range(v, n + 1):
            le_mask[hi] |= bit

    def is_valid_add(x, S_mask):
        """Fast check: can we add x to set represented by S_mask?"""
        if x in S_mask:
            return False
        # Case 1: x is 'a'. Need no b,c in S with x < b <= c and x | (b+c).
        # For each b in S with b > x: c must be in S, c >= b, c ≡ -b (mod x).
        b_iter = S_mask & ge_mask[x + 1]
        while b_iter:
            b = (b_iter & -b_iter).bit_length() - 1
            b_iter &= b_iter - 1
            need_c = mod_mask[x][(-b) % x] & S_mask & ge_mask[b]
            if need_c:
                return False
        # Case 2: x is 'b'. Need no a,c in S with a < x <= c and a | (x+c).
        # For each a in S with a < x: c in S, c >= x, c ≡ -x (mod a).
        a_iter = S_mask & le_mask[x - 1]
        while a_iter:
            a = (a_iter & -a_iter).bit_length() - 1
            a_iter &= a_iter - 1
            need_c = mod_mask[a][(-x) % a] & S_mask & ge_mask[x]
            if need_c:
                return False
        # Case 3: x is 'c'. Need no a,b in S with a < b <= x and a | (b+x).
        # For each a in S with a < x: b in S, a < b <= x, b ≡ -x (mod a).
        a_iter = S_mask & le_mask[x - 1]
        while a_iter:
            a = (a_iter & -a_iter).bit_length() - 1
            a_iter &= a_iter - 1
            need_b = mod_mask[a][(-x) % a] & S_mask & ge_mask[a + 1] & le_mask[x]
            if need_b:
                return False
        return True

    def greedy(order, S_mask=0):
        """Greedy add in given order."""
        for x in order:
            if time.time() > deadline:
                break
            if is_valid_add(x, S_mask):
                S_mask |= (1 << x)
        return S_mask

    def mask_to_set(S_mask):
        return [i for i in range(1, n + 1) if (S_mask >> i) & 1]

    def local_add(S_mask):
        """Add any single element that fits, repeat until none."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in range(1, n + 1):
                if time.time() > deadline:
                    return S_mask
                if not ((S_mask >> x) & 1) and is_valid_add(x, S_mask):
                    S_mask |= (1 << x)
                    changed = True
                    break
        return S_mask

    def two_swap(S_mask):
        """Try removing one element and adding one or more to improve."""
        S_list = mask_to_set(S_mask)
        best = S_mask
        best_size = bin(S_mask).count('1')
        for rem in S_list:
            if time.time() > deadline:
                break
            S_new = S_mask & ~(1 << rem)
            # greedily refill
            S_new = local_add(S_new)
            sz = bin(S_new).count('1')
            if sz > best_size:
                best_size = sz
                best = S_new
        return best

    def three_swap(S_mask):
        """Try removing two elements and refilling."""
        S_list = mask_to_set(S_mask)
        best = S_mask
        best_size = bin(S_mask).count('1')
        L = len(S_list)
        for i in range(L):
            if time.time() > deadline:
                break
            for j in range(i + 1, L):
                if time.time() > deadline:
                    break
                S_new = S_mask & ~(1 << S_list[i]) & ~(1 << S_list[j])
                S_new = local_add(S_new)
                sz = bin(S_new).count('1')
                if sz > best_size:
                    best_size = sz
                    best = S_new
        return best

    # ---- Build initial candidates with multiple greedy orderings ----
    best_mask = 0
    best_size = 0

    def consider(S_mask):
        nonlocal best_mask, best_size
        S_mask = local_add(S_mask)
        sz = bin(S_mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = S_mask

    # Descending
    consider(greedy(range(n, 0, -1)))
    # Ascending
    consider(greedy(range(1, n + 1)))

    # Conflict-score ordering: elements that "cause" many conflicts go last
    # Precompute conflict_score[x] = number of (b,c) with x<b<=c and x|(b+c)
    conflict_score = [0] * (n + 1)
    for x in range(1, n + 1):
        cnt = 0
        for b in range(x + 1, n + 1):
            # c in [b, n] with c ≡ -b (mod x)
            r = (-b) % x
            # smallest c >= b with c % x == r
            c0 = b + ((r - b) % x)
            if c0 < b:
                c0 += x
            if c0 <= n:
                cnt += (n - c0) // x + 1
        conflict_score[x] = cnt

    order_by_conflict = sorted(range(1, n + 1), key=lambda x: conflict_score[x])
    consider(greedy(order_by_conflict))

    # Random shuffles
    for seed in range(200):
        if time.time() > deadline - 10:
            break
        random.seed(seed * 7919 + 13)
        order = list(range(1, n + 1))
        random.shuffle(order)
        consider(greedy(order))

    # ---- Local improvement loop ----
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        S_mask = two_swap(best_mask)
        if bin(S_mask).count('1') > best_size:
            best_size = bin(S_mask).count('1')
            best_mask = S_mask
            improved = True
        if time.time() < deadline - 5:
            S_mask = three_swap(best_mask)
            if bin(S_mask).count('1') > best_size:
                best_size = bin(S_mask).count('1')
                best_mask = S_mask
                improved = True

    # Final add
    best_mask = local_add(best_mask)

    result = mask_to_set(best_mask)
    if not result:
        result = [256]
    return result
