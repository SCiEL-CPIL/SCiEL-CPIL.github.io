import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0

    n = 12
    m = 40
    N = 1 << n  # 4096

    # Precompute popcount (not strictly needed but handy)
    popcount = [0] * N
    for mask in range(1, N):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)

    def is_strict_subset(a, b):
        return (a & b) == a and a != b

    def count_comparable(family):
        cnt = 0
        L = len(family)
        for i in range(L):
            ai = family[i]
            for j in range(i + 1, L):
                aj = family[j]
                if (ai & aj) == ai and ai != aj:
                    cnt += 1
                elif (aj & ai) == aj and aj != ai:
                    cnt += 1
        return cnt

    def random_chain():
        perm = list(range(n))
        random.shuffle(perm)
        chain = [0]
        mask = 0
        for v in perm:
            mask |= (1 << v)
            chain.append(mask)
        return chain

    # Build initial family
    family = []
    family_set = set()
    def add_set(mask):
        if mask not in family_set:
            family_set.add(mask)
            family.append(mask)

    add_set(0)
    add_set(N - 1)

    attempts = 0
    while len(family) < m and attempts < 300:
        attempts += 1
        chain = random_chain()
        for mask in chain:
            add_set(mask)
            if len(family) >= m:
                break

    while len(family) < m:
        mask = random.randint(0, N - 1)
        add_set(mask)

    family = family[:m]
    family_set = set(family)

    best_family = family[:]
    best_family_set = set(best_family)
    best_score = count_comparable(best_family)

    def compute_whole(family):
        whole = [0] * N
        for c in range(N):
            cnt = 0
            for fj in family:
                if (c & fj) == c and c != fj:
                    cnt += 1
                elif (fj & c) == fj and fj != c:
                    cnt += 1
            whole[c] = cnt
        return whole

    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        whole = compute_whole(best_family)
        whole_fi = [whole[x] for x in best_family]
        best_gain = 0
        best_i = -1
        best_c = -1
        for i in range(m):
            fi = best_family[i]
            wfi = whole_fi[i]
            for c in range(N):
                if c == fi:
                    continue  # no change
                if c in best_family_set:
                    continue  # would create duplicate
                # comparable between c and fi?
                if (c & fi) == c and c != fi:
                    comp = 1
                elif (fi & c) == fi and fi != c:
                    comp = 1
                else:
                    comp = 0
                rel = whole[c] - comp
                gain = rel - wfi
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_c = c
        if best_gain > 0 and best_i >= 0:
            # update
            old = best_family[best_i]
            best_family[best_i] = best_c
            best_family_set.remove(old)
            best_family_set.add(best_c)
            best_score += best_gain
            improved = True

    # Convert to lists
    result = []
    for mask in best_family:
        lst = [b for b in range(n) if (mask >> b) & 1]
        result.append(lst)

    return result
