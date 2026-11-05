import time

def construct():
    start = time.time()
    deadline = 55.0  # safety margin before the 60s/90s kill
    P = 101
    DENOMS = list(range(1, 11))
    inv = [0] * 11
    for x in DENOMS:
        inv[x] = pow(x, -1, P)

    # best_size[r] = minimum subset size achieving residue r
    # best_terms[r] = corresponding subset
    best_size = [10**9] * P
    best_terms = [None] * P

    # Precompute bit positions for speed
    n = len(DENOMS)
    for mask in range(1, 1 << n):
        if time.time() - start > deadline:
            break
        terms = []
        s = 0
        for i in range(n):
            if (mask >> i) & 1:
                x = DENOMS[i]
                terms.append(x)
                s += inv[x]
        s %= P
        sz = len(terms)
        if sz < best_size[s]:
            best_size[s] = sz
            best_terms[s] = terms

    result = []
    for r in range(P):
        if best_terms[r] is not None:
            result.append(best_terms[r])
        else:
            result.append([1])
    return result
