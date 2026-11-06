import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    p = 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, p)

    # best_len[r] = minimum number of denominators whose inverses sum to r mod p
    # best_terms[r] = one such set of denominators
    best_len = [11] * 101
    best_terms = [None] * 101

    # Enumerate all non-empty subsets of {1,...,10}
    for mask in range(1, 1 << 10):
        s = 0
        count = 0
        terms = []
        for i in range(10):
            if mask & (1 << i):
                x = i + 1
                s += inv[x]
                count += 1
                terms.append(x)

        r = s % p
        if count < best_len[r]:
            best_len[r] = count
            best_terms[r] = terms

        if time.time() > deadline:
            break

    result = []
    for r in range(101):
        if best_terms[r] is not None:
            result.append(best_terms[r])
        else:
            result.append([1])

    return result
