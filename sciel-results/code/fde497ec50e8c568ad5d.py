"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import numpy as np

    N = 1_000_001  # need powerful for n and n+1 where n in [10000, 1_000_000]

    # Linear sieve to compute:
    # spf[i]: smallest prime factor of i
    # exp[i]: exponent of spf[i] in i
    # powerful[i]: product of p^e over primes p with exact exponent e>=2 in i
    #
    # Recurrence for i >= 2:
    # Let p = spf[i].
    # If i % (p*p) == 0:
    #   exp[i] = exp[i//p] + 1
    #   powerful[i] = powerful[i//p] * p
    # Else:
    #   exp[i] = 1
    #   powerful[i] = powerful[i//p]  (since p appears with exponent 1, not counted)
    #
    # This is O(N) and avoids per-prime inner loops.

    spf = np.zeros(N + 1, dtype=np.int64)
    exp = np.zeros(N + 1, dtype=np.int8)   # exponents are small
    powerful = np.ones(N + 1, dtype=np.int64)
    primes = []

    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            exp[i] = 1
            # powerful[i] stays 1
        for p in primes:
            x = i * p
            if x > N:
                break
            spf[x] = p
            if p == spf[i]:
                # p divides i, so exponent of p in x is exponent in i plus 1
                exp[i] += 1  # wait, we can't modify exp[i] here because i is reused
                # Actually, we need exp[x] = exp[i] + 1
                # But we must not alter exp[i]. Let's restructure.
                break
            else:
                # p < spf[i], so p is the smallest prime factor of x
                exp[x] = 1
                # powerful[x] = powerful[i] (since p appears with exponent 1)
                powerful[x] = powerful[i]
                break

    # The above loop has a bug: we can't compute exp[x] correctly in the linear sieve
    # because we need exp[i] to be already computed, and we do compute it, but the
    # logic for p == spf[i] case needs exp[x] = exp[i] + 1 and powerful[x] = powerful[i] * p.
    # Let's rewrite the sieve properly.

    spf = np.zeros(N + 1, dtype=np.int64)
    exp = np.zeros(N + 1, dtype=np.int8)
    powerful = np.ones(N + 1, dtype=np.int64)
    primes = []

    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
            exp[i] = 1
            # powerful[i] = 1
        for p in primes:
            x = i * p
            if x > N:
                break
            spf[x] = p
            if p == spf[i]:
                # p is the smallest prime factor of both i and x
                exp[x] = exp[i] + 1
                powerful[x] = powerful[i] * p
                break
            else:
                # p < spf[i], so p is the smallest prime factor of x, appearing with exponent 1
                exp[x] = 1
                powerful[x] = powerful[i]
                break

    # Now find n in [10000, 1000000] that maximizes powerful[n] * powerful[n+1]
    best_n = 10000
    best_val = powerful[10000] * powerful[10001]

    for n in range(10000, 1_000_000):
        val = powerful[n] * powerful[n + 1]
        if val > best_val:
            best_val = val
            best_n = n

    return int(best_n)
# EVOLVE-BLOCK-END
