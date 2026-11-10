import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 60  # safety margin for 90s kill switch

    N = 1000000
    best_n = 10000

    # Sieve for powerful parts
    pp = [1] * (N + 2)

    limit = int(math.isqrt(N)) + 1
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))

    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    for p in primes:
        p2 = p * p
        if p2 > N + 1:
            break

        # For each power p^e (e>=2), multiply pp[m] by p^e for m divisible by p^e but not p^(e+1)
        pe = p2
        while pe <= N + 1:
            pe_next = pe * p
            if pe_next > N + 1:
                # All multiples of pe within [1, N+1] have exactly e factors of p
                for m in range(pe, N + 2, pe):
                    pp[m] *= pe
                if time.time() > deadline:
                    return best_n
            else:
                # Multiply pp[m] by pe for m = k*pe where k is not a multiple of p
                max_k = (N + 1) // pe
                # Iterate k from 1 to max_k, skipping every p-th
                k = 1
                while k <= max_k:
                    # k, k+1, ..., k+p-1 are not multiples of p
                    end = min(k + p - 1, max_k)
                    for kk in range(k, end + 1):
                        pp[kk * pe] *= pe
                    k += p
                if time.time() > deadline:
                    return best_n

            pe = pe_next

    # Find the max product
    max_prod = 0
    for n in range(10000, N + 1):
        prod = pp[n] * pp[n + 1]
        if prod > max_prod:
            max_prod = prod
            best_n = n

    return best_n
