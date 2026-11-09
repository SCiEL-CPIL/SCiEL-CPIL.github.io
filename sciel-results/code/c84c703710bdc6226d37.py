"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s kill

    # Sieve primes up to max value in gaps
    max_val = max(hi for _, hi in P['gaps'])
    is_prime = bytearray(b'\x01') * (max_val + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for p in range(2, int(max_val**0.5) + 1):
        if is_prime[p]:
            for multiple in range(p * p, max_val + 1, p):
                is_prime[multiple] = 0

    # Build least-prime-factor array using linear sieve approach
    lpf = [0] * (max_val + 1)
    lpf[1] = 1
    primes = []
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            lpf[i] = i
            primes.append(i)
        for p in primes:
            if p > lpf[i] or i * p > max_val:
                break
            lpf[i * p] = p

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        for m in range(lo + 1, hi):
            if lpf[m] >= gap:
                result[i] = m
                break

    # Ensure all entries are Python int or None
    final = []
    for entry in result:
        if entry is None:
            final.append(None)
        else:
            final.append(int(entry))

    return final

# EVOLVE-BLOCK-END
