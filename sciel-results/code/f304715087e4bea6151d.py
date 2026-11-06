"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill

    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    K = len(primes)

    # Bitmask for each residue modulo 23 (smallest prime)
    mask_res = [0] * 23
    for r in range(23):
        m = 0
        for j, p in enumerate(primes):
            if r % p == 0:
                m |= (1 << j)
        mask_res[r] = m

    def score(n):
        # number of distinct primes dividing at least one of n+1..n+24
        s = 0
        for i in range(1, 25):
            s |= mask_res[(n + i) % 23]
        return s.bit_count()

    # Global scan with sliding window over mask_res residues
    best_n = 1000
    best_s = score(1000)

    # Sliding window: window covers positions (n+1)..(n+24) in the periodic sequence mask_res
    # Use a frequency array of size 512 (max mask value < 2^9)
    freq = [0] * 512
    cur_or = 0

    def add(pos):
        nonlocal cur_or
        m = mask_res[pos % 23]
        freq[m] += 1
        cur_or |= m

    def remove(pos):
        nonlocal cur_or
        m = mask_res[pos % 23]
        freq[m] -= 1
        if freq[m] == 0:
            # recompute OR without this mask
            # Since masks are small, just recompute from freq
            new_or = 0
            for mm in range(512):
                if freq[mm] > 0:
                    new_or |= mm
            cur_or = new_or

    # Initialize window for n=1000: positions (1001)..(1024)
    for pos in range(1001, 1025):
        add(pos)
    s0 = cur_or.bit_count()
    if s0 < best_s:
        best_s = s0
        best_n = 1000

    # Slide n from 1001 to 1000000
    for n in range(1001, 1000001):
        # Remove (n-1+1) = n, add (n+24)
        remove(n)
        add(n + 24)
        s = cur_or.bit_count()
        if s < best_s:
            best_s = s
            best_n = n
        if s == 1:
            break  # can't do better than 1
        if time.time() > deadline:
            break

    # Local refinement: try to improve best_n by scanning nearby n
    # If best_s is small, search a wider neighborhood for even lower
    if best_s > 1:
        # Search around best_n in a wider window
        lo = max(1000, best_n - 2000)
        hi = min(1000000, best_n + 2000)
        for n in range(lo, hi + 1):
            s = score(n)
            if s < best_s:
                best_s = s
                best_n = n
            if s == 1:
                break
        if time.time() > deadline:
            return best_n

    # If we found a good candidate, try to polish: scan a larger range
    if best_s <= 2:
        # Do a broader scan to confirm or improve
        for n in range(1000, 1000001):
            if time.time() > deadline:
                break
            s = score(n)
            if s < best_s:
                best_s = s
                best_n = n
            if s == 1:
                break

    return best_n
# EVOLVE-BLOCK-END
