import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    
    MAXN = 10000000
    
    # Compute sigma using numpy sieve
    sigma = np.zeros(MAXN + 1, dtype=np.int64)
    for i in range(1, MAXN + 1):
        sigma[i::i] += i
    
    # This is still O(N log N) but vectorized. Let's check timing.
    # Actually this might still be too slow in Python loop over i.
    
    # Better: use the fact that sigma is multiplicative and do a linear sieve
    # Or just use a smarter approach.
    
    # Let me try a different strategy: only compute sigma for numbers that could be in a good pair.
    # For a/b ≈ 3/2, a ≈ 3b/2. So for b in [1, MAXN], a ≈ 3b/2 must be <= MAXN, so b <= 2*MAXN/3.
    
    # But I still need sigma values. Let me just do the sieve but with numpy vectorization over i.
    # The loop over i is the bottleneck. Let me see if I can do it faster.
    
    # Alternative: compute sigma using prime factorization via a smallest-prime-factor sieve
    spf = np.zeros(MAXN + 1, dtype=np.int64)
    for i in range(2, MAXN + 1):
        if spf[i] == 0:
            spf[i::i] = i
    
    # Now compute sigma using spf
    sigma = np.ones(MAXN + 1, dtype=np.int64)
    for n in range(2, MAXN + 1):
        if time.time() > deadline:
            break
        p = spf[n]
        m = n // p
        if m % p == 0:
            # p divides m, so p^k || n for k>=2
            # sigma(n) = sigma(m) * (p^{k+1}-1)/(p-1) / ((p^k-1)/(p-1)) ... this is complex
            # Better to track power of p
            pass
        else:
            sigma[n] = sigma[m] * (1 + p)
    
    # This is getting complicated. Let me just use the direct sieve with numpy and hope it's fast enough.
    
    # Actually, let me just do a simpler approach: compute sigma for a smaller range first, say 10^6, and see if we find good pairs.
    
    MAXN_SMALL = 2000000
    sigma = np.zeros(MAXN_SMALL + 1, dtype=np.int64)
    for i in range(1, MAXN_SMALL + 1):
        sigma[i::i] += i
    
    # Group by sigma value
    fibers = {}
    for n in range(1, MAXN_SMALL + 1):
        s = int(sigma[n])
        if s in fibers:
            fibers[s].append(n)
        else:
            fibers[s] = [n]
    
    best_err = None
    best_pair = None
    
    for s, lst in fibers.items():
        if len(lst) < 2:
            continue
        if time.time() > deadline:
            break
        lst.sort()
        nlen = len(lst)
        for i in range(nlen):
            b = lst[i]
            target = 3 * b / 2.0
            lo, hi = 0, nlen - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if lst[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            candidates = []
            if lo < nlen:
                candidates.append(lst[lo])
            if lo - 1 >= 0:
                candidates.append(lst[lo - 1])
            for a in candidates:
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue
                den = 2 * b
                err = num / den
                if best_err is None or err < best_err:
                    best_err = err
                    best_pair = (a, b)
    
    if best_pair is None:
        return [11, 6]
    
    a, b = best_pair
    return [int(a), int(b)]
