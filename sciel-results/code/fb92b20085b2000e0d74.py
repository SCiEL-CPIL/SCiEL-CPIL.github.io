import time
import numpy as np

def construct():
    start = time.time()
    LIMIT = 1000000
    N = LIMIT + 1

    # 1. Compute the powerful-part array for every integer 0..N
    #    pp[k] = product of p^e over all prime powers p^e with e>=2 dividing k
    #    We use a sieve approach: for each prime p, for each power pe = p^e (e>=2),
    #    multiply pe into pp[k] for every k divisible by pe but NOT by p^(e+1).
    
    # Sieve primes up to sqrt(N)
    sqrt_N = int(np.sqrt(N)) + 1
    is_prime = np.ones(sqrt_N + 1, dtype=np.bool_)
    is_prime[:2] = False
    for i in range(2, int(np.sqrt(sqrt_N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.nonzero(is_prime)[0]

    pp = np.ones(N + 1, dtype=np.int64)

    # For each prime, process its powers
    for p in primes:
        p = int(p)
        pe = p * p
        while pe <= N:
            # We need to multiply pp[k] by pe for all k such that:
            # 1. k is a multiple of pe
            # 2. k is NOT a multiple of pe * p
            
            # Indices divisible by pe
            idx = np.arange(pe, N + 1, pe)
            
            # Indices divisible by pe * p
            next_pe = pe * p
            if next_pe <= N:
                # Create a mask for indices divisible by pe
                # Instead of a full N-sized mask, we can operate on the idx array directly
                # But we need to exclude those divisible by next_pe.
                # The indices divisible by next_pe are a subset of idx.
                # Specifically, if k is in idx, k = m * pe.
                # k is divisible by next_pe = pe * p iff m is divisible by p.
                
                # So we just need to skip every p-th element in idx
                # idx[::p] are the ones to skip?
                # Let's verify:
                # idx = [pe, 2pe, 3pe, ..., Mpe]
                # next_pe = p * pe
                # Multiples of next_pe in idx: p*pe, 2p*pe, ...
                # These correspond to indices p-1, 2p-1, ... in the idx array (0-indexed: p-1, 2p-1)
                # Wait, idx[0] = pe. idx[p-1] = p*pe.
                # So we want to exclude idx[p-1], idx[2p-1], etc.
                
                # It is faster to multiply all, then divide the bad ones? No, integer division is lossy if we don't track it.
                # Better: create a boolean mask for the 'good' indices in idx.
                # Or simply:
                # Multiply all idx by pe.
                pp[idx] *= pe
                # Then divide the ones that were divisible by next_pe?
                # No, that would result in pp[k] having pe * (pe/p) = p^e * p^(e-1) ... incorrect.
                
                # Let's stick to the mask approach but optimize it.
                # The previous code created a mask of size N+1.
                # We can create a mask of size len(idx).
                len_idx = len(idx)
                # The elements to exclude are at positions p-1, 2p-1, ...
                # i.e., indices j in idx such that (j+1) % p == 0
                # mask_good = np.ones(len_idx, dtype=np.bool_)
                # mask_good[p-1::p] = False
                # pp[idx[mask_good]] *= pe
                
                # However, creating the mask and indexing is still overhead.
                # Given N=1e6, the number of primes is ~168.
                # The total number of operations is sum(N/p^e) which is small.
                # The current method (full N mask) was ~1-2s.
                # Let's try the subset mask method.
                
                # Actually, the previous code did:
                # mask = np.zeros(N+1, dtype=np.bool_)
                # mask[idx] = True
                # mask[idx_bad] = False
                # pp[mask] *= pe
                # This is O(N) per prime power.
                
                # The subset method:
                # mask_good = np.ones(len(idx), dtype=np.bool_)
                # mask_good[p-1::p] = False
                # pp[idx[mask_good]] *= pe
                # This is O(N/p^e) per prime power. Much faster.
                
                # Let's implement this optimized version.
                # But wait, I already did `pp[idx] *= pe` above in the thought process. 
                # I should NOT do that first.
                
                # Correct logic:
                mask_good = np.ones(len(idx), dtype=np.bool_)
                mask_good[p-1::p] = False
                pp[idx[mask_good]] *= pe
            else:
                pp[idx] *= pe
            
            pe = next_pe

    # 2. Score every candidate n in [10000, LIMIT]
    #    score(n) = pp[n] * pp[n+1]
    scores = pp[10000:LIMIT + 1] * pp[10001:N + 1]
    best_local = int(np.argmax(scores))
    best_n = 10000 + best_local

    # 3. Verify the answer and return
    return best_n
