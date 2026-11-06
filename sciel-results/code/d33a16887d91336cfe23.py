import numpy as np
from math import isqrt

def construct():
    # Generate powerful numbers up to L
    L = 10**7
    powerful_set = set()
    # u^2 * v^3 <= L
    # v is square-free? No, v in u^2 v^3 is just an integer, but for unique representation 
    # we usually take v square-free. However, for generation, we can just iterate 
    # squares and cubes.
    # A number is powerful if it is of the form u^2 * v^3 where v is square-free.
    # But simpler: just generate all u^2 * v^3.
    # To avoid duplicates, use a set.
    
    max_u = isqrt(L)
    for u in range(1, max_u + 1):
        u2 = u * u
        max_v = int((L // u2) ** (1/3)) + 1
        for v in range(1, max_v + 1):
            val = u2 * (v ** 3)
            if val <= L:
                powerful_set.add(val)
            else:
                break
    
    powerful_list = sorted(powerful_set)
    P = np.array(powerful_list, dtype=np.int64)
    
    # Compute all sums
    # To save memory, we can do it in chunks if P is large, 
    # but for L=10^7, len(P) is approx 3000-4000.
    # 4000x4000 = 16M elements. 128MB. OK.
    
    sums = P[:, None] + P[None, :]
    
    # We only care about a <= b, but bincount on all is fine, 
    # we just need to filter pairs later.
    # Actually, if we use all sums, we double count (a,b) and (b,a).
    # The problem asks for distinct pairs a <= b.
    # So we should only consider a <= b.
    # We can create a mask or just use the full matrix and divide counts by 2 (approx)?
    # No, diagonal is counted once.
    # Better: iterate or use triu indices.
    
    # Efficient way:
    # counts = np.bincount(sums.ravel())
    # This counts (a,b) and (b,a) separately.
    # For a != b, count is doubled.
    # For a == b, count is 1.
    # We want number of pairs (a,b) with a <= b.
    # Let C(N) be the raw count from bincount.
    # Let D(N) be the number of pairs with a=b (i.e., N=2a).
    # Then pairs_count = (C(N) - D(N)) / 2 + D(N) = (C(N) + D(N)) / 2.
    
    raw_counts = np.bincount(sums.ravel())
    
    # Find D(N): number of a in P such that 2a = N
    # This is 1 if N/2 is in P and N is even, else 0.
    # We can compute this for the candidate best Ns.
    
    # First, find top candidates from raw_counts
    # Since (C+D)/2 is monotonic with C (mostly), we can just look at top C.
    
    # Get indices of top raw counts
    # We need to be careful: a number with C=100 and D=0 has 50 pairs.
    # A number with C=101 and D=1 has 51 pairs.
    # So we should evaluate the score for the top few candidates.
    
    # Get top 100 candidates by raw count
    top_indices = np.argsort(raw_counts)[-100:][::-1]
    
    best_score = 0
    best_N = -1
    best_pairs = []
    
    P_set = set(powerful_list)
    
    for idx in top_indices:
        N = int(idx)
        if N < 2:
            continue
        # Calculate exact pair count for a <= b
        # Iterate through P to find pairs
        # Since P is small (~4000), this is fast.
        pairs = []
        for a in powerful_list:
            if 2 * a > N:
                break
            b = N - a
            if b in P_set:
                pairs.append([a, b])
        
        if len(pairs) > best_score:
            best_score = len(pairs)
            best_N = N
            best_pairs = pairs
            
    if best_N == -1:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': best_pairs}
