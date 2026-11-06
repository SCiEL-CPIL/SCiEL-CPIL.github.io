import numpy as np

def construct():
    M = 5 * 10**8
    
    # Generate squarefree numbers up to M^(1/3)
    max_b = int(M ** (1/3)) + 2
    
    is_sqfree = np.ones(max_b + 1, dtype=bool)
    is_sqfree[0] = False
    for p in range(2, int(max_b**0.5) + 1):
        p2 = p * p
        for j in range(p2, max_b + 1, p2):
            is_sqfree[j] = False
    
    sqfree = [b for b in range(1, max_b + 1) if is_sqfree[b]]
    
    # Generate powerful numbers: n = b^3 * c^2 <= M, b squarefree
    powerful = set()
    for b in sqfree:
        b3 = b ** 3
        if b3 > M:
            break
        max_c = int((M // b3) ** 0.5) + 1
        for c in range(1, max_c + 1):
            n = b3 * c * c
            if n <= M:
                powerful.add(n)
            else:
                break
    
    powerful_list = sorted(powerful)
    K = len(powerful_list)
    P = np.array(powerful_list, dtype=np.int32)
    
    if K < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    max_N = 2 * M
    counts = np.zeros(max_N + 1, dtype=np.int32)
    
    # Chunked upper-triangular sums to avoid full K x K matrix
    chunk_size = 500
    for start in range(0, K, chunk_size):
        end = min(start + chunk_size, K)
        chunk = P[start:end]
        cl = end - start
        sums = chunk[:, None] + P[None, :]
        lr = np.arange(cl)
        gr = start + lr
        cols = np.arange(K)
        mask = cols[None, :] >= gr[:, None]
        ms = sums[mask]
        counts += np.bincount(ms, minlength=max_N + 1).astype(np.int32)
    
    counts[:2] = 0
    max_idx = np.argmax(counts)
    N = int(max_idx)
    
    P_set = set(powerful_list)
    pairs = []
    for a in powerful_list:
        b = N - a
        if b >= a and b in P_set:
            pairs.append([a, b])
    
    return {'N': N, 'pairs': pairs}
