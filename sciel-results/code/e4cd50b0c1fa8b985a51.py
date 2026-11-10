def _has_matching(r, L):
    freq = np.zeros(_M, dtype=np.int32)
    for k in range(1, L + 1):
        b_mod = (r + k) % _P
        dm = _DIV_MASK[b_mod]
        freq[dm] += 1
    
    F = freq.copy()
    for i in range(_N):
        step = 1 << (i + 1)
        offset = 1 << i
        for j in range(offset):
            F[offset + j :: step] += F[j :: step]
        
    full = _M - 1
    masks = full ^ np.arange(1, _M)
    vals = F[masks]
    if np.any(L - vals < _BIT_COUNTS_ARR):
        return False
    return True
