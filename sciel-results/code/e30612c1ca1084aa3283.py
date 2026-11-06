def _compute_multiplicities_fast(A):
    """Compute representation counts for all s in [128,256] using NumPy convolution."""
    if not A:
        return {}
    
    arr = np.zeros(257, dtype=np.int32)
    for a in A:
        if 0 <= a <= 128:
            arr[a] = 1
            
    conv = np.convolve(arr, arr, mode='full')
    
    mult = {}
    for s in range(128, 257):
        c = conv[s]
        e = 0
        if s % 2 == 0:
            half = s // 2
            if 0 <= half <= 128 and arr[half] == 1:
                e = 1
        n = (c + e) // 2
        mult[s] = n
        
    return mult
