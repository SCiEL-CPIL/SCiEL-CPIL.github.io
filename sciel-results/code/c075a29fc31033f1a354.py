<<<<<<< SEARCH
def _local_search_flip(h, max_passes=10, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses FFT incrementally for speedup.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    
    # Precompute frequency basis vectors for speed
    freqs = np.arange(_FSIZE//2+1)
    basis_cache = {}
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            # Flip sign: h[i] -> -h[i]
            old_val = h[i]
            h[i] = -old_val
            # Update FFT incrementally
            if i not in basis_cache:
                basis_cache[i] = np.exp(-2j * np.pi * freqs * i / _FSIZE)
            delta = -2 * old_val * basis_cache[i]
            H_new = H + delta
            conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
            integral = np.sum(h)
            if abs(integral) < 1e-12:
                new_val = float('inf')
            else:
                new_val = np.max(np.abs(conv)) / (integral * integral)
            
            if new_val < best_val:
                best_val = new_val
                H = H_new
                improved = True
            else:
                h[i] = old_val  # revert
        if not improved:
            break
    return h, best_val
=======
def _local_search_flip(h, max_passes=10, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses FFT incrementally for speedup with precomputed basis.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    s = np.sum(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            # Flip sign: h[i] -> -h[i]
            old_val = h[i]
            h[i] = -old_val
            s -= 2 * old_val
            # Update FFT incrementally using precomputed basis
            delta = -2 * old_val * _BASIS[:, i]
            H_new = H + delta
            conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
            if abs(s) < 1e-12:
                new_val = float('inf')
            else:
                new_val = np.max(np.abs(conv)) / (s * s)
            
            if new_val < best_val:
                best_val = new_val
                H = H_new
                improved = True
            else:
                h[i] = old_val  # revert
                s += 2 * old_val
        if not improved:
            break
    return h, best_val
>>>>>>> REPLACE

