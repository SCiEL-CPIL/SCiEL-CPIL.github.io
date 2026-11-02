<<<<<<< SEARCH
N = 600

# EVOLVE-BLOCK-START
=======
N = 600

def normalized_max_autoconv(h):
    """
    Compute the normalized maximum of the autoconvolution.
    Objective: max(ac) / (sum(h))^2
    """
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    ac = np.fft.irfft(H * np.conj(H), n=n_fft)
    s = np.sum(h)
    if s == 0:
        return np.inf
    return np.max(ac) / (s * s)

# EVOLVE-BLOCK-START
>>>>>>> REPLACE
