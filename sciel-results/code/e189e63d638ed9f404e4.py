<<<<<<< SEARCH
def _score_set(U):
    """Compute the score for a finite set U using FFT for speed."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    maxU = max(U)
    if maxU == 0:
        return -1
    
    # Compute |U+U| using FFT
    sum_len = 2 * maxU + 1
    fft_len = 1
    while fft_len < 2 * sum_len:
        fft_len *= 2
    
    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0
    
    freq = fft(indicator)
    conv = ifft(freq * freq).real
    sumset_size = int(np.sum(conv[:sum_len] > 1e-9))
    
    # Compute |U-U| using autocorrelation
    # diff[k] = |{a in U : a - k in U}| for k in [-maxU, maxU]
    # We want the number of k where diff[k] > 0
    # Using FFT: convolve indicator with reversed indicator
    rev = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        rev[u] = 1.0
    # Actually, for difference set, we can compute:
    # For each pair (a,b) in U, a-b is a difference
    # The difference set size = number of distinct values of a-b
    # We can compute this by checking which shifts have at least one pair
    
    # More efficient: compute the "difference count" via convolution
    # diff_count[d] = number of pairs (a,b) with a-b = d
    # This is the cross-correlation of indicator with itself
    # diff_count[d] = sum_x indicator[x] * indicator[x-d]
    
    # Using FFT: correlate indicator with reversed indicator
    rev_ind = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        rev_ind[fft_len - 1 - u] = 1.0
    
    freq_rev = fft(rev_ind)
    diff_conv = ifft(freq * freq_rev).real
    
    # The difference d = a - b corresponds to position...
    # Let me think: if indicator[i]=1 for i in U, and rev[j]=1 for j such that fft_len-1-j in U
    # Then conv[k] = sum_i indicator[i] * rev[k-i] = sum_i indicator[i] * indicator[fft_len-1-(k-i)]
    # = sum_i indicator[i] * indicator[fft_len-1-k+i]
    # Let j = fft_len-1-k+i, so i = j - (fft_len-1-k)
    # conv[k] = sum_j indicator[j - (fft_len-1-k)] * indicator[j]
    # This is the cross-correlation at lag (fft_len-1-k)
    
    # Actually, let me just use a direct approach for the difference set
    # For reasonable sizes, we can compute it directly
    if n <= 200:
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        diffset_size = len(diffset)
    else:
        # Use the convolution result
        # The differences range from -maxU to maxU
        # In the convolution, position k corresponds to...
        # Let me just count non-zero entries in the relevant range
        # From the convolution: diff_conv[k] = sum_i ind[i] * rev[k-i]
        # rev[j] = 1 iff (fft_len-1-j) in U
        # So diff_conv[k] = sum_i ind[i] * ind[fft_len-1-k+i]
        # Let d = i - (fft_len-1-k+i) = 2i - (fft_len-1-k)... this is getting complex
        
        # Simpler: just compute directly but with numpy
        arr = np.array(U)
        diffs = arr[:, None] - arr[None, :]
        diffset_size = len(np.unique(diffs))
    
    if sumset_size == 0:
        return -1
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return -1
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return -1
    return 1.0 + math.log(ratio) / denom

def _score_set_fast(U):
    """Faster score computation using numpy vectorization."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    maxU = max(U)
    if maxU == 0:
        return -1
    
    arr = np.array(U, dtype=np.int64)
    
    # Sumset: all pairwise sums
    sums = arr[:, None] + arr[None, :]
    sumset_size = len(np.unique(sums))
    
    # Difference set: all pairwise differences  
    diffs = arr[:, None] - arr[None, :]
    diffset_size = len(np.unique(diffs))
    
    if sumset_size == 0:
        return -1
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return -1
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return -1
    return 1.0 + math.log(ratio) / denom
=======
def _score_set(U):
    """Compute the score for a finite set U using FFT for speed."""
    return _score_set_fast(U)

def _score_set_fft(U):
    """Compute score using FFT for both sumset and difference set sizes.
    This is O(M log M) where M is the FFT length, enabling much larger sets."""
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
    maxU = max(U)
    if maxU == 0:
        return -1.0
    
    # Sumset: |U+U|
    # Convolve indicator with itself: result[s] = #{(a,b) in U x U : a+b = s}
    # s ranges from 0 to 2*maxU
    sum_len = 2 * maxU + 1
    fft_len = 1
    while fft_len < 2 * sum_len:
        fft_len *= 2
    
    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0
    
    freq = fft(indicator)
    sum_conv = ifft(freq * freq).real
    sumset_size = int(np.sum(sum_conv[:sum_len] > 0.5))
    
    # Difference set: |U-U|
    # diff[d] = #{(a,b) in U x U : a-b = d}
    # d ranges from -maxU to maxU
    # Use cross-correlation: correlate indicator with reversed indicator
    # rev[k] = 1 iff (fft_len - 1 - k) in U
    # Then cross_conv[m] = sum_i ind[i] * rev[m-i]
    # = sum_i ind[i] * ind[fft_len-1-(m-i)]
    # = sum_i ind[i] * ind[fft_len-1-m+i]
    # Let j = i, then this is sum_j ind[j] * ind[fft_len-1-m+j]
    # The difference d = a - b where a is in U and b is in U
    # If we think of it as: for shift d, count pairs where a = b + d
    # cross_conv[m] counts pairs (i, j) where i + j = fft_len - 1 - m... 
    
    # Simpler approach: 
    # diff_count[d] = sum_x ind[x] * ind[x-d] for d in [-maxU, maxU]
    # This is the autocorrelation at lag d
    # Using FFT: autocorr[d] = ifft(|FFT(ind)|^2)[d] but with proper indexing
    # Actually autocorr via FFT: 
    #   A = FFT(indicator)
    #   autocorr = ifft(A * conj(A))
    #   autocorr[k] = sum_j ind[j] * ind[j-k] (circular)
    # For k in [0, fft_len), autocorr[k] gives sum of ind[j]*ind[j-k mod fft_len]
    # We want linear autocorrelation for d in [-maxU, maxU]
    
    # The linear autocorrelation at lag d (where d >= 0) is:
    #   sum_j ind[j] * ind[j-d] for j from d to fft_len-1
    # This equals ifft(|FFT(ind)|^2)[d] when fft_len is large enough (no wrap-around)
    
    # For negative lag d < 0, autocorr[-d] = sum_j ind[j] * ind[j+d] = autocorr at lag |d|
    # (by symmetry for real signals, autocorr[d] = autocorr[-d])
    
    # So: |U-U| = number of d in [-maxU, maxU] where autocorr[|d|] > 0
    # But we need to be careful: autocorr[0] = n (always > 0)
    # autocorr[d] for d > 0: count of pairs (a,b) with a - b = d, i.e., a = b + d
    
    # Let me verify: if U = {0, 1, 3}, then:
    # Differences: 0-0=0, 0-1=-1, 0-3=-3, 1-0=1, 1-1=0, 1-3=-2, 3-0=3, 3-1=2, 3-3=0
    # Distinct: {-3,-2,-1,0,1,2,3} => size 7
    # autocorr[0] = 3 (pairs: (0,0),(1,1),(3,3))
    # autocorr[1] = sum_j ind[j]*ind[j-1] = ind[1]*ind[0] + ind[3]*ind[2] = 1*1 + 1*0 = 1 (pair: (1,0))
    # autocorr[2] = sum_j ind[j]*ind[j-2] = ind[2]*ind[0] + ind[3]*ind[1] = 0 + 1*1 = 1 (pair: (3,1))
    # autocorr[3] = sum_j ind[j]*ind[j-3] = ind[3]*ind[0] = 1 (pair: (3,0))
    # So |U-U| = 1 (for d=0) + 2*(1+1+1) = 1+6 = 7. Correct!
    
    autocorr = ifft(freq * np.conj(freq)).real
    # autocorr[k] for k in [0, fft_len)
    # For d in [0, maxU]: count = autocorr[d]
    # For d in [-maxU, -1]: count = autocorr[|d|] = autocorr[-d]
    # |U-U| = #{d in [-maxU,maxU] : autocorr[|d|] > 0}
    #        = 1 + 2 * #{d in [1, maxU] : autocorr[d] > 0}
    
    positive_lags = autocorr[1:maxU+1]
    num_positive = int(np.sum(positive_lags > 0.5))
    diffset_size = 1 + 2 * num_positive
    
    if sumset_size == 0:
        return -1.0
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return -1.0
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / denom

def _score_set_fast(U):
    """Fast score computation. Uses FFT for larger sets, direct for small."""
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
    maxU = max(U)
    if maxU == 0:
        return -1.0
    
    if n <= 80:
        # Direct computation is faster for small sets
        arr = np.array(U, dtype=np.int64)
        sums = arr[:, None] + arr[None, :]
        sumset_size = len(np.unique(sums))
        diffs = arr[:, None] - arr[None, :]
        diffset_size = len(np.unique(diffs))
    else:
        # Use FFT
        sum_len = 2 * maxU + 1
        fft_len = 1
        while fft_len < 2 * sum_len:
            fft_len *= 2
        
        indicator = np.zeros(fft_len, dtype=np.float64)
        for u in U:
            indicator[u] = 1.0
        
        freq = fft(indicator)
        sum_conv = ifft(freq * freq).real
        sumset_size = int(np.sum(sum_conv[:sum_len] > 0.5))
        
        autocorr = ifft(freq * np.conj(freq)).real
        positive_lags = autocorr[1:maxU+1]
        num_positive = int(np.sum(positive_lags > 0.5))
        diffset_size = 1 + 2 * num_positive
    
    if sumset_size == 0:
        return -1.0
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return -1.0
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / denom
>>>>>>> REPLACE

Now let me improve the local search to leverage the faster FFT scoring and explore larger sets:

