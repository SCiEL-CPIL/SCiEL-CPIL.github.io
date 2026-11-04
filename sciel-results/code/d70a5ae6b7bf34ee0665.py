<<<<<<< SEARCH
def _compute_score(U):
    """Compute the score 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U or max(U) == 0:
        return 0.0
    U = list(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    # Compute sumset and difference set
    # Use numpy for efficiency
    U_arr = np.array(U, dtype=np.int64)
    # Sumset: all pairs (i,j) with i,j in U
    # Difference set: all pairs (i,j) with i,j in U
    # For large sets, this can be expensive. Use the fact that |U+U| <= |U|^2
    # and |U-U| <= |U|^2
    
    # Efficient computation using convolution-like approach
    # Actually, for sets up to ~10000 elements, direct computation might be too slow
    # Let's use a smarter approach: represent as boolean array and use FFT
    # But that's complex. Let's try direct for small sets, and for large sets use approximation
    
    if n <= 2000:
        # Direct computation
        sums = set()
        diffs = set()
        for i in U:
            for j in U:
                sums.add(i + j)
                diffs.add(i - j)
        num_sums = len(sums)
        num_diffs = len(diffs)
    else:
        # Use FFT-based approach
        # Create boolean array
        max_val = max_u * 2
        arr = np.zeros(max_val + 1, dtype=np.float64)
        for x in U:
            arr[x] = 1.0
        # Convolution for sumset
        fft_size = 1
        while fft_size < 2 * (max_val + 1):
            fft_size *= 2
        fft_arr = np.fft.fft(arr, fft_size)
        conv = np.fft.ifft(fft_arr * fft_arr).real
        num_sums = int(np.sum(conv > 0.5))
        
        # For difference set, use cross-correlation
        # U-U = {a-b : a,b in U}
        # This is equivalent to the support of the autocorrelation
        # Actually, U-U can be computed similarly
        # Let's use the fact that |U-U| can be found via:
        # For each d, d in U-U iff there exists a in U such that a-d in U
        # This is the cross-correlation of U with itself
        # Using FFT: autocorrelation
        fft_arr2 = np.fft.fft(arr, fft_size)
        autocorr = np.fft.ifft(fft_arr2 * np.conj(fft_arr2)).real
        # The autocorrelation at lag k gives sum of arr[i]*arr[i+k]
        # U-U contains d iff autocorrelation at appropriate lag is > 0
        # Actually, let's just compute it directly for differences
        # For large sets, this might still be slow. Let's use a different approach.
        # 
        # Alternative: |U-U| is related to the number of distinct differences.
        # We can estimate it, but for scoring we need exact values.
        # Let's just do direct computation but with early termination if it's taking too long
        start = time.time()
        diffs = set()
        for i in U:
            for j in U:
                diffs.add(i - j)
                if time.time() - start > 10:
                    break
            if time.time() - start > 10:
                break
        num_diffs = len(diffs)
    
    if num_sums == 0:
        return 0.0
    ratio = num_diffs / num_sums
    if ratio <= 1.0:
        return 1.0
    log_term = math.log(ratio) / math.log(2 * max_u + 1)
    return 1.0 + log_term
=======
def _compute_score(U):
    """Compute the score 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U or max(U) == 0:
        return 0.0
    U = list(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    
    # Use FFT-based approach for efficiency O(N log N)
    # Sumset U+U corresponds to convolution of indicator function of U with itself.
    # Difference set U-U corresponds to autocorrelation (convolution with reversed indicator).
    
    # We need an array large enough to hold the convolution.
    # Max sum is 2*max_u. Max diff is max_u - 0 = max_u, min diff is 0 - max_u = -max_u.
    # So we need to handle range [-max_u, 2*max_u].
    # Shift the set by max_u so all elements are non-negative and fit in [0, 2*max_u + max_u] = [0, 3*max_u].
    # Actually, simpler:
    # Let f(x) be 1 if x in U, 0 otherwise.
    # Support of f*f is U+U.
    # Support of f*reverse(f) is U-U (shifted).
    
    # To avoid circular convolution artifacts, we need FFT size >= 2 * (2*max_u) = 4*max_u for sums.
    # For differences, the range is [-max_u, max_u], so length 2*max_u + 1.
    # Let's use a single FFT size that covers the worst case.
    # Max value in sum: 2*max_u.
    # Max value in diff (if we shift): max_u - (-max_u) is not right.
    # Let's just compute both using a sufficiently large FFT.
    
    # Create indicator array
    # We need to represent U in an array of size L.
    # If we use array of size L, convolution has size 2L-1.
    # We need 2L-1 > 2*max_u => L > max_u + 0.5.
    # Let's pick L = next_power_of_2(2 * max_u + 1) to be safe for sums.
    
    L = 1
    while L < 2 * max_u + 1:
        L *= 2
        
    arr = np.zeros(L, dtype=np.float64)
    for x in U:
        arr[x] = 1.0
        
    fft_arr = np.fft.fft(arr, L)
    
    # Sumset: f * f
    conv_sum = np.fft.ifft(fft_arr * fft_arr).real
    # The support of conv_sum is U+U.
    # conv_sum[k] > 0 iff k in U+U.
    num_sums = int(np.sum(conv_sum > 0.5))
    
    # Difference set: U-U = {a-b}.
    # This is equivalent to the support of the cross-correlation.
    # Correlation c[d] = sum f[i] f[i+d].
    # This is conv of f and reversed f.
    # Let's create reversed arr.
    # arr_rev[i] = arr[L-1-i] ? No, we need to align indices properly.
    # Standard trick: 
    # Let g = reverse(arr). 
    # conv(f, g) gives correlations.
    # Specifically, if we define g[i] = f[-i] (in cyclic sense), then conv(f,g)[k] = sum f[j]g[k-j] = sum f[j]f[j-k].
    # This is related to differences.
    
    # Easier: Use the fact that |U-U| = |U+(-U)|.
    # Let V = -U. Compute U+V.
    # But we can do it with one FFT if we are careful.
    # Let's just compute it directly using the same FFT array but with a reversed version.
    
    # Reversing the array in numpy:
    # arr_rev = arr[::-1]
    # But we need to handle the circular shift correctly for the FFT convolution.
    # Actually, it's safer to just do a second FFT for the difference set to avoid subtle bugs with shifts,
    # or use the property that the number of distinct differences is the number of non-zero terms in the autocorrelation.
    
    # Autocorrelation R[k] = sum_i f[i] f[i+k].
    # This is the inverse FFT of |FFT(f)|^2.
    # The support of R corresponds to the differences U-U.
    # Specifically, R[k] > 0 iff there exist a,b in U such that a-b = k (mod L, but since L > 2*max_u, no wrap-around for valid diffs).
    
    auto_corr = np.fft.ifft(fft_arr * np.conj(fft_arr)).real
    # The autocorrelation is centered at 0.
    # In the circular convolution output, lag 0 is at index 0.
    # Positive lags k are at index k. Negative lags -k are at index L-k.
    # We need to count all indices where auto_corr > 0.5.
    num_diffs = int(np.sum(auto_corr > 0.5))
    
    if num_sums == 0:
        return 0.0
    ratio = num_diffs / num_sums
    if ratio <= 1.0:
        return 1.0
    log_term = math.log(ratio) / math.log(2 * max_u + 1)
    return 1.0 + log_term
>>>>>>> REPLACE

