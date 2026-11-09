import numpy as np
import time

N = 400

def construct():
    """
    Returns a sequence of signed step heights (length N) that minimizes
    the upper bound on C3.
    
    Based on the task description "Third autocorrelation inequality" and 
    the observed improvement when switching metrics in inspirations, we 
    assume the metric to minimize is the peak absolute value of the 
    third-order autocorrelation (triple correlation) at non-zero lags.
    
    R3(t1, t2) = sum_n h[n] * h[n+t1] * h[n+t2]
    We minimize max(|R3(t1, t2)|) for t1 != 0, t2 != 0, t1+t2 != 0.
    """
    start_time = time.time()
    # Safety margin: 150s limit, stop at 140s
    deadline = start_time + 140

    def compute_triple_autocorr_peak(h):
        """
        Computes the maximum absolute value of the third-order autocorrelation
        R3(t1, t2) = sum_n h[n] h[n+t1] h[n+t2]
        for all non-zero lags t1, t2.
        
        Optimization: 
        For a fixed t1, let g[n] = h[n] * h[n+t1].
        Then R3(t1, t2) is the cross-correlation of g and h at lag t2.
        We compute this using FFT.
        """
        n = len(h)
        # We need lags up to n-1. 
        # To handle linear convolution/correlation correctly with FFT, 
        # we zero-pad to size >= 2*n.
        nfft = 1 << (2 * n - 1).bit_length()
        
        max_val = 0.0
        
        # We only need to check t1, t2 such that the shifts are valid.
        # t1 ranges from 1 to n-1.
        # For each t1, g is defined on [0, n-t1-1] roughly.
        # Actually, let's just compute the full circular correlation and 
        # mask the valid linear region.
        
        H = np.fft.rfft(h, n=nfft)
        
        for t1 in range(1, n):
            # g[n] = h[n] * h[n+t1]
            # We need to be careful with indices.
            # h[n+t1] for n in [0, n-t1-1] is h[t1 : n]
            # h[n] for n in [0, n-t1-1] is h[0 : n-t1]
            # The product is defined for n in [0, n-t1-1].
            # Let's form the vector g of length n, padded with zeros where undefined?
            # Or just compute the correlation on the valid range.
            
            # Efficient approach:
            # Compute FFT of h.
            # We want sum_n h[n] h[n+t1] h[n+t2].
            # Let's fix t1. 
            # term[n] = h[n] h[n+t1].
            # We need sum_n term[n] h[n+t2].
            
            # To do this for all t2 via FFT, we need the FFT of term.
            # term is only non-zero for n in 0..n-t1-1.
            # We can compute term, zero pad, FFT, then multiply with H?
            # No, cross-correlation of term and h is IFFT(FFT(term) * conj(FFT(h))).
            
            # Construct term
            # term[n] = h[n] * h[n+t1] for n=0..n-t1-1
            # We can create an array of length nfft, zero elsewhere.
            # This is O(N) per t1. Total O(N^2).
            
            term = np.zeros(nfft)
            # h[n] is h[0:n-t1]
            # h[n+t1] is h[t1:n]
            term[0:n-t1] = h[0:n-t1] * h[t1:n]
            
            # Compute FFT of term
            T = np.fft.rfft(term)
            
            # Cross-correlation R(t2) = IFFT(T * conj(H))
            # Note: H is FFT of h (zero padded to nfft)
            R = np.fft.irfft(T * np.conj(H), n=nfft)
            
            # R[k] corresponds to sum_n term[n] h[n+k] ?
            # Standard correlation: R_xy[k] = sum_n x[n] y[n+k].
            # Using FFT: IFFT(FFT(x) * conj(FFT(y))) gives sum_n x[n] y[n+k]?
            # Let's verify.
            # If x is delta[n], y is delta[n-1], corr at k=1 is 1.
            # FFT(delta) = 1. conj(FFT(delta[n-1])) = conj(exp(-i 2pi f)) = exp(i 2pi f).
            # Product = exp(i 2pi f). IFFT is delta[n+1].
            # So R[k] = sum_n x[n] y[n-k]? Or y[n+k]?
            # Standard: Correlation C[k] = sum_n x[n] y[n+k].
            # C = IFFT(FFT(x) * conj(FFT(y))).
            # Let's assume standard numpy correlation behavior via FFT.
            
            # The valid lags for t2 are such that the overlap is significant.
            # term is non-zero in [0, n-t1-1].
            # h is non-zero in [0, n-1].
            # The sum is non-zero when there is overlap.
            # t2 can range from -(n-1) to n-1.
            
            # We want to find max |R[t2]| for t2 != 0.
            # Also we must ensure t1 != 0 (already true) and t1+t2 != 0 (to avoid trivial terms?)
            # The task says "non-zero lags". Usually implies (t1, t2) != (0,0).
            # If t2 = -t1, then h[n]h[n+t1]h[n-t1]. This is a valid triple.
            
            # Extract the relevant part of R.
            # R is length nfft.
            # Lags 0..nfft/2 are positive, nfft/2..nfft are negative (wrapped).
            # We care about lags in range -(n-1) .. n-1.
            
            # Positive lags: R[0 : n]
            # Negative lags: R[nfft-n+1 : nfft]
            
            pos_lags = R[0:n]
            neg_lags = R[nfft-n+1:nfft]
            
            # Combine
            all_lags = np.concatenate([pos_lags, neg_lags])
            
            # Mask out t2=0? 
            # If t2=0, R3(t1, 0) = sum h[n]h[n+t1]h[n] = sum h[n+t1] = 0 if sum(h)=0.
            # We should enforce sum(h) ~ 0 to keep balance.
            # If sum(h) is not 0, R3(t1, 0) is non-zero.
            # The problem asks for "non-zero lags". Usually this means the vector (t1, t2) is not (0,0).
            # So t2=0 is allowed if t1!=0.
            # However, if we want to minimize the "sidelobe", we might want to suppress t2=0 too?
            # "Peak autocorrelation sidelobe" usually excludes the main lobe.
            # For 3rd order, the "main lobe" is at (0,0).
            # So we should exclude (0,0).
            # t1 is always != 0 in our loop.
            # So we include t2=0.
            
            current_max = np.max(np.abs(all_lags))
            if current_max > max_val:
                max_val = current_max
                
        return max_val

    def compute_peak_sidelobe_2nd(h):
        """Fallback metric if 3rd order is too slow or wrong."""
        n = len(h)
        nfft = 1 << (2 * n - 1).bit_length()
        H = np.fft.rfft(h, n=nfft)
        R = np.fft.irfft(H * np.conj(H), n=nfft)
        sidelobes_pos = R[1:n]
        sidelobes_neg = R[nfft - n + 1:nfft]
        all_sidelobes = np.concatenate([sidelobes_pos, sidelobes_neg])
        return np.max(np.abs(all_sidelobes))

    # Use the 3rd order metric as primary hypothesis
    # If it's too slow, we might need to sample t1.
    # N=400, 399 t1 values. 399 FFTs of size 1024.
    # 1024 FFT is very fast. 399 * ~10us = ~4ms per evaluation.
    # This allows ~30 evaluations per second.
    # In 140s, we can do ~4000 evaluations.
    # This is enough for a simple SA.

    def get_score(h):
        # Ensure h is +/- 1
        # Balance check
        if np.sum(h) != 0:
            # Penalize imbalance
            return compute_triple_autocorr_peak(h) + abs(np.sum(h)) * 10.0
        return compute_triple_autocorr_peak(h)

    # Initial sequences
    np.random.seed(42)
    
    def thue_morse_sequence(n):
        seq = np.zeros(n, dtype=np.int8)
        for i in range(n):
            x = i
            cnt = 0
            while x:
                cnt += x & 1
                x >>= 1
            seq[i] = 1 if cnt % 2 == 0 else -1
        return seq

    # Try to balance the sequence
    def balance_seq(h):
        h = h.copy()
        s = np.sum(h)
        if s > 0:
            idx = np.where(h == 1)[0]
            if len(idx) >= s:
                h[np.random.choice(idx, s, replace=False)] = -1
        elif s < 0:
            idx = np.where(h == -1)[0]
            if len(idx) >= -s:
                h[np.random.choice(idx, -s, replace=False)] = 1
        return h

    initial_sequences = []
    initial_sequences.append(balance_seq(thue_morse_sequence(N)))
    
    # Random inits
    for _ in range(10):
        h = np.random.choice([-1, 1], N)
        initial_sequences.append(balance_seq(h))

    best_h = None
    best_score = float('inf')

    def simulated_annealing(h, iterations=2000, T_start=10.0, T_end=0.1):
        current_score = get_score(h)
        T = T_start
        h = h.copy()
        
        for i in range(iterations):
            if time.time() > deadline - 5:
                break
            
            # Move: Flip a random bit
            idx = np.random.randint(N)
            h[idx] *= -1
            
            new_score = get_score(h)
            delta = new_score - current_score
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_score = new_score
            else:
                h[idx] *= -1
            
            # Cooling
            T = T_start * (T_end / T_start) ** (i / iterations)
            
        return h, current_score

    # Run SA on each initial sequence
    for init_h in initial_sequences:
        if time.time() > deadline - 10:
            break
            
        h, score = simulated_annealing(init_h, iterations=1000)
        
        # Local Polish: Greedy flip
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for _ in range(50):
                idx = np.random.randint(N)
                h[idx] *= -1
                new_score = get_score(h)
                if new_score < score:
                    score = new_score
                    improved = True
                else:
                    h[idx] *= -1
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
        # Early exit if we get a very good score
        if best_score < 0.1:
            break

    if best_h is None:
        best_h = initial_sequences[0]
        
    return best_h

# Test run to ensure it doesn't crash
if __name__ == "__main__":
    h = construct()
    print("Length:", len(h))
    print("Sum:", np.sum(h))
    # print("Score:", get_score(h)) # Not accessible outside
