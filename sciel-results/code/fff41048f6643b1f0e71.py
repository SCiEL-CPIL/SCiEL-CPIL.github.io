import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    # 150s limit, keep 20s margin for safety
    deadline = start_time + 130.0
    
    # 1. Generate M-sequence of length 511 (2^9 - 1)
    # Polynomial x^9 + x^4 + 1 (taps 4 and 0) is a primitive polynomial for m=9
    def gen_m_seq(m, poly_taps):
        state = 1
        seq = []
        for _ in range(2**m - 1):
            seq.append(state & 1)
            fb = 0
            for tap in poly_taps:
                if tap == 0:
                    fb ^= (state & 1)
                else:
                    fb ^= (state >> tap) & 1
            state = ((state >> 1) | (fb << (m - 1))) & ((1 << m) - 1)
        return np.array(seq, dtype=np.float64) * 2 - 1

    # M-sequence of length 511, take first 400
    m_seq_511 = gen_m_seq(9, [4, 0])
    h = m_seq_511[:N].copy()

    # 2. Compute initial autocorrelation
    # R[k] = sum_{i} h[i] h[i+k]
    # Using FFT for initial calculation
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = np.abs(F) ** 2
    R_full = np.fft.irfft(power, n=n_fft)
    # R_full[0..N-1] are the lags 0..N-1
    # Note: R_full[k] for k>0 corresponds to lag k.
    # Due to floating point errors, we might need to clean up, but for now, use as is.
    R = R_full[:N].copy()
    
    # Ensure R[0] is exactly N (it should be, but floating point might make it 399.9999)
    R[0] = float(N)

    def get_metric(R):
        # C3 = max(|R[k]| for k != 0) / R[0]
        # R[0] is fixed at N for binary sequences
        if len(R) <= 1:
            return 0.0
        return np.max(np.abs(R[1:])) / R[0]

    best_h = h.copy()
    best_val = get_metric(R)
    
    k = np.arange(1, N)

    # 3. Greedy Local Search with Incremental Update
    # Flipping h[i] -> -h[i] changes R[k] by:
    # R_new[k] = R[k] - 2*h[i]*(h[i+k] + h[i-k]) (with boundary checks)
    # Specifically:
    # Delta R[k] = -2 * h[i] * (h[i+k] if i+k < N else 0 + h[i-k] if i-k >= 0 else 0)
    
    improved = True
    passes = 0
    max_passes = 500 # Safety cap, but time limit is the real constraint
    
    while improved and passes < max_passes:
        if time.time() > deadline - 10.0:
            break
        improved = False
        passes += 1
        
        # Iterate through indices. 
        # To avoid bias, we can shuffle indices, but sequential is faster.
        # Let's try sequential first. If stuck, we can randomize later.
        for i in range(N):
            if time.time() > deadline - 10.0:
                break
            
            # Calculate delta for flipping h[i]
            # Term 1: h[i+k] for k in 1..N-1
            # Valid if i+k < N
            idx1 = i + k
            valid1 = idx1 < N
            term1 = np.zeros(N-1)
            if np.any(valid1):
                term1[valid1] = h[idx1[valid1]]
            
            # Term 2: h[i-k] for k in 1..N-1
            # Valid if i-k >= 0
            idx2 = i - k
            valid2 = idx2 >= 0
            term2 = np.zeros(N-1)
            if np.any(valid2):
                term2[valid2] = h[idx2[valid2]]
            
            delta = -2.0 * h[i] * (term1 + term2)
            
            new_R = R + delta
            new_val = get_metric(new_R)
            
            if new_val < best_val:
                # Accept move
                h[i] *= -1.0
                R = new_R
                best_val = new_val
                best_h = h.copy()
                improved = True
            # If not improved, keep original h[i] and R

    # 4. Optional: Simulated Annealing or Random Restarts if time permits
    # Given the speed of greedy, we might have time for a second pass or SA.
    # Let's try a few random restarts from the best solution found so far
    # to escape local minima.
    
    if time.time() < deadline - 30.0:
        np.random.seed(42)
        # Try 5 random restarts
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            
            # Perturb best_h
            h_curr = best_h.copy()
            # Flip 10% of bits randomly
            flip_mask = np.random.rand(N) < 0.1
            h_curr[flip_mask] *= -1.0
            
            # Recompute R for this perturbed h
            F = np.fft.rfft(h_curr, n=n_fft)
            power = np.abs(F) ** 2
            R_curr = np.fft.irfft(power, n=n_fft)[:N].copy()
            R_curr[0] = float(N)
            
            curr_val = get_metric(R_curr)
            
            # Run a few passes of greedy on this perturbed version
            improved = True
            passes = 0
            while improved and passes < 50: # Limit passes for restarts
                if time.time() > deadline - 10.0:
                    break
                improved = False
                passes += 1
                for i in range(N):
                    if time.time() > deadline - 10.0:
                        break
                    idx1 = i + k
                    valid1 = idx1 < N
                    term1 = np.zeros(N-1)
                    if np.any(valid1):
                        term1[valid1] = h_curr[idx1[valid1]]
                    
                    idx2 = i - k
                    valid2 = idx2 >= 0
                    term2 = np.zeros(N-1)
                    if np.any(valid2):
                        term2[valid2] = h_curr[idx2[valid2]]
                    
                    delta = -2.0 * h_curr[i] * (term1 + term2)
                    new_R = R_curr + delta
                    new_val = get_metric(new_R)
                    
                    if new_val < curr_val:
                        h_curr[i] *= -1.0
                        R_curr = new_R
                        curr_val = new_val
                        improved = True
            
            if curr_val < best_val:
                best_val = curr_val
                best_h = h_curr.copy()

    return best_h
