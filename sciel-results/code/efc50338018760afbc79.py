import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 120s. We aim to finish by 110s.
    deadline = start_time + 110.0

    def thue_morse(n):
        seq = [1]
        while len(seq) < n:
            seq += [-x for x in seq]
        return np.array(seq[:n], dtype=np.int8)

    def get_score(sig):
        # Compute aperiodic autocorrelation using FFT
        # Zero-pad to 2*N to avoid circular convolution artifacts
        f = np.fft.fft(sig, n=2*N)
        ac = np.fft.ifft(f * np.conj(f)).real
        # The resulting ac has length 2*N.
        # Indices 0..N-1 are lags 0..N-1.
        # Indices N..2N-1 are lags -(N-1)..-1.
        # We want the max absolute value over all aperiodic lags.
        return np.max(np.abs(ac))

    best_sigma = thue_morse(N)
    best_score = get_score(best_sigma)
    
    sigma = best_sigma.copy()
    
    # Main search loop
    while time.time() < deadline:
        # Local Search Phase
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_idx = -1
            current_score = get_score(sigma)
            
            # Evaluate all flips
            # With FFT, this is fast enough to do a full pass
            for i in range(N):
                sigma[i] = -sigma[i]
                new_score = get_score(sigma)
                delta = current_score - new_score
                if delta > best_delta:
                    best_delta = delta
                    best_idx = i
                sigma[i] = -sigma[i] # revert
                
            if best_idx != -1:
                sigma[best_idx] = -sigma[best_idx]
                new_score = get_score(sigma)
                if new_score < best_score:
                    best_score = new_score
                    best_sigma = sigma.copy()
                improved = True
            else:
                # Stuck in local minimum
                break
        
        # If we are stuck and have time, do a random restart
        if time.time() < deadline - 5.0:
            # Generate a new random start
            sigma = np.random.choice([-1, 1], size=N).astype(np.int8)
            current_score = get_score(sigma)
            
            # Quick local search on this new start
            # Limit iterations to avoid getting stuck in a bad basin for too long
            max_local_steps = 20
            for _ in range(max_local_steps):
                if time.time() > deadline - 2.0:
                    break
                improved = False
                best_delta = 0
                best_idx = -1
                c_score = get_score(sigma)
                
                for i in range(N):
                    sigma[i] = -sigma[i]
                    n_score = get_score(sigma)
                    delta = c_score - n_score
                    if delta > best_delta:
                        best_delta = delta
                        best_idx = i
                    sigma[i] = -sigma[i]
                    
                if best_idx != -1:
                    sigma[best_idx] = -sigma[best_idx]
                    improved = True
                else:
                    break
                    
            current_score = get_score(sigma)
            if current_score < best_score:
                best_score = current_score
                best_sigma = sigma.copy()

    return best_sigma
