"""Rewritten program using discrete local search on signed sequences."""
import numpy as np
import time

N = 400

def _compute_c3prime(x):
    """Compute max|autoconv| / integral^2 for vector x."""
    n = len(x)
    # Linear autoconvolution via FFT
    f = np.fft.rfft(x, n=2*n)
    ac = np.fft.irfft(f * np.conj(f), n=2*n)[:n]
    max_ac = np.max(np.abs(ac))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e10
    return max_ac / (integral * integral)

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n mapped to +/-1."""
    t = np.zeros(n, dtype=int)
    t[0] = 0
    for i in range(1, n):
        # t[i] = parity of number of 1s in binary representation of i
        # Or recursive: t[i] = t[i//2] if i even, else 1 - t[i//2]
        # Efficiently: t[i] = i ^ (i >> 1) ... no, that's not it.
        # t[i] = sum of bits of i mod 2
        # We can compute iteratively
        pass
    
    # Efficient generation
    t = np.zeros(n, dtype=np.int8)
    t[0] = 0
    for i in range(1, n):
        # t[i] = t[i >> 1] if i is even, else 1 - t[i >> 1]
        # Actually, t[i] = t[i // 2] XOR (i % 2)
        t[i] = t[i // 2] ^ (i % 2)
        
    # Map 0 -> 1, 1 -> -1
    return 1 - 2 * t

def construct():
    t0 = time.time()
    deadline = 145.0 # Safety margin
    
    best_x = None
    best_score = 1e10
    
    # Candidate 1: Thue-Morse
    candidates = []
    candidates.append(_thue_morse(N))
    
    # Candidate 2: Random +/-1
    rng = np.random.default_rng(42)
    for _ in range(5):
        candidates.append(rng.choice([-1.0, 1.0], size=N))
        
    # Candidate 3: Low frequency sinusoid mapped to +/-1
    x_sin = np.sin(2 * np.pi * np.arange(N) / N)
    candidates.append(np.sign(x_sin) * 1.0) # Where sign is 0, it stays 0? No, sin doesn't hit 0 exactly usually.
    # Ensure no zeros
    candidates[-1] = np.where(candidates[-1] == 0, 1.0, candidates[-1])

    # Local Optimization Function
    def local_optimize(x_init):
        x = x_init.astype(np.float64)
        n = len(x)
        
        # Precompute autoconvolution
        f = np.fft.rfft(x, n=2*n)
        A = np.fft.irfft(f * np.conj(f), n=2*n)[:n]
        S = np.sum(x)
        
        current_score = _compute_c3prime(x)
        
        # If S is close to 0, the score is huge. We want to move away from S=0.
        # The local search flips signs.
        
        improved = True
        while improved and (time.time() - t0 < deadline - 5.0):
            improved = False
            # Random order to avoid bias
            indices = np.random.permutation(n)
            
            for i in indices:
                if time.time() - t0 > deadline - 5.0:
                    return x, current_score
                
                old_val = x[i]
                new_val = -old_val
                
                # Update autoconvolution A
                # A[k] = sum_j x[j] x[j+k]
                # Flipping x[i] changes terms where j=i or j+k=i.
                # A_new[k] = A_old[k] - 2 * old_val * (x[i+k] + x[i-k])
                # Note: boundaries.
                
                # We can update A in place for speed, but we need to check score first.
                # Let's compute the new A efficiently?
                # Actually, for N=400, recomputing A via FFT is ~10us.
                # Looping 400 times * 100 iterations = 40000 FFTs = 0.4s.
                # This is fast enough.
                
                x[i] = new_val
                
                f_new = np.fft.rfft(x, n=2*n)
                A_new = np.fft.irfft(f_new * np.conj(f_new), n=2*n)[:n]
                S_new = S - 2.0 * old_val
                
                if abs(S_new) < 1e-12:
                    x[i] = old_val
                    continue
                    
                max_ac_new = np.max(np.abs(A_new))
                score_new = max_ac_new / (S_new * S_new)
                
                if score_new < current_score:
                    # Accept
                    A = A_new
                    S = S_new
                    current_score = score_new
                    improved = True
                else:
                    # Reject
                    x[i] = old_val
        
        return x, current_score

    # Run local optimization on each candidate
    for cand in candidates:
        if time.time() - t0 > deadline - 5.0:
            break
            
        x_opt, score_opt = local_optimize(cand)
        
        if score_opt < best_score:
            best_score = score_opt
            best_x = x_opt
    
    # Fallback if something went wrong
    if best_x is None:
        best_x = np.ones(N)
        
    return best_x
