"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin for 120s limit
    
    # Initial random binary sequence
    # Ensure sum is not too small initially
    f = np.random.choice([-1, 1], size=N)
    
    # Helper to compute score
    def get_score(f):
        S = np.sum(f)
        if abs(S) < 1e-6:
            return float('inf')
        # Autoconvolution
        # Linear convolution length 2N-1
        # C_k for k in -(N-1) to N-1
        # We only need max abs for k != 0
        # Using FFT for speed? N=400, direct is O(N^2) = 160k ops. Fast enough.
        # But we do many iterations. FFT is O(N log N).
        # Let's use FFT.
        f_pad = np.zeros(2*N)
        f_pad[:N] = f
        F = np.fft.fft(f_pad, n=2*N)
        P = F * np.conj(F)
        C = np.fft.ifft(P).real
        # C[0] is C_0. C[1:N] is C_1...C_{N-1}. C[N+1:] is negative lags.
        # Max abs sidelobe
        # Exclude C[0]
        # Sidelobes are C[1:N] and C[N+1:2*N] (which corresponds to negative lags)
        # Actually C[N-k] is C_{-k}.
        # So we take max of C[1:N] and C[N+1:]
        side1 = np.abs(C[1:N])
        side2 = np.abs(C[N+1:2*N])
        max_side = max(np.max(side1), np.max(side2))
        
        return max_side / (S * S)

    current_score = get_score(f)
    best_f = f.copy()
    best_score = current_score
    
    # Simulated Annealing
    T = 0.1  # Initial temperature
    cooling = 0.99
    
    while time.time() < deadline:
        # Pick random index to flip
        i = random.randint(0, N-1)
        
        # Flip
        f[i] *= -1
        
        new_score = get_score(f)
        
        # Acceptance
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = f.copy()
        else:
            # Probability of accepting worse solution
            # Delta = new_score - current_score
            # If T is small, reject
            if T > 1e-6:
                delta = new_score - current_score
                if delta < 0:
                    accept = True
                else:
                    # Avoid overflow
                    if delta / T > 50:
                        accept = False
                    else:
                        accept = random.random() < np.exp(-delta / T)
                if not accept:
                    # Revert
                    f[i] *= -1
        
        T *= cooling
        
    return best_f
# EVOLVE-BLOCK-END
