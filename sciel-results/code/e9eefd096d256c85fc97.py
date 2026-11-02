"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random

N = 100

def construct():
    # Theoretical analysis:
    # For circular shifts, sum_{k=0}^{N-1} O_k = N^2/4.
    # Average O_k = N/4.
    # Max O_k >= N/4.
    # The constant sequence h_i = 0.5 achieves O_k = N/4 for all k.
    # Thus, 0.5 is the global optimum.
    
    h = np.full(N, 0.5)
    
    # Efficient overlap calculation using FFT
    def get_max_overlap(h):
        # h is real.
        # Autocorrelation A[k] = sum h_i h_{i+k}
        # O[k] = N/2 - A[k]
        # We want min max O[k] = min max (N/2 - A[k]) = N/2 - min A[k]
        # So we want to maximize min A[k].
        
        H = np.fft.fft(h)
        P = np.abs(H)**2
        A = np.fft.ifft(P).real
        # A[k] corresponds to shift k.
        # Due to floating point, A might have tiny errors.
        # We look for min A.
        min_A = np.min(A)
        max_ov = N / 2.0 - min_A
        return max_ov

    current_max_ov = get_max_overlap(h)
    
    # Local search on discrete set {0, 0.5, 1}
    # We start at 0.5. Any move to 0/1 will increase variance and likely increase max_ov.
    # We perform a few iterations to ensure we don't miss a better discrete solution (though theory says none exists).
    
    for _ in range(50):
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i == j:
            continue
        
        old_i, old_j = h[i], h[j]
        
        # Only move if we are at 0.5,0.5 to 0,1 or 1,0
        if old_i == 0.5 and old_j == 0.5:
            h[i] = 0.0
            h[j] = 1.0
            new_max_ov = get_max_overlap(h)
            if new_max_ov < current_max_ov:
                current_max_ov = new_max_ov
            else:
                h[i] = old_i
                h[j] = old_j
        # If we are at 0,1 we can revert to 0.5,0.5
        elif (old_i == 0.0 and old_j == 1.0) or (old_i == 1.0 and old_j == 0.0):
            h[i] = 0.5
            h[j] = 0.5
            new_max_ov = get_max_overlap(h)
            if new_max_ov < current_max_ov:
                current_max_ov = new_max_ov
            else:
                h[i] = old_i
                h[j] = old_j

    return h

# EVOLVE-BLOCK-END
