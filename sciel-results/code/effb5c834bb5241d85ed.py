"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Spectral amplitude optimization with phase projection (Gerchberg-Saxton).
    We optimize the power spectrum P = |H|^2 to minimize the maximum overlap
    D_k = N/2 - C_k, while using GS to find phases that keep h in [0,1].
    """
    np.random.seed(42)
    
    # Power spectrum P_j = |H_j|^2
    # H_0 is fixed by sum(h) = N/2, so P_0 = (N/2)^2
    # Initialize P with a flat spectrum for j > 0
    P = np.ones(N)
    P[0] = (N/2)**2
    
    best_h = np.full(N, 0.5)
    best_score = np.inf
    
    # Number of outer iterations (amplitude updates)
    n_outer = 200
    # Number of inner GS iterations (phase updates)
    n_inner = 100
    
    for outer_it in range(n_outer):
        # 1. Find phases for current P such that h is in [0,1]
        # Initialize phases randomly or from previous iteration?
        # Random restarts might help escape local minima, but GS is fast.
        # Let's keep phases from the previous iteration if available, else random.
        if outer_it == 0:
            phases = np.random.uniform(0, 2 * np.pi, N)
        # else: phases are updated in the loop below
        
        h = np.full(N, 0.5)
        for inner_it in range(n_inner):
            # Construct H with current P magnitudes and current phases
            H = np.sqrt(P) * np.exp(1j * phases)
            
            # Inverse FFT to get h
            h = np.fft.ifft(H).real
            
            # Project h to [0, 1]
            h = np.clip(h, 0, 1)
            
            # Compute new FFT
            H_new = np.fft.fft(h)
            
            # Update phases to match H_new (keep magnitudes from P)
            # Handle zero magnitudes to avoid angle(0) issues
            magnitudes = np.abs(H_new)
            phases = np.angle(H_new)
            # If magnitude is very small, phase is unstable, keep old phase
            small_mask = magnitudes < 1e-8
            phases[small_mask] = phases[small_mask] # Keep old
            
        # 2. Evaluate the current solution
        H_final = np.fft.fft(h)
        autocorr = np.fft.ifft(np.abs(H_final)**2).real
        # autocorr[k] = sum_i h_i h_{i+k}
        
        D_vals = N/2 - autocorr
        score = np.max(D_vals)
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
        
        # 3. Update P to reduce the worst lag
        k_worst = np.argmax(D_vals)
        
        # Gradient of C_k w.r.t P_j
        # C_k = (1/N) sum_j P_j exp(2 pi i j k / N)
        # dC_k / dP_j = (1/N) cos(2 pi j k / N)
        # We want to increase C_k_worst to decrease D_k_worst.
        # So we add a positive multiple of the gradient.
        
        # Step size decay
        eta = 0.5 / (1 + outer_it)
        
        grad = np.cos(2 * np.pi * np.arange(N) * k_worst / N)
        
        # Update P
        P_new = P + eta * grad
        # Ensure P is non-negative
        P_new = np.maximum(P_new, 0)
        
        # Keep P[0] fixed
        P_new[0] = (N/2)**2
        
        # Optional: Normalize P to prevent explosion
        # The sum of P is sum h^2. We don't want this to drift too much.
        # But letting it evolve is part of the optimization.
        
        P = P_new
        
    # Final correction to ensure sum is exactly N/2
    sum_h = np.sum(best_h)
    if abs(sum_h - N/2) > 1e-10:
        best_h = best_h - (sum_h - N/2) / N
        best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
