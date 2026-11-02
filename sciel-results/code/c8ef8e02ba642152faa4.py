"""Rewritten program using Spectral Phase Retrieval and Local Search for signed step functions."""
import numpy as np
from scipy.signal import fftconvolve

N = 400

def score_sequence(x):
    """Calculate the score: max abs autoconvolution / squared integral."""
    # Calculate autoconvolution
    # Use FFT for efficiency
    c = np.real(np.fft.ifft(np.abs(np.fft.fft(x))**2))
    max_abs_c = np.max(np.abs(c))
    
    integral = np.sum(x)
    if integral == 0:
        return float('inf')
    
    squared_integral = integral**2
    return max_abs_c / squared_integral

def construct():
    # Initial candidate: All ones (Seed)
    x_seed = np.ones(N)
    best_score = score_sequence(x_seed)
    best_x = x_seed
    
    # Strategy: Use Gerchberg-Saxton to find sequences with low peak autocorrelation
    # Then quantize to {-1, 1} and refine with local search.
    
    # 1. Define a target magnitude spectrum. 
    # A flat-top window in frequency domain minimizes time-domain ringing? 
    # Actually, we want the time-domain sequence to be a step function.
    # Let's try to find a phase for a flat magnitude that yields a binary-like sequence.
    
    # Initialize random phase
    np.random.seed(42)
    X = np.ones(N)
    X *= np.exp(1j * np.random.uniform(0, 2*np.pi, N))
    
    # Target time-domain constraint: Values should be close to -1 or 1
    # We will iterate:
    # 1. FFT -> modify magnitude to flat-top -> IFFT
    # 2. Quantize to {-1, 1}
    # 3. Check if it's a "step function" (piecewise constant). 
    #    If not, we might penalize or just accept it as a candidate for local search.
    
    candidates = [x_seed]
    
    for i in range(20): # 20 iterations of GS
        # Time domain: Quantize to {-1, 1}
        x = np.sign(np.real(np.fft.ifft(X)))
        x[x == 0] = 1 # Handle zeros
        
        # Calculate FFT of quantized sequence
        X_new = np.fft.fft(x)
        
        # Apply flat-top magnitude constraint
        # Flat-top window in freq domain
        # We want |X| to be flat? 
        # If |X| is flat, x is sinc-like. 
        # Let's just keep the magnitude from the current X? No, that's standard GS.
        # The prompt says "magnitude spectrum is fixed to a flat-top window".
        # Let's use a flat-top window definition.
        W = np.ones(N)
        # Apply window
        X_new = W * np.exp(1j * np.angle(X_new))
        
        # Update
        X = X_new
        
        # Evaluate candidate
        s = score_sequence(x)
        if s < best_score:
            best_score = s
            best_x = x
            
        # Add to candidates for local search later
        candidates.append(x)
        
        # Also try a variant with a different random phase initialization occasionally
        if i % 5 == 0 and i > 0:
             X_rand = np.exp(1j * np.random.uniform(0, 2*np.pi, N))
             X = W * X_rand
             x_rand = np.sign(np.real(np.fft.ifft(X)))
             x_rand[x_rand == 0] = 1
             candidates.append(x_rand)

    # Local Search on the best candidates
    # We will perform local search on the top 5 candidates by score.
    candidates.sort(key=score_sequence)
    top_candidates = candidates[:5]
    
    for x0 in top_candidates:
        x = x0.copy()
        current_score = score_sequence(x)
        
        # Simple hill climbing: flip one bit at a time
        for step in range(50): # Limit steps
            best_neighbor_score = current_score
            best_neighbor_x = x
            best_idx = -1
            
            # Try flipping a random subset of indices to avoid O(N^2) per step
            # Just try flipping each index one by one? N=400 is small enough for 50 steps * 400 flips
            for i in range(N):
                x_temp = x.copy()
                x_temp[i] *= -1
                s = score_sequence(x_temp)
                if s < best_neighbor_score:
                    best_neighbor_score = s
                    best_neighbor_x = x_temp
                    best_idx = i
            
            if best_idx == -1:
                break # Local minimum found
            
            x = best_neighbor_x
            current_score = best_neighbor_score
            
            if current_score < best_score:
                best_score = current_score
                best_x = x

    return best_x

# EVOLVE-BLOCK-START
# The code above is the implementation.
# We wrap it to match the required interface.
# The prompt asks to keep function names.
# construct() is the entry point.
# EVOLVE-BLOCK-END
