"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _autocorr_max_abs(x):
    """Compute max absolute value of linear autocorrelation using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    x_padded = np.zeros(size, dtype=np.float64)
    x_padded[:n] = x
    
    X = np.fft.fft(x_padded, n=size)
    C = np.fft.ifft(X * np.conj(X)).real
    
    return np.max(np.abs(C))

def _psd_peak(x):
    """Compute the peak of the Power Spectral Density (PSD).
    Minimizing the peak of the PSD is equivalent to minimizing the max absolute autocorrelation
    for binary sequences, because autocorrelation is the IFFT of PSD.
    """
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    x_padded = np.zeros(size, dtype=np.float64)
    x_padded[:n] = x
    
    X = np.fft.fft(x_padded, n=size)
    PSD = np.abs(X)**2
    
    # The peak of the autocorrelation corresponds to the max of the IFFT of the PSD.
    # However, a good proxy for "flat" autocorrelation is a flat PSD.
    # Let's return the max PSD value. A flatter PSD implies lower peaks in time domain (autocorrelation).
    return np.max(PSD)

def _phase_objective(phase, n):
    """
    Objective for L-BFGS: Minimize the peak of the PSD.
    Input: phase (array of length n/2 + 1 for real signal symmetry, or full length if we handle conjugate symmetry)
    Actually, for real signals, we can optimize the full complex spectrum but constrain it to be conjugate symmetric.
    Easier: Optimize the phase of the first half, construct the full spectrum, take IFFT, clip to binary? No, that breaks the gradient.
    
    Better approach for spectral flatness:
    Let X[k] be the DFT. We want to minimize max |IFFT(|X|^2)|.
    This is non-smooth.
    
    Alternative: Minimize the variance of the autocorrelation (excluding lag 0).
    Sum_{k != 0} |C[k]|^2.
    By Parseval's theorem, Sum_{k != 0} |C[k]|^2 = Sum_{k != 0} |X[k]|^4 / N.
    Wait, Parseval: Sum |c[k]|^2 = (1/N) Sum |C[k]|^2 where C is DFT of c.
    If c is autocorrelation, C is PSD.
    So minimizing the energy of the off-peak autocorrelation is equivalent to minimizing the energy of the PSD (excluding DC?).
    
    Let's try minimizing the max PSD value directly using a smooth approximation or just the max.
    L-BFGS can handle max if we use a log-sum-exp trick or just the max (subgradient).
    
    Let's use the "spectral amplitude" approach from inspiration 2 but adapted.
    We fix the amplitude of the DFT to be constant (ideal flat spectrum) and optimize the phase.
    For a binary sequence, the amplitude is not flat, but we can start from a flat amplitude and project.
    
    Strategy:
    1. Start with a random phase for a flat amplitude spectrum.
    2. IFFT to get a real signal.
    3. Threshold to binary.
    4. Compute actual autocorrelation peak.
    5. Use Simulated Annealing to refine.
    """
    pass

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    np.random.seed(42)
    
    best_val = float('inf')
    best_x = np.random.choice([-1, 1], N).astype(np.float64)
    
    # Phase 1: Spectral Flatness Initialization
    # We want a sequence whose PSD is as flat as possible.
    # A random binary sequence has a roughly flat PSD but with noise.
    # We can try to optimize the phase of the DFT to flatten the PSD.
    
    # Let's try a few random starts and a simple "spectral" polish.
    # For each start, we will do a few iterations of:
    # 1. Compute DFT
    # 2. Keep amplitude, randomize phase? No.
    # 3. Actually, let's just use Simulated Annealing with a good initial guess.
    
    # Better Initial Guess:
    # Use a Barker-like construction or just random.
    # Let's try 10 random starts.
    
    num_starts = 10
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 60.0:
            break
            
        x = np.random.choice([-1, 1], N).astype(np.float64)
        val = _autocorr_max_abs(x)
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
        # Simulated Annealing Polish
        # We want to minimize the max absolute autocorrelation.
        # This is a combinatorial optimization problem.
        # SA is effective for such "noisy" landscapes.
        
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        
        x_current = x.copy()
        val_current = val
        x_best_sa = x_current.copy()
        val_best_sa = val_current
        
        steps = 2000
        step_count = 0
        
        while step_count < steps and T > T_min and time.time() < deadline - 10.0:
            # Pick a random index to flip
            i = np.random.randint(N)
            
            # Calculate change in autocorrelation?
            # Efficient update:
            # C[k] = sum x[j] x[j+k]
            # If we flip x[i], the change in C[k] is:
            # -2 * x[i] * sum_{j: j or j+k == i} x[j] ... this is complex.
            # For N=400, FFT is fast enough.
            # 2000 steps * FFT(1024) is very fast.
            
            x_trial = x_current.copy()
            x_trial[i] = -x_trial[i]
            
            val_trial = _autocorr_max_abs(x_trial)
            
            delta = val_trial - val_current
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                x_current = x_trial
                val_current = val_trial
                
                if val_current < val_best_sa:
                    val_best_sa = val_current
                    x_best_sa = x_current.copy()
            else:
                # Reject
                pass
            
            T *= cooling
            step_count += 1
            
        if val_best_sa < best_val:
            best_val = val_best_sa
            best_x = x_best_sa.copy()
            
    # Phase 2: Local Search on the best solution found
    # Try flipping single bits and pairs of bits if time permits.
    
    x = best_x.copy()
    val = best_val
    
    # Single bit flips
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            x[i] = -x[i]
            val_new = _autocorr_max_abs(x)
            if val_new < val:
                val = val_new
                improved = True
            else:
                x[i] = -x[i] # Revert
    
    if val < best_val:
        best_val = val
        best_x = x.copy()
        
    # Convert to int8 for return
    return best_x.astype(np.int8)

# EVOLVE-BLOCK-END
