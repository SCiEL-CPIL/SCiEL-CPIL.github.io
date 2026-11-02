"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal.windows import flattop

def construct():
    """
    Construct signed step heights minimizing the maximum absolute autoconvolution
    divided by squared integral.
    
    Strategy:
    1. Formulate as spectral phase retrieval.
    2. Fix magnitude spectrum to a shape that maximizes the DC component (sum)
       to minimize the ratio max|c|/S^2.
    3. The theoretical optimum is the all-ones sequence (S=N, max|c|=N, score=1/N).
    4. We use Gerchberg-Saxton iterations with a target magnitude corresponding
       to the all-ones sequence (delta at DC) and a flattop window to find
       robust solutions, then select the best.
    """
    
    def score(x):
        X = np.fft.fft(x)
        power = X * np.conj(X)
        c = np.fft.ifft(power).real
        S = np.sum(x)
        if abs(S) < 1e-10:
            return 1e10
        return np.max(np.abs(c)) / (S * S)
        
    # Candidate 1: All ones (Delta spectrum)
    x_ones = np.ones(N)
    M_ones = np.abs(np.fft.fft(x_ones))
    
    # Candidate 2: Flattop window magnitude
    w = flattop(N)
    M_flattop = np.abs(np.fft.fft(w))
    
    results = []
    
    for M_target in [M_ones, M_flattop]:
        np.random.seed(42)
        # Initialize with random phase
        phase = np.exp(1j * np.random.rand(N) * 2 * np.pi)
        X = M_target * phase
        
        for _ in range(100):
            x_temp = np.fft.ifft(X).real
            # Project to {-1, 1}
            x_proj = np.sign(x_temp)
            # Handle zeros (map to 1)
            x_proj[x_proj == 0] = 1.0
            
            # Update phase
            X_new = np.fft.fft(x_proj)
            abs_X_new = np.abs(X_new)
            
            # Avoid division by zero
            mask = abs_X_new > 1e-10
            phase_new = np.ones(N, dtype=complex)
            phase_new[mask] = X_new[mask] / abs_X_new[mask]
            
            X = M_target * phase_new
            
        x_final = np.sign(np.fft.ifft(X).real)
        x_final[x_final == 0] = 1.0
        
        results.append((score(x_final), x_final))
        
    # Return the best
    results.sort(key=lambda item: item[0])
    return results[0][1]

# EVOLVE-BLOCK-END
