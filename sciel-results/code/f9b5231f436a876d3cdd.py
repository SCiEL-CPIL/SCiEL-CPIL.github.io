"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # New Approach: Spectral Method (Error Reduction) + Local Search
    # Objective: Minimize ISL (Integrated Sidelobe Level)
    # Method: Iterate between time and frequency domains to flatten the spectrum.
    
    def compute_isl(f):
        F = np.fft.fft(f)
        power = F * np.conj(F)
        R = np.real(np.fft.ifft(power))
        # R[0] is DC, exclude it
        return np.sum(R[1:]**2)

    # 1. Spectral Iteration to find a good starting point
    # Initialize random binary
    f = np.random.choice([-1, 1], size=N)
    
    # Tapering factor for the target spectrum to reduce high-freq sidelobes
    # A flat spectrum is ideal for low ISL, but a slight taper can help.
    # Let's try a simple raised cosine taper on the target magnitude.
    k = np.arange(N)
    # Create a target magnitude profile. 
    # Center is DC. We want to suppress high frequencies? 
    # Actually, for binary sequences, the spectrum is complex.
    # Standard "flat" target: |F| = 1.
    # Let's use a target that is 1 for low freqs and tapers off.
    # However, for N=400, a flat spectrum is usually the best bet for minimal ISL.
    target_mag = np.ones(N)
    
    iters = 200
    for _ in range(iters):
        F = np.fft.fft(f)
        # Avoid division by zero
        mag = np.abs(F)
        mag[mag < 1e-12] = 1e-12
        phase = F / mag
        
        # New spectrum with target magnitude
        F_new = target_mag * phase
        
        # Inverse transform
        f_real = np.real(np.fft.ifft(F_new))
        
        # Threshold to binary
        # Add small noise to break symmetry
        f_real += np.random.normal(0, 0.1, N)
        f = np.sign(f_real)
        
    # 2. Local Search (Steepest Descent / Random Hill Climbing) to polish
    # Start from the spectrally derived solution
    current_isl = compute_isl(f)
    best_f = f.copy()
    best_isl = current_isl
    
    # Perform local search with random restarts of the search process
    # We will do a fixed number of iterations of single-bit flips
    max_iter = 20000
    # Use a simple cooling schedule or just random search with acceptance
    # Let's use a simple simulated annealing on top of the spectral start
    T = 1.0
    cooling_rate = 0.9995
    min_T = 1e-3
    
    for i in range(max_iter):
        if T < min_T:
            break
            
        idx = np.random.randint(0, N)
        old_val = f[idx]
        f[idx] = -f[idx]
        
        new_isl = compute_isl(f)
        delta = new_isl - current_isl
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_isl = new_isl
            if new_isl < best_isl:
                best_isl = new_isl
                best_f = f.copy()
        else:
            f[idx] = old_val
            
        T *= cooling_rate
        
    return best_f
# EVOLVE-BLOCK-END
