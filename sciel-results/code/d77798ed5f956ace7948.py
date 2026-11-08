import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def score(a):
        # Normalize to sum=1 for consistent comparison if needed, 
        # but the ratio is scale invariant.
        # We compute max |conv| / (sum a)^2
        s = np.sum(a)
        if abs(s) < 1e-12:
            return float('inf')
        conv = np.convolve(a, a, mode='full')
        return np.max(np.abs(conv)) / (s * s)
    
    # Baseline: Constant sequence
    best_a = np.ones(N) / N
    best_score = score(best_a)
    
    np.random.seed(42)
    
    # Spectral Shaping Strategy
    # We construct x as IDFT of a random phase mask with unit amplitude.
    # To ensure x is real, we enforce Hermitian symmetry in the spectrum.
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Generate random phase
        # Frequency bins 0 to N-1
        # For real signal, X[k] = conj(X[N-k])
        # Phase must satisfy phi[k] = -phi[N-k] (mod 2pi) for real IDFT?
        # Actually, if X[k] = A[k] exp(i phi[k]), and we want real x,
        # we can just generate random complex X, take real part of IDFT, and re-normalize.
        # Or enforce symmetry.
        
        # Let's use the simpler approach: Generate random complex spectrum, take real part of IDFT.
        # This effectively creates a random real signal with specific spectral properties.
        
        # Band-limited: Set high frequencies to 0 to reduce time-domain oscillations?
        # The uncertainty principle suggests that band-limiting increases time-domain width,
        # potentially reducing peaks if the total energy is fixed?
        # But we are minimizing max |autoconv| / sum^2.
        
        num_bands = N // 2
        # Create a random spectrum
        X = np.zeros(N, dtype=np.complex128)
        
        # Random amplitudes in [0, 1] and random phases
        # Keep DC component (k=0) and Nyquist (k=N/2 if even) real and positive?
        # Actually, let's just use random complex numbers and take real part.
        
        # To make it "band-limited", we only populate lower frequencies.
        cutoff = np.random.randint(N // 4, N // 2)
        
        # Generate random complex values for the first 'cutoff' bins
        # and enforce Hermitian symmetry for the rest to get a real signal.
        
        phases = np.random.uniform(0, 2 * np.pi, cutoff)
        amps = np.random.uniform(0.5, 1.0, cutoff) # Avoid zero amplitude to keep signal non-trivial
        
        # Set DC (k=0)
        X[0] = amps[0] # Real, positive
        
        # If N is even, Nyquist bin N/2 must be real
        if N % 2 == 0:
             if cutoff > N//2:
                 X[N//2] = amps[N//2]
        
        for k in range(1, cutoff):
             X[k] = amps[k] * np.exp(1j * phases[k])
             X[N-k] = np.conj(X[k])
        
        # If cutoff > N/2, we need to handle the upper half carefully?
        # The loop above covers 1 to cutoff-1.
        # If cutoff > N/2, we are overwriting the Hermitian symmetric parts incorrectly if not careful.
        # Let's stick to cutoff <= N//2 for simplicity and validity.
        
        # Re-calculate with safe cutoff
        cutoff = min(cutoff, N // 2)
        X = np.zeros(N, dtype=np.complex128)
        
        if cutoff > 0:
            phases = np.random.uniform(0, 2 * np.pi, cutoff)
            amps = np.random.uniform(0.5, 1.0, cutoff)
            
            X[0] = amps[0]
            
            for k in range(1, cutoff):
                X[k] = amps[k] * np.exp(1j * phases[k])
                X[N-k] = np.conj(X[k])
            
            if N % 2 == 0 and cutoff > N//2:
                 # This case is handled by cutoff <= N//2 above, so N//2 is not in 1..cutoff-1 if cutoff=N//2?
                 # If cutoff = N//2, k goes to N//2 - 1.
                 # We need to set X[N//2] separately if it's in the band.
                 if cutoff >= N//2:
                     X[N//2] = amps[N//2] if N//2 < len(amps) else 1.0
        
        # Inverse DFT
        a = np.fft.ifft(X).real
        
        # Normalize
        s = a.sum()
        if abs(s) < 1e-12:
            continue
        a /= s
        
        cur_score = score(a)
        
        if cur_score < best_score:
            best_score = cur_score
            best_a = a.copy()
            
        # Local search on the phase?
        # Simple perturbation of the time-domain signal as fallback/improvement
        # But the directive asks for spectral shaping.
        # We can also try to optimize the phases via gradient descent or random search.
        
        # Given the time budget, we just do multiple random restarts with different band limits.
        
    return best_a
