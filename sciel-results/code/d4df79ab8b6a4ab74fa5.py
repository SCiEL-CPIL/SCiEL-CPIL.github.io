import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE = 135.0

def _autocorr_fft(x):
    """Compute aperiodic autocorrelation using FFT."""
    n = len(x)
    n_fft = 2 * n
    X = np.fft.rfft(x, n=n_fft)
    acf = np.fft.irfft(X * np.conj(X), n=n_fft)
    return acf[:n]

def _max_sidelobe(f):
    """Compute the maximum absolute sidelobe of the autocorrelation."""
    acf = _autocorr_fft(f)
    if len(acf) <= 1:
        return 0.0
    return np.max(np.abs(acf[1:]))

def _local_search(f, deadline):
    """Local search via bit flips with time budget."""
    n = len(f)
    f = f.copy()
    current_max = _max_sidelobe(f)
    
    # Use a list of indices to shuffle for diversity
    indices = np.arange(n)
    
    # Run multiple passes until deadline
    while time.time() < deadline:
        np.random.shuffle(indices)
        improved = False
        for j in indices:
            if time.time() > deadline:
                break
            
            old_val = f[j]
            f[j] = -old_val
            new_max = _max_sidelobe(f)
            
            if new_max < current_max:
                current_max = new_max
                improved = True
            else:
                f[j] = old_val
        
        if not improved:
            # If no improvement in a full pass, try a few random restarts or break
            # For simplicity, if no improvement, we can stop or try a different strategy
            # Let's break to save time for other candidates
            break
            
    return f, current_max

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    # Phase 1: Fourier Phase Retrieval / Spectral Flattening
    # We want to minimize the peak sidelobe of the autocorrelation.
    # This is equivalent to minimizing the peak of the power spectral density (PSD)
    # relative to the mean, or flattening the PSD.
    
    # Let X be the DFT of the sequence x.
    # PSD = |X|^2.
    # We can optimize the phases of X to flatten |X|^2.
    
    # Initialize with random phases
    n = N
    # We'll work with the full FFT for symmetry, or rfft.
    # Let's use rfft for efficiency.
    n_fft = 2 * n
    
    # Start with a random complex spectrum with unit magnitude?
    # Or start from a random time-domain sequence and iterate.
    
    best_f = None
    best_score = float('inf')
    
    # Try a few random initializations
    for restart in range(5):
        if time.time() - start_time > DEADLINE - 30:
            break
            
        # Random time-domain sequence
        x_init = np.random.randn(n)
        
        # Initialize phases
        X = np.fft.rfft(x_init, n=n_fft)
        mags = np.abs(X)
        # Avoid division by zero
        mags[mags == 0] = 1e-10
        phases = np.exp(1j * np.angle(X))
        
        # Gradient descent on phases to minimize soft-max of PSD sidelobes
        # Objective: minimize max(|X|^2) / mean(|X|^2) or similar.
        # Let's minimize the soft-max of the PSD values.
        
        # Note: The autocorrelation sidelobes are the inverse FFT of the PSD minus the DC term.
        # Minimizing peak sidelobe in time domain is equivalent to flattening the PSD.
        
        # We will optimize the phase vector theta.
        # X_new = mags * exp(1j * theta)
        # x_new = ifft(X_new)
        
        # To keep x real, we need Hermitian symmetry.
        # rfft handles this if we construct the full spectrum correctly or use irfft.
        # irfft expects the positive frequency half.
        
        # Let's optimize the phases of the rfft components.
        # Constraint: The resulting time-domain signal should be real.
        # irfft guarantees this if we pass a complex array of length n//2 + 1.
        
        # However, the magnitude of the DC and Nyquist bins (if n is even) are real.
        # For rfft of real signal:
        # X[0] is real (DC)
        # X[n//2] is real (Nyquist) if n is even
        # Others are complex.
        
        # We can fix the magnitudes to be roughly uniform?
        # If we set |X| to be constant, the time-domain signal is a Dirac comb (impulse train).
        # For a finite sequence, this corresponds to a specific structure.
        
        # Let's try to flatten the PSD by adjusting phases while keeping magnitudes fixed?
        # Or adjust both?
        # The directive says: "optimizing the complex-valued Fourier phase to minimize the peak sidelobe level of the magnitude spectrum".
        # So we fix magnitudes?
        # If we fix magnitudes to be uniform, we get a specific sequence.
        # Let's try uniform magnitudes first.
        
        # Uniform magnitude spectrum
        target_mags = np.ones(n//2 + 1)
        # Scale to match energy?
        # Energy in time domain = sum(x^2).
        # Parseval: sum(x^2) = 1/n_fft * sum(|X|^2).
        # If |X|^2 = C, then sum(x^2) = C * (n//2 + 1) / n_fft.
        
        # Let's just optimize phases with uniform magnitudes.
        # theta is the phase vector.
        # We want to minimize the peak sidelobe of the autocorrelation of x.
        
        # Let's define the objective function for phase optimization.
        # x = irfft(target_mags * exp(1j * theta))
        # acf = autocorr(x)
        # obj = max(|acf[1:]|)
        
        # Gradient of obj w.r.t theta?
        # This is tricky because of the max and the FFT.
        # We can use a smooth approximation (soft-max) and backpropagate through FFT.
        
        # Let's implement a simple gradient descent on phases.
        
        # Initialize theta randomly
        theta = np.random.randn(n//2 + 1)
        
        # Handle real constraints for DC and Nyquist
        # For DC (index 0), phase is 0 or pi.
        # For Nyquist (index n//2), phase is 0 or pi.
        # Let's mask these out or handle them separately.
        
        # Actually, irfft assumes the input is the positive half of a Hermitian symmetric spectrum.
        # The DC and Nyquist components must be real.
        # So their phases are effectively 0 or pi.
        # We can optimize the signs of DC and Nyquist separately, or just fix them to 1 for now.
        
        # Let's optimize the complex phases for indices 1 to n//2-1.
        # And treat DC and Nyquist as real variables (amplitude).
        
        # Let's simplify:
        # Fix magnitudes to 1.
        # Optimize phases for 1..n//2-1.
        # Fix DC and Nyquist to 1.
        
        # X = [1, mags[1]*exp(1j*theta[1]), ..., 1]
        
        # Let's run gradient descent.
        
        for step in range(100):
            if time.time() - start_time > DEADLINE - 30:
                break
                
            # Construct X
            X = np.zeros(n//2 + 1, dtype=complex)
            X[0] = 1.0
            if n % 2 == 0:
                X[-1] = 1.0
            X[1:-1] = np.exp(1j * theta[1:-1])
            
            # Compute x
            x = np.fft.irfft(X, n=n_fft)
            x = x[:n]
            
            # Compute autocorrelation
            acf = _autocorr_fft(x)
            
            # Soft max objective
            if len(acf) <= 1:
                break
            sidelobes = np.abs(acf[1:])
            max_val = np.max(sidelobes)
            if max_val == 0:
                break
            beta = 10.0
            shifted = beta * (sidelobes - max_val)
            exp_shifted = np.exp(shifted)
            weights = exp_shifted / np.sum(exp_shifted)
            
            # Gradient of soft-max w.r.t acf
            # d obj / d acf[k] = weights[k-1] * sign(acf[k])
            
            # We need d acf[k] / d theta[j]
            # acf[k] = sum_{i} x[i] x[i+k]
            # x = irfft(X)
            # This is complex.
            
            # Alternative: Use a finite difference approximation for the gradient?
            # Or just use a random search / hill climbing on the phases?
            
            # Given the time constraints, let's try a simpler approach:
            # Generate candidate sequences from uniform-magnitude spectra with random phases,
            # and use local search to polish them.
            
            # This avoids the complexity of gradient computation through FFT.
            
            # Let's generate 50 candidates per restart.
            for cand in range(50):
                if time.time() - start_time > DEADLINE - 20:
                    break
                    
                # Random phases
                theta_rand = np.random.randn(n//2 + 1)
                X_rand = np.zeros(n//2 + 1, dtype=complex)
                X_rand[0] = 1.0
                if n % 2 == 0:
                    X_rand[-1] = 1.0
                X_rand[1:-1] = np.exp(1j * theta_rand[1:-1])
                
                x_cand = np.fft.irfft(X_rand, n=n_fft)
                x_cand = x_cand[:n]
                
                # Threshold to binary
                f_cand = np.sign(x_cand)
                f_cand[f_cand == 0] = 1.0
                
                # Evaluate
                score_cand = _max_sidelobe(f_cand)
                
                if score_cand < best_score:
                    best_score = score_cand
                    best_f = f_cand.copy()
                    
                # Local search polish
                if time.time() - start_time < DEADLINE - 10:
                    f_polished, score_polished = _local_search(f_cand, DEADLINE - 10)
                    if score_polished < best_score:
                        best_score = score_polished
                        best_f = f_polished.copy()
                        
    if best_f is None:
        # Fallback to random
        best_f = np.random.choice([-1, 1], size=n)
        
    return best_f
# EVOLVE-BLOCK-END
