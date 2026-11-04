import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Helper to compute autocorrelation using FFT
    def get_autocorr(x):
        # x is length N
        # We need linear convolution of x with x
        # Pad to next power of 2 for efficiency? N=400, 2N-1=799. Next pow2 is 1024.
        n_fft = 1024
        X = np.fft.rfft(x, n=n_fft)
        C = np.fft.irfft(X * X, n=n_fft)
        # The linear convolution is in the first 2N-1 elements
        # Note: irfft returns real values
        return C[:2*N-1]
    
    # Objective function: Soft max of absolute autocorrelation
    def objective(x):
        c = get_autocorr(x)
        # Use soft max to approximate max |c|
        # lambda controls the sharpness
        lam = 50.0
        # To avoid overflow, subtract max
        abs_c = np.abs(c)
        max_c = np.max(abs_c)
        if max_c == 0:
            return 0.0
        # log(sum(exp(lam * (abs_c - max_c)))) + lam * max_c
        # This is stable
        val = np.log(np.sum(np.exp(lam * (abs_c - max_c)))) / lam + max_c
        return val

    # Initial Guess: Flat Spectrum
    # We want |X|^2 to be flat.
    # X is the FFT of x.
    # Let X_k = e^{j phi_k} for k != 0, and X_0 = 1.
    # To ensure x is real, we need X_{N-k} = conj(X_k).
    # We generate random phases for k=1 to N//2.
    
    rng = np.random.RandomState(42)
    
    # Try a few different random initializations and pick the best one after a quick run
    best_x = None
    best_score = float('inf')
    
    for attempt in range(5):
        if (time.time() - start_time) > deadline - 10.0:
            break
            
        # Generate random phases
        # N=400. FFT bins 0..399.
        # Real signal: X[k] = conj(X[N-k])
        # We define X[1] to X[N//2]
        
        X = np.zeros(N//2 + 1, dtype=complex)
        X[0] = 1.0 + 0.0j # DC component
        
        # Random phases for the rest
        phases = rng.uniform(0, 2*np.pi, size=N//2)
        X[1:] = np.exp(1j * phases)
        
        # Ensure symmetry for IFFT
        # scipy.fft.irfft expects the positive frequency bins
        x_init = np.fft.irfft(X, n=N)
        
        # Normalize sum to 1
        s = np.sum(x_init)
        if abs(s) < 1e-10:
            continue
        x_init = x_init / s
        
        # Quick polish with L-BFGS-B
        # Limit time per attempt
        attempt_deadline = start_time + (deadline - start_time) / 6.0 # Allow ~5 attempts
        
        try:
            res = minimize(objective, x_init, method='L-BFGS-B', 
                           options={'maxiter': 50, 'maxfun': 200, 'disp': False,
                                    'ftol': 1e-6, 'gtol': 1e-6})
            
            if res.success or res.fun < best_score:
                x_candidate = res.x
                # Evaluate true metric
                c = get_autocorr(x_candidate)
                max_abs_c = np.max(np.abs(c))
                sum_x = np.sum(x_candidate)
                if abs(sum_x) < 1e-10:
                    continue
                score = max_abs_c / (sum_x * sum_x)
                
                if score < best_score:
                    best_score = score
                    best_x = x_candidate.copy()
                    
        except Exception:
            pass

    # If no good result found, fallback to random signs
    if best_x is None:
        x = rng.choice([-1.0, 1.0], size=N)
        s = np.sum(x)
        if s == 0:
            x[0] = -x[0]
            s = np.sum(x)
        best_x = x / s
        c = get_autocorr(best_x)
        best_score = np.max(np.abs(c)) / (s*s)

    # Final Local Search on the best continuous solution
    # We can try small perturbations
    # Or project to signs if that helps?
    # The problem asks for "signed step heights". 
    # If I return continuous values, is it valid?
    # "Signed equal-width step functions"
    # Usually implies the values are constant on steps.
    # The heights can be anything.
    
    # Let's do a few iterations of "sign flipping" on the continuous solution to see if it improves
    # Actually, flipping signs of continuous values is not a standard move.
    # Better: Use the continuous solution as is.
    
    # One more polish with a different lambda or method?
    # Let's try a few more iterations of L-BFGS-B on the best_x found
    if (time.time() - start_time) < deadline - 5.0:
        try:
            res2 = minimize(objective, best_x, method='L-BFGS-B', 
                           options={'maxiter': 100, 'maxfun': 500, 'disp': False,
                                    'ftol': 1e-8, 'gtol': 1e-8})
            if res2.fun < best_score:
                # Check true metric
                c = get_autocorr(res2.x)
                max_abs_c = np.max(np.abs(c))
                sum_x = np.sum(res2.x)
                if abs(sum_x) > 1e-10:
                    score2 = max_abs_c / (sum_x * sum_x)
                    if score2 < best_score:
                        best_x = res2.x
                        best_score = score2
        except Exception:
            pass

    return best_x
