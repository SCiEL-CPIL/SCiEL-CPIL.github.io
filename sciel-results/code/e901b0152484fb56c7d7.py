import numpy as np
from scipy.optimize import minimize
import time

N = 400

def compute_score(x):
    """Compute max |autocorr| / (sum x)^2"""
    S = np.sum(x)
    if abs(S) < 1e-10:
        return 1e10
    # Linear autocorrelation
    R = np.correlate(x, x, mode='full')
    # R has length 2N-1, centered at N-1
    # Take the max absolute value
    max_abs_R = np.max(np.abs(R))
    return max_abs_R / (S * S)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_x = np.ones(N)
    best_score = compute_score(best_x)
    
    # Strategy: Flat-spectrum design via spectral parameterization
    # Use a parametric model: X[k] = A * exp(i * phi[k]) where phi is a smooth function
    # Optimize the phase parameters to minimize autocorrelation peaks
    
    # Approach: Use inverse DFT with optimized phases
    # Parameterize phases as a low-dimensional function
    
    def objective(params):
        # params: coefficients for phase function
        k = np.arange(N)
        # Phase as a polynomial/chirp: phi[k] = sum_j params[j] * k^j
        # Use up to a few terms
        phi = np.zeros(N)
        for j in range(len(params)):
            phi += params[j] * (k ** j) / (N ** j)  # Normalize
        # Complex signal
        X = np.exp(1j * phi)
        # Inverse DFT to get real signal
        x_complex = np.fft.ifft(X)
        x_real = np.real(x_complex)
        # Quantize to ±1
        x = np.sign(x_real)
        x[x == 0] = 1
        return compute_score(x)
    
    # Try multiple random initializations
    np.random.seed(42)
    n_params = 6  # Low-dimensional parameterization
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Random initial parameters
        params0 = np.random.randn(n_params) * 0.5
        
        # Use Nelder-Mead or similar
        try:
            result = minimize(objective, params0, method='Nelder-Mead', 
                            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
        except:
            continue
        
        if result.fun < best_score:
            best_score = result.fun
            # Reconstruct best_x
            k = np.arange(N)
            phi = np.zeros(N)
            for j in range(len(result.x)):
                phi += result.x[j] * (k ** j) / (N ** j)
            X = np.exp(1j * phi)
            x_complex = np.fft.ifft(X)
            x_real = np.real(x_complex)
            best_x = np.sign(x_real)
            best_x[best_x == 0] = 1
    
    # Also try direct random search with local refinement
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Random ±1 sequence
        x = np.random.choice([-1, 1], size=N)
        score = compute_score(x)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
        
        # Local search: flip random bits
        x_trial = x.copy()
        for _ in range(100):
            idx = np.random.randint(N)
            x_trial[idx] *= -1
            score_trial = compute_score(x_trial)
            if score_trial < score:
                score = score_trial
                x = x_trial.copy()
            else:
                x_trial[idx] *= -1  # Revert
    
        if score < best_score:
            best_score = score
            best_x = x.copy()
    
    return best_x
