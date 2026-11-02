"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autocorr_sidelobes(a):
    """
    Compute the maximum absolute value of the autocorrelation 
    for non-zero lags.
    """
    n = len(a)
    size = 2 * n
    fa = np.fft.rfft(a, size)
    power = fa * np.conj(fa)
    c = np.fft.irfft(power, size)
    sidelobes = c[1:n]
    return np.max(np.abs(sidelobes))

def _score_val(a):
    """
    Calculate the raw value: max_abs_sidelobe / (sum a^2)^2
    Assuming sum a^2 = N for binary sequences.
    """
    max_side = _autocorr_sidelobes(a)
    integral_sq = (np.sum(a**2))**2
    return max_side / integral_sq

def _psd_variance(x):
    """
    Objective function for continuous relaxation:
    Minimize the variance of the Power Spectral Density.
    A flatter PSD tends to yield lower peak autocorrelation sidelobes.
    """
    n = len(x)
    # Compute PSD
    fx = np.fft.rfft(x)
    psd = np.abs(fx)**2
    # Variance of PSD
    return np.var(psd)

def _psd_variance_grad(x):
    """
    Gradient of PSD variance.
    This is complex to derive analytically for FFT, so we might use numerical 
    or a simpler proxy. However, for L-BFGS, we can provide a numerical gradient 
    or just let it approximate. 
    Actually, let's use a simpler objective: minimize the max PSD peak? No, that's non-smooth.
    Let's stick to minimizing the sum of squares of the PSD (which is constant by Parseval) 
    minus the square of the mean? 
    Wait, sum(psd) is constant for fixed norm. 
    Minimizing variance is equivalent to maximizing (mean^2 - variance) or similar.
    
    Let's try a different smooth proxy: 
    Minimize the L4 norm of the PSD? Or just use numerical gradient for L-BFGS.
    Given time constraints, we will rely on L-BFGS-B's numerical differentiation 
    or provide a simple gradient if possible.
    
    Actually, a better proxy for "flatness" that is smooth:
    Minimize sum(psd^2). Since sum(psd) is constant (Parseval), 
    minimizing sum(psd^2) is equivalent to minimizing variance.
    
    Gradient of sum(psd^2) where psd = |X|^2.
    This is a quartic function of x.
    We can use scipy.optimize.minimize with method='L-BFGS-B' and let it compute 
    the gradient numerically, or we can implement it. 
    For safety and speed, we will use numerical gradient if analytic is too complex, 
    but N=400 is small enough for numerical grad in L-BFGS if we limit iterations.
    """
    # We will use the objective function directly and let L-BFGS handle the gradient 
    # numerically if we don't provide one, but providing one is faster.
    # Let's use a finite difference check or just pass None for grad to minimize.
    pass

def _spectral_optimize(rng, time_budget):
    """
    Phase 1: Continuous optimization to find a 'flat-spectrum' structure.
    """
    # Initialize with random signs
    x0 = rng.choice([-1.0, 1.0], size=N)
    
    # We want to minimize PSD variance.
    # Objective: var(|FFT(x)|^2)
    def objective(x):
        fx = np.fft.rfft(x)
        psd = np.abs(fx)**2
        return np.var(psd)
    
    # Use L-BFGS-B
    # x is continuous, but we start at +/-1.
    # We don't constrain x to +/-1 during optimization, allowing it to move.
    # After optimization, we will threshold to +/-1.
    
    # To save time, we limit the number of iterations.
    # N=400, 1D FFT is fast.
    result = minimize(
        objective, 
        x0, 
        method='L-BFGS-B', 
        options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
    )
    
    # Threshold the result to binary
    x_opt = result.x
    a_binary = np.sign(x_opt)
    # Ensure no zeros
    a_binary[a_binary == 0] = 1.0
    
    return a_binary

def _peak_targeted_repair(a, time_budget):
    """
    Phase 2: Discrete repair targeting the maximum sidelobe lag.
    """
    current_val = _score_val(a)
    best_a = a.copy()
    best_val = current_val
    
    deadline = time.time() + time_budget
    
    no_improve_count = 0
    
    while time.time() < deadline and no_improve_count < 500:
        # Find the lag with the maximum absolute autocorrelation
        n = len(a)
        size = 2 * n
        fa = np.fft.rfft(a, size)
        power = fa * np.conj(fa)
        c = np.fft.irfft(power, size)
        
        # Get sidelobes
        sidelobes = c[1:n]
        abs_sidelobes = np.abs(sidelobes)
        
        # Find the worst lag
        worst_lag = np.argmax(abs_sidelobes) + 1
        worst_val = c[worst_lag]
        
        if worst_val == 0:
            break
            
        # Strategy: Flip bits that contribute most to this lag's correlation.
        # c[k] = sum_{i} a[i] a[i+k]
        # To reduce |c[k]|, we want to flip a[i] such that a[i]a[i+k] has the same sign as c[k].
        # If c[k] > 0, we want to make a[i]a[i+k] negative.
        # If c[k] < 0, we want to make a[i]a[i+k] positive.
        # Flipping a[i] changes the term from a[i]a[i+k] to -a[i]a[i+k].
        # Change in c[k] = -2 * a[i] * a[i+k].
        # We want this change to have the opposite sign of c[k].
        # So we want -2 * a[i] * a[i+k] to have sign opposite to c[k].
        # i.e., sign(a[i] * a[i+k]) should be same as sign(c[k]).
        
        # Calculate contribution of each i to c[worst_lag]
        # Note: c[k] = sum_{i=0}^{n-k-1} a[i] a[i+k]
        # Also c[k] includes contributions from negative lags in the full convolution, 
        # but here we are looking at linear autocorrelation defined by the FFT method.
        # The FFT method computes circular autocorrelation if not padded, but we padded to 2N.
        # c[k] for k in 1..N-1 is sum_{i=0}^{N-k-1} a[i]a[i+k].
        
        # Let's compute the gradient of c[worst_lag] with respect to a[i] flips.
        # This is O(N) per lag.
        
        # We can simply pick a random i and check if flipping it reduces the max sidelobe.
        # But to be smarter, we pick i that is part of the sum for worst_lag.
        
        # Indices involved in c[worst_lag]:
        # i from 0 to N - worst_lag - 1.
        # Also, due to symmetry in autocorrelation, flipping a[i] affects c[k] for multiple k.
        # But we are targeting the current max.
        
        # Heuristic: Try flipping a random index that is "active" in the worst lag.
        # Active indices for lag k: 0 <= i < N-k.
        if worst_lag < N:
            # Choose a random active index
            idx = rng.integers(0, N - worst_lag)
            
            # Flip it
            a[idx] = -a[idx]
            
            new_val = _score_val(a)
            
            if new_val < best_val:
                best_val = new_val
                best_a = a.copy()
                no_improve_count = 0
            else:
                # Revert
                a[idx] = -a[idx]
                no_improve_count += 1
        else:
            break
            
    return best_a, best_val

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Spectral Optimization
    # Allocate 30 seconds for spectral optimization
    spectral_time = 30.0
    if time.time() - start_time < deadline - spectral_time:
        a_spectral = _spectral_optimize(rng, spectral_time)
    else:
        a_spectral = rng.choice([-1, 1], size=N)
    
    current_val = _score_val(a_spectral)
    best_a = a_spectral.copy()
    best_val = current_val
    
    # Phase 2: Peak-Targeted Repair
    # Allocate remaining time for repair
    remaining_time = deadline - time.time() - 5.0 # 5s safety margin
    if remaining_time > 0:
        repaired_a, repaired_val = _peak_targeted_repair(a_spectral, remaining_time)
        if repaired_val < best_val:
            best_val = repaired_val
            best_a = repaired_a
    
    # Phase 3: Quick Random Hill Climbing on the best result
    # Use any remaining time
    remaining_time = deadline - time.time() - 2.0
    if remaining_time > 0:
        a = best_a.copy()
        current_val = best_val
        no_improve = 0
        
        while time.time() < deadline - 2.0 and no_improve < 1000:
            idx = rng.integers(0, N)
            a[idx] = -a[idx]
            new_val = _score_val(a)
            
            if new_val < current_val:
                current_val = new_val
                no_improve = 0
                if current_val < best_val:
                    best_val = current_val
                    best_a = a.copy()
            else:
                a[idx] = -a[idx]
                no_improve += 1
                
    return best_a
# EVOLVE-BLOCK-END
