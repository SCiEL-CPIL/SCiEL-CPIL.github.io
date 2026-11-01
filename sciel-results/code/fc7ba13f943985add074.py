"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


_NFFT = 1 << (2 * N - 1).bit_length()


def _autocorr(h):
    """Compute aperiodic autocorrelation using FFT. Returns array of length N."""
    F = np.fft.rfft(h, n=_NFFT)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=_NFFT).real[:N]
    return acf


def _c3_value(h):
    """Compute C3 bound: sum of |sidelobe|^3 normalized."""
    acf = _autocorr(h)
    sidelobes = acf[1:]
    return np.sum(np.abs(sidelobes) ** 3) / (N ** 3)


def _smooth_objective(x, beta=10.0):
    """Smooth continuous objective using tanh parameterization and log-sum-exp."""
    h = np.tanh(x)
    acf = _autocorr(h)
    sidelobes = acf[1:]
    cubes = np.abs(sidelobes) ** 3
    
    max_c = np.max(cubes)
    if max_c < 1e-15:
        return 0.0
    
    # Log-sum-exp for smooth sum of cubes
    shifted = cubes / beta - max_c / beta
    lse = max_c / beta + np.log(np.sum(np.exp(shifted)))
    return lse * beta / (N ** 3)


def _local_search_polish(h, deadline, max_rounds=100):
    """Deterministic local search: flip single bits to reduce C3."""
    best_h = h.copy()
    best_val = _c3_value(best_h)
    
    for _ in range(max_rounds):
        if time.time() >= deadline:
            break
        improved = False
        for i in range(N):
            if time.time() >= deadline:
                break
            best_h[i] = -best_h[i]
            val = _c3_value(best_h)
            if val < best_val:
                best_val = val
                improved = True
            else:
                best_h[i] = -best_h[i]  # revert
        if not improved:
            break
    
    return best_h, best_val


def _lbfgs_relax(x0, deadline, maxiter=200):
    """Run L-BFGS-B on the smooth continuous objective."""
    def objective(x):
        return _smooth_objective(x, beta=10.0)
    
    res = minimize(objective, x0, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-6})
    
    if time.time() >= deadline:
        return x0
    return res.x


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    rng = np.random.default_rng(42)
    
    best_discrete_h = None
    best_val = np.inf
    
    # Multiple restarts with L-BFGS-B continuous relaxation
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() >= deadline - 20.0:
            break
        
        # Different initializations for each restart
        if restart == 0:
            # Start from all positive (tanh parameter = large positive)
            x0 = np.full(N, 3.0)
        elif restart == 1:
            # Start from alternating pattern
            x0 = np.where(np.arange(N) % 2 == 0, 3.0, -3.0)
        else:
            # Random initialization
            x0 = rng.standard_normal(N) * 2.0
        
        # Run L-BFGS-B to find a good continuous solution
        x_opt = _lbfgs_relax(x0, deadline, maxiter=150)
        
        # Project to discrete ±1
        h_discrete = np.sign(np.tanh(x_opt))
        # Handle zeros (shouldn't happen with tanh, but just in case)
        h_discrete[h_discrete == 0] = 1.0
        
        # Evaluate discrete value
        val = _c3_value(h_discrete)
        
        if val < best_val:
            best_val = val
            best_discrete_h = h_discrete.copy()
    
    # If no discrete solution found, use random
    if best_discrete_h is None:
        best_discrete_h = rng.choice([-1.0, 1.0], size=N)
        best_val = _c3_value(best_discrete_h)
    
    # Final deterministic local search polish
    if time.time() < deadline:
        polished_h, polished_val = _local_search_polish(best_discrete_h, deadline)
        if polished_val < best_val:
            best_discrete_h = polished_h
            best_val = polished_val
    
    return best_discrete_h
# EVOLVE-BLOCK-END
