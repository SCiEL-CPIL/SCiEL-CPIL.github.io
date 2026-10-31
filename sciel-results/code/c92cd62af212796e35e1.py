import numpy as np
import scipy.optimize
import scipy.signal
import time

N = 400

def _compute_bound(f):
    """
    Computes the peak sidelobe magnitude of the linear autocorrelation.
    """
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    PSD = F * np.conj(F)
    R = np.fft.irfft(PSD, n=n_fft)
    # R[0] is the peak (energy)
    # R[1:N] are the sidelobes
    return np.max(np.abs(R[1:N]))

def _objective_smooth_peak(f, p=20):
    """
    Objective function: Sum of high-power sidelobes.
    This approximates the minimization of the maximum sidelobe (L-infinity)
    while remaining differentiable.
    """
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    PSD = F * np.conj(F)
    R = np.fft.irfft(PSD, n=n_fft)
    sidelobes = R[1:N]
    # Use power 20 to strongly penalize large sidelobes
    return np.sum(np.abs(sidelobes)**p)

def _local_search_peak(f, max_iters=500, temp=0.01, cool_rate=0.99):
    """
    Simulated Annealing / Local Search to minimize peak sidelobe.
    """
    current_f = f.copy()
    current_score = _compute_bound(current_f)
    best_f = current_f.copy()
    best_score = current_score
    
    for i in range(max_iters):
        # Pick a random index to perturb
        idx = np.random.randint(0, N)
        
        # Generate a perturbation
        # Small Gaussian noise
        delta = np.random.normal(0, temp)
        new_f = current_f.copy()
        new_f[idx] += delta
        
        # Normalize to maintain unit energy
        norm = np.linalg.norm(new_f)
        if norm > 1e-9:
            new_f = new_f / norm
        
        new_score = _compute_bound(new_f)
        
        # Acceptance criterion
        if new_score < current_score or np.random.rand() < np.exp(-(new_score - current_score) / max(temp, 1e-9)):
            current_f = new_f
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_f = current_f.copy()
        
        temp *= cool_rate
        
    return best_f

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    best_f = None
    best_score = float('inf')

    # 1. Try optimization with high-power objective from multiple starts
    # Start with Kaiser windows of varying beta
    for beta in [8, 12, 16, 20]:
        if time.time() > deadline - 20:
            break
        f_init = scipy.signal.windows.kaiser(N, beta=beta)
        f_init = f_init / np.linalg.norm(f_init)
        
        try:
            # Optimize the smoothed peak
            res = scipy.optimize.minimize(
                _objective_smooth_peak, 
                f_init, 
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            f_opt = res.x
            f_opt = f_opt / np.linalg.norm(f_opt)
            
            # Polish with local search
            f_local = _local_search_peak(f_opt, max_iters=200, temp=0.005, cool_rate=0.995)
            
            score = _compute_bound(f_local)
            
            if score < best_score:
                best_score = score
                best_f = f_local.copy()
                
        except Exception as e:
            continue

    # 2. Try a pure simulated annealing start from a Gaussian signal
    if time.time() < deadline - 10:
        x = np.linspace(-1, 1, N)
        f_init = np.exp(-3 * x**2)
        f_init = f_init / np.linalg.norm(f_init)
        
        f_sa = _local_search_peak(f_init, max_iters=500, temp=0.01, cool_rate=0.99)
        score = _compute_bound(f_sa)
        if score < best_score:
            best_score = score
            best_f = f_sa.copy()

    # Fallback
    if best_f is None:
        f_init = np.ones(N) / np.sqrt(N)
        best_f = f_init

    return best_f
