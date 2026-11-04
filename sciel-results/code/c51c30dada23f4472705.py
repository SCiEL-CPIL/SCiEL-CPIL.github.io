"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt
import scipy.fft as fft

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.fft as fft

def _autoconv_lin(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    c = np.fft.irfft(X * X, fsize)[:size]
    return c

def _metric_offdiag(x):
    """Compute max |c_k| for k != 0 divided by (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e9
    c = _autoconv_lin(x)
    # Exclude k=0 (the DC term, which is (sum x)^2)
    max_abs = np.max(np.abs(c[1:]))
    return max_abs / (s * s)

def _soft_max_abs(c, temp=0.01):
    """Smooth approximation of max |c| using log-sum-exp."""
    abs_c = np.abs(c)
    # Avoid numerical issues
    abs_c = np.maximum(abs_c, 1e-12)
    scaled = abs_c / temp
    max_scaled = np.max(scaled)
    result = temp * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    return result

def _objective_phase(phases, temp=0.01):
    """
    Objective for phase optimization.
    phases: array of phases (length N)
    We fix amplitude to 1, so x_k = exp(i * phases_k)
    Then we compute the real part and scale to satisfy sum constraint.
    """
    # Complex signal with unit amplitude
    x_complex = np.exp(1j * phases)
    # Take real part as our signed step heights
    x_real = np.real(x_complex)
    
    # Normalize to sum 1
    s = np.sum(x_real)
    if abs(s) < 1e-12:
        return 1e9
    x_norm = x_real / s
    
    # Compute autoconvolution
    c = _autoconv_lin(x_norm)
    
    # Use soft max for differentiability, excluding k=0
    return _soft_max_abs(c[1:], temp) / (1.0)  # (sum x_norm)^2 = 1

def _gradient_phase(phases, temp=0.01):
    """
    Compute gradient of the soft-max objective w.r.t. phases.
    Uses automatic differentiation via finite differences for simplicity,
    or we can derive it analytically.
    """
    # For efficiency, use a simple finite difference gradient
    eps = 1e-7
    grad = np.zeros_like(phases)
    f0 = _objective_phase(phases, temp)
    
    for i in range(len(phases)):
        phases_plus = phases.copy()
        phases_plus[i] += eps
        f_plus = _objective_phase(phases_plus, temp)
        grad[i] = (f_plus - f0) / eps
    
    return grad

def _gd_phase(phases0, max_iters=5000, lr=0.01, temp=0.01, deadline=None):
    """
    Gradient descent on phases.
    """
    phases = phases0.copy()
    best_val = _objective_phase(phases, temp)
    best_phases = phases.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _gradient_phase(phases, temp)
        phases -= lr * grad
        
        val = _objective_phase(phases, temp)
        if val < best_val:
            best_val = val
            best_phases = phases.copy()
    
    return best_phases, best_val

def _phases_to_heights(phases):
    """Convert phases to real signed heights, normalized to sum 1."""
    x_complex = np.exp(1j * phases)
    x_real = np.real(x_complex)
    s = np.sum(x_real)
    if abs(s) < 1e-12:
        return np.ones(N)
    return x_real / s

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
    # Baseline: all ones
    best_A = np.ones(N)
    best_val = _metric_offdiag(best_A)
    
    # Helper to check time
    def time_up():
        return time.time() > deadline
    
    # Generate diverse initial phase candidates
    candidates = []
    
    # 1. All zero phases (corresponds to all ones in time domain)
    candidates.append(np.zeros(N))
    
    # 2. Random phases
    for _ in range(10):
        candidates.append(np.random.uniform(0, 2 * np.pi, size=N))
    
    # 3. Structured: linear phase ramp
    candidates.append(np.linspace(0, 2 * np.pi, N, endpoint=False))
    
    # 4. Structured: quadratic phase
    i = np.arange(N)
    candidates.append(2 * np.pi * (i / N) ** 2 % 1.0)
    
    # 5. Structured: block phases
    for block_size in [2, 4, 8, 16]:
        phases = []
        for j in range(0, N, block_size):
            phase_val = np.pi * (j // block_size) % (2 * np.pi)
            phases.extend([phase_val] * min(block_size, N - j))
        candidates.append(np.array(phases))
    
    # Optimize each candidate using gradient descent on phases
    for i, phases0 in enumerate(candidates):
        if time_up():
            break
        
        try:
            # Adaptive learning rate: start small, can increase
            lr = 0.005
            temp = 0.02  # Start with smoother surrogate
            
            # Phase 1: Coarse optimization with larger temp
            phases_coarse, val_coarse = _gd_phase(phases0, max_iters=2000, lr=lr, temp=temp, deadline=deadline)
            
            if time_up():
                # Still try to convert current best
                A_candidate = _phases_to_heights(phases_coarse)
                val_candidate = _metric_offdiag(A_candidate)
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_A = A_candidate
                continue
            
            # Phase 2: Fine optimization with smaller temp
            temp_fine = 0.005
            phases_fine, val_fine = _gd_phase(phases_coarse, max_iters=3000, lr=lr * 0.5, temp=temp_fine, deadline=deadline)
            
            if time_up():
                A_candidate = _phases_to_heights(phases_fine)
                val_candidate = _metric_offdiag(A_candidate)
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_A = A_candidate
                continue
            
            # Convert to heights and evaluate true metric
            A_candidate = _phases_to_heights(phases_fine)
            val_candidate = _metric_offdiag(A_candidate)
            
            if val_candidate < best_val:
                best_val = val_candidate
                best_A = A_candidate.copy()
                
        except Exception as e:
            continue
    
    # If we have time left, do a few more random restarts with shorter budgets
    restart_count = 0
    while not time_up() and restart_count < 5:
        phases0 = np.random.uniform(0, 2 * np.pi, size=N)
        try:
            phases_opt, _ = _gd_phase(phases0, max_iters=1500, lr=0.005, temp=0.01, deadline=deadline)
            A_candidate = _phases_to_heights(phases_opt)
            val_candidate = _metric_offdiag(A_candidate)
            if val_candidate < best_val:
                best_val = val_candidate
                best_A = A_candidate.copy()
        except Exception:
            break
        restart_count += 1
    
    return best_A
# EVOLVE-BLOCK-END
