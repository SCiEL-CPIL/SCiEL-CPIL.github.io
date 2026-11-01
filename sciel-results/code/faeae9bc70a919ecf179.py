"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 400
_NFFT = 1 << (2 * N - 1).bit_length()


def _c3_value(h):
    """Compute C3 bound: sum of |sidelobe|^3 normalized."""
    F = np.fft.rfft(h, n=_NFFT)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=_NFFT).real[:N]
    sidelobes = acf[1:]
    return np.sum(np.abs(sidelobes) ** 3) / (N ** 3)


def _smooth_c3(x, beta=10.0):
    """
    Smooth surrogate for C3 on continuous variables x in R^N.
    Uses soft absolute value: |s|_beta = sqrt(s^2 + 1/beta^2)
    Objective: sum(|s_i|_beta^3) / N^3
    """
    F = np.fft.rfft(x, n=_NFFT)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=_NFFT).real[:N]
    sidelobes = acf[1:]
    abs_s = np.sqrt(sidelobes ** 2 + 1.0 / (beta * beta))
    cubes = abs_s ** 3
    return np.sum(cubes) / (N ** 3)


def _grad_smooth_c3(x, beta=10.0):
    """
    Gradient of smooth C3 with respect to x.
    
    d/dx_i [sum_j (sqrt(s_j^2 + eps))^3] where eps = 1/beta^2
    = sum_j 3*(sqrt(s_j^2+eps))^2 * (1/(2*sqrt(s_j^2+eps))) * 2*s_j * ds_j/dx_i
    = sum_j 3*(sqrt(s_j^2+eps))^2 * s_j / sqrt(s_j^2+eps) * ds_j/dx_i
    = sum_j 3*sqrt(s_j^2+eps) * s_j * ds_j/dx_i
    
    where s_j = acf[j] - acf[0] (sidelobes, j=1..N-1)
    and ds_j/dx_i = d(acf[j])/dx_i - d(acf[0])/dx_i
    
    acf = ifft(|FFT(x)|^2)
    d(acf[k])/dx_i = 2 * Re{ ifft( FFT(x) * conj(FFT(x)) ... ) }
    
    Actually, let's use the property:
    acf = ifft( |F|^2 )
    d(acf[k])/dx_i = 2 * Re{ ifft( F * conj(dF_k/dx_i) ) } ... no
    
    Let's use: acf[k] = sum_m F[m] * conj(F[m-k]) (circular)
    For linear autocorrelation via zero-padding, it's similar.
    
    Actually, a cleaner approach:
    Let G = ifft(|F|^2) = acf (circular autocorrelation of length _NFFT)
    d(G[k])/dx_i = 2 * Re{ ifft( F * conj( dF/dx_i ) ) } ... this is getting complex.
    
    Let's use numerical gradient for simplicity and correctness, but vectorize the FFT part.
    Actually, for N=400, numerical gradient with central differences requires 2*N FFTs per evaluation.
    That's 800 FFTs of size 1024. Each FFT is ~1024*log2(1024) ~ 10k ops.
    800 * 10k = 8M ops per gradient. With 100 iterations, that's 800M ops. Might be tight.
    
    Let's try an analytical gradient.
    
    acf = ifft( F * conj(F) )
    Let H = F * conj(F). Then acf = ifft(H).
    d(acf[k])/dx_i = ifft( dH/dx_i )[k]
    dH[m]/dx_i = dF[m]/dx_i * conj(F[m]) + F[m] * d(conj(F[m]))/dx_i
    dF[m]/dx_i = (1/_NFFT) * exp(-2*pi*i*m*i/_NFFT)  (from FFT definition)
    Wait, F[m] = sum_n x[n] * exp(-2*pi*i*m*n/_NFFT)
    dF[m]/dx_i = exp(-2*pi*i*m*i/_NFFT)
    d(conj(F[m]))/dx_i = conj(exp(-2*pi*i*m*i/_NFFT)) = exp(2*pi*i*m*i/_NFFT)
    
    So dH[m]/dx_i = exp(-2*pi*i*m*i/_NFFT) * conj(F[m]) + F[m] * exp(2*pi*i*m*i/_NFFT)
    = conj(F[m]) * exp(-2*pi*i*m*i/_NFFT) + F[m] * exp(2*pi*i*m*i/_NFFT)
    
    Then d(acf[k])/dx_i = (1/_NFFT) * sum_m dH[m]/dx_i * exp(2*pi*i*m*k/_NFFT)
    = (1/_NFFT) * sum_m [ conj(F[m]) * exp(-2*pi*i*m*i/_NFFT) + F[m] * exp(2*pi*i*m*i/_NFFT) ] * exp(2*pi*i*m*k/_NFFT)
    = (1/_NFFT) * [ sum_m conj(F[m]) * exp(2*pi*i*m*(k-i)/_NFFT) + sum_m F[m] * exp(2*pi*i*m*(k+i)/_NFFT) ]
    = ifft( conj(F) )[k-i] + ifft( F )[k+i]  (with appropriate indexing)
    
    Let's define:
    x = ifft(F)  (which is just the original x, zero-padded)
    x_conj_fft = ifft(conj(F)) = conj(x)  (for real x, ifft(conj(F)) = conj(ifft(F)) = conj(x))
    
    So d(acf[k])/dx_i = x_conj_fft[k-i] + x[k+i]  (circular indexing)
    = conj(x)[k-i] + x[k+i]
    
    Since x is real, conj(x) = x.
    So d(acf[k])/dx_i = x[k-i] + x[k+i]
    
    This is the derivative of circular autocorrelation. For linear autocorrelation with zero-padding,
    we need to be careful, but for k < N and i < N, and _NFFT >= 2N-1, the circular and linear
    autocorrelations agree for lags 0..N-1.
    
    So ds_j/dx_i = d(acf[j])/dx_i - d(acf[0])/dx_i = (x[j-i] + x[j+i]) - (x[0-i] + x[0+i])
    = x[j-i] + x[j+i] - x[-i] - x[i]  (circular indexing on zero-padded x)
    
    Let's implement this.
    """
    eps = 1.0 / (beta * beta)
    F = np.fft.rfft(x, n=_NFFT)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=_NFFT).real[:N]
    sidelobes = acf[1:]
    abs_s = np.sqrt(sidelobes ** 2 + eps)
    
    # d(cubes)/d(sidelobes[j]) = 3 * abs_s[j] * sidelobes[j]
    d_obj_ds = 3.0 * abs_s * sidelobes
    
    # ds_j/dx_i = acf_deriv[j] - acf_deriv[0]
    # acf_deriv[k] = x_padded[k-i] + x_padded[k+i] for each i
    # We need sum_j d_obj_ds[j] * (acf_deriv[j] - acf_deriv[0])
    # = sum_j d_obj_ds[j] * acf_deriv[j] - (sum_j d_obj_ds[j]) * acf_deriv[0]
    
    # Let's compute this efficiently.
    # acf_deriv[k] for fixed i is x_padded[k-i] + x_padded[k+i]
    # sum_j d_obj_ds[j] * acf_deriv[j] = sum_j d_obj_ds[j] * (x_padded[j-i] + x_padded[j+i])
    # = sum_j d_obj_ds[j] * x_padded[j-i] + sum_j d_obj_ds[j] * x_padded[j+i]
    
    # Let w = d_obj_ds (length N-1, indices 1..N-1)
    # sum_{j=1}^{N-1} w[j] * x_padded[j-i]
    # This is a cross-correlation of w with x_padded.
    
    # For simplicity and correctness, let's use numerical gradient.
    # With N=400, 800 FFTs of size 1024 should be manageable within the time budget.
    
    grad = np.zeros(N)
    eps_num = 1e-6
    x_plus = x.copy()
    x_minus = x.copy()
    
    # Vectorize: perturb each coordinate separately
    # This is O(N) FFTs, which is 400 FFTs for forward and 400 for backward = 800 total.
    # Each FFT of size 1024 is fast.
    
    for i in range(N):
        x_plus[i] += eps_num
        x_minus[i] -= eps_num
        
        val_plus = _smooth_c3(x_plus, beta)
        val_minus = _smooth_c3(x_minus, beta)
        
        grad[i] = (val_plus - val_minus) / (2 * eps_num)
        
        x_plus[i] -= eps_num
        x_minus[i] += eps_num
    
    return grad


def _local_search_polish(h, deadline, max_rounds=50):
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


def _simulated_annealing(h, deadline):
    """Simulated annealing to minimize C3."""
    rng = np.random.default_rng(123)
    best_h = h.copy()
    best_val = _c3_value(best_h)
    current_h = h.copy()
    current_val = best_val
    
    T = 5.0
    T_min = 0.001
    cooling = 0.998
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
        
        # Generate candidate by flipping 1-3 random positions
        n_flips = rng.integers(1, 4)
        candidate = current_h.copy()
        indices = rng.choice(N, size=n_flips, replace=False)
        candidate[indices] = -candidate[indices]
        
        new_val = _c3_value(candidate)
        
        if new_val < current_val:
            current_h = candidate
            current_val = new_val
            if new_val < best_val:
                best_h = candidate.copy()
                best_val = new_val
        else:
            delta = new_val - current_val
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_h = candidate
                current_val = new_val
        
        T *= cooling
    
    return best_h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous relaxation via L-BFGS-B
    # Multiple random starts in continuous space, optimize, then threshold
    n_continuous_starts = 3
    per_start_time = 30.0  # 30s per continuous optimization
    
    for s in range(n_continuous_starts):
        phase_deadline = start_time + 10.0 + per_start_time * (s + 1)
        if time.time() >= phase_deadline - 5.0:
            break
        
        # Initialize with random continuous values in [-1, 1]
        x_init = rng.uniform(-1.0, 1.0, size=N)
        
        # Optimize continuous relaxation
        try:
            result = minimize(
                _smooth_c3,
                x_init,
                method='L-BFGS-B',
                jac=_grad_smooth_c3,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 200},
                args=(10.0,)  # beta
            )
        except Exception:
            continue
        
        x_opt = result.x
        
        # Threshold to binary
        h_bin = np.sign(x_opt)
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = rng.choice([-1.0, 1.0], size=np.sum(zero_mask))
        
        h_bin_val = _c3_value(h_bin)
        
        if h_bin_val < best_val:
            best_h = h_bin.copy()
            best_val = h_bin_val
    
    # Phase 2: Discrete SA from the best continuous result
    if best_h is not None and time.time() < deadline:
        sa_deadline = min(deadline - 10.0, start_time + 80.0)
        sa_h, sa_val = _simulated_annealing(best_h, sa_deadline)
        if sa_val < best_val:
            best_h = sa_h
            best_val = sa_val
    
    # Phase 3: Local search polish
    if best_h is not None and time.time() < deadline:
        polished_h, polished_val = _local_search_polish(best_h, deadline)
        if polished_val < best_val:
            best_h = polished_h
            best_val = polished_val
    
    if best_h is None:
        # Fallback: return random
        best_h = rng.choice([-1.0, 1.0], size=N)
    
    return best_h
# EVOLVE-BLOCK-END
