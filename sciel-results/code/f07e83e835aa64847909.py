import numpy as np
import time
from scipy.optimize import linprog, minimize

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 20.0

def compute_value(h):
    """Exact evaluation of the objective function."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return float('inf')
    if not np.all(np.isfinite(h)):
        return float('inf')
    z = np.convolve(h, h)
    return abs(2 * N * np.max(z) / (s * s))

def solve_sdp_relaxation():
    """
    Solves the LP relaxation of the peak autocorrelation minimization.
    Variables: P[0..N-1] (power spectrum), M (peak bound).
    Constraints:
    1. P[i] >= 0
    2. P[0] = 1 (normalization sum(h)=1)
    3. P[i] = P[N-i] (symmetry for real h)
    4. Circular Autocorrelation r[k] <= M for all k.
       r[k] = (1/N) * sum_m P[m] * exp(2j*pi*k*m/N)
       We take the real part.
    """
    # Variables: x = [P[0], P[1], ..., P[N-1], M]
    # Total variables: N + 1
    
    # Objective: Minimize M
    # c = [0, ..., 0, 1]
    c = np.zeros(N + 1)
    c[-1] = 1.0
    
    # Inequality constraints: A_ub @ x <= b_ub
    # 1. r[k] <= M  =>  r[k] - M <= 0
    # r[k] = (1/N) * sum_m P[m] * cos(2*pi*k*m/N)  (since P is symmetric, sin terms cancel)
    # So: (1/N) * sum_m P[m] * cos(2*pi*k*m/N) - M <= 0
    
    # Number of k values: 0 to N-1 (or just 0 to N/2 due to symmetry of r[k])
    # r[k] = r[N-k]. So we only need constraints for k=0..N//2.
    num_k = N // 2 + 1
    
    A_ub = np.zeros((num_k, N + 1))
    b_ub = np.zeros(num_k)
    
    for k in range(num_k):
        # Coefficients for P[m]
        for m in range(N):
            A_ub[k, m] = (1.0 / N) * np.cos(2.0 * np.pi * k * m / N)
        # Coefficient for M
        A_ub[k, -1] = -1.0
        
    # Equality constraints: A_eq @ x = b_eq
    # 1. P[0] = 1
    # 2. P[i] = P[N-i] for i=1..N//2-1
    #    => P[i] - P[N-i] = 0
    
    num_eq = 1 + (N // 2 - 1)
    A_eq = np.zeros((num_eq, N + 1))
    b_eq = np.zeros(num_eq)
    
    # P[0] = 1
    A_eq[0, 0] = 1.0
    b_eq[0] = 1.0
    
    idx = 1
    for i in range(1, N // 2):
        A_eq[idx, i] = 1.0
        A_eq[idx, N - i] = -1.0
        b_eq[idx] = 0.0
        idx += 1
        
    # Bounds: P[i] >= 0, M is free (but effectively bounded by constraints)
    # M can be negative? No, r[0] = sum(P)/N = (1+sum(P[1..]))/N >= 0. So M >= 0.
    bounds = [(0, None)] * N + [(None, None)]
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.status != 0:
        return None, None
        
    P = res.x[:N]
    M_opt = res.x[-1]
    
    # Construct h from P
    # H[m] = sqrt(P[m])
    # We choose phase 0 for all, so h is real and even.
    H = np.sqrt(np.maximum(P, 0))
    h = np.real(np.fft.ifft(H))
    
    # Normalize h such that sum(h) = 1
    s = np.sum(h)
    if abs(s) > 1e-12:
        h = h / s
    else:
        return None, None
        
    return h, M_opt

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - SAFETY_MARGIN)
    
    best_h = None
    best_val = float('inf')
    
    # 1. Solve the SDP/LP Relaxation
    h_sdp, M_sdp = solve_sdp_relaxation()
    if h_sdp is not None:
        val_sdp = compute_value(h_sdp)
        if val_sdp < best_val:
            best_val = val_sdp
            best_h = h_sdp.copy()
            
    # 2. If SDP fails or gives poor result, fallback to simple heuristic
    if best_h is None or best_val > 2.0:
        # Seed
        h_seed = np.ones(N)
        val_seed = compute_value(h_seed)
        if val_seed < best_val:
            best_val = val_seed
            best_h = h_seed.copy()
            
    # 3. Local Refinement of the SDP solution
    # The SDP solution minimizes circular autocorrelation peak.
    # We can refine it to minimize linear autocorrelation peak using gradient descent.
    if best_h is not None and time.time() < deadline:
        
        def smooth_objective(h):
            if not np.all(np.isfinite(h)):
                return float('inf')
            s = np.sum(h)
            if abs(s) < 1e-9:
                return float('inf')
            # Use FFT for linear convolution
            fft_len = 1024
            H_lin = np.fft.fft(h, n=fft_len)
            Z_lin = np.fft.ifft(H_lin * H_lin)
            z_lin = Z_lin.real[:2*N-1]
            
            # Smooth max
            tau = 1.0
            z_scaled = z_lin / tau
            z_max_scaled = np.max(z_scaled)
            exp_terms = np.exp(z_scaled - z_max_scaled)
            sum_exp = np.sum(exp_terms)
            if sum_exp == 0:
                log_sum_exp = z_max_scaled
            else:
                log_sum_exp = z_max_scaled + np.log(sum_exp)
            max_z_smooth = log_sum_exp * tau
            
            val = abs(2 * N * max_z_smooth / (s * s))
            return val

        try:
            res = minimize(
                smooth_objective, 
                best_h, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            if res.fun < best_val:
                h_check = res.x
                if np.all(np.isfinite(h_check)) and abs(np.sum(h_check)) > 1e-9:
                    val_check = compute_value(h_check)
                    if val_check < best_val:
                        best_val = val_check
                        best_h = h_check.copy()
        except Exception:
            pass

    # Ensure validity
    if best_h is None or not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    h = construct()
    print(len(h))
    print(np.sum(h))
    z = np.convolve(h, h)
    print(abs(2 * N * np.max(z) / np.sum(h)**2))
