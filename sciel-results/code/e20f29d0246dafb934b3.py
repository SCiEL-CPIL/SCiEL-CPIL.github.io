import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.fft import fft, ifft, rfft, irfft

N = 600
TIME_LIMIT = 145  # Safety margin for 150s kill

def compute_score(f):
    """
    Compute the normalized maximum of the autoconvolution.
    f: array of shape (N,)
    """
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-12):
        # Handle small numerical negatives by clipping, but if significant, it's invalid
        # For scoring purposes in optimization, we assume f is non-negative.
        # The evaluator likely clips or checks. Let's be strict here for internal tracking.
        if np.any(f < -1e-8):
            return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    
    # Autoconvolution via FFT
    # We need linear convolution, so pad to 2N-1
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    conv_freq = F * F
    conv_time = np.fft.irfft(conv_freq, n=n_fft)
    
    # The convolution of two length N signals has length 2N-1.
    # irfft returns length n_fft. The relevant part is indices 0 to N-1?
    # Actually, np.convolve(f, f, 'full') has length 2N-1.
    # The peak can be anywhere.
    # Normalization: max(conv) / (sum(f)^2)
    
    # Note: The problem statement says "normalized maximum".
    # Usually this implies max((f*f)(t)) / (integral(f)^2).
    # Since we are on a grid, sum(f) approximates the integral.
    # The autoconvolution values are sums of products.
    
    # Let's stick to the definition in the seed: np.max(conv) / (s*s)
    # But wait, the seed used np.convolve. Let's use FFT for speed in objective/grad.
    # For the final score, we must be consistent.
    
    max_val = np.max(conv_time[:2*N-1]) # Just to be safe, though full is fine
    return max_val / (s * s)

def objective_and_grad(z):
    """
    z: log-parameters. f = exp(z).
    This ensures f > 0 strictly.
    We optimize in log-space to handle non-negativity naturally.
    However, f can be very small or large.
    Alternatively, use f = softplus(z) or just f = z with projection?
    L-BFGS-B handles bounds. Let's use f directly with bounds [0, inf].
    But the gradient of max(conv) is not smooth.
    
    Let's use the log-sum-exp approximation for the max to get a smooth objective.
    Objective: J(z) = log(sum(exp(beta * conv(z)))) / (sum(exp(z))^2) ? No.
    
    Let's define f = exp(z).
    S = sum(f)
    C = autoconv(f)
    We want to minimize max(C) / S^2.
    
    To make it smooth, let M = log-sum-exp(beta * C) / beta.
    J = M / S^2.
    
    Gradient of J with respect to z:
    f_i = exp(z_i)
    S = sum(f)
    C_k = sum_{i+j=k} f_i f_j
    
    dC/dz_i = f_i * (dC/d f_i)
    dC_k / d f_i = sum_{j: i+j=k} f_j = (f * f)_k (circular or linear?)
    Since we use FFT for linear convolution, the derivative is the cross-correlation.
    Specifically, if C = f * f (linear), then dC_k / df_i = sum_{j} f_j delta_{i+j, k} = f_{k-i} (with boundary checks).
    
    This is getting complicated to derive analytically for linear convolution with FFT padding.
    
    Alternative Strategy:
    Use a smooth approximation of the max:
    LSE(x, beta) = (1/beta) * log(sum(exp(beta * x)))
    
    Objective:
    f = exp(z)
    S = sum(f)
    C = linear_conv(f, f)
    M = LSE(C, beta)
    J = M / S^2
    
    Gradient:
    dJ/dz_i = (1/S^2) * dM/dz_i - (2 * M / S^3) * dS/dz_i
    dS/dz_i = f_i
    
    dM/dz_i = (1/beta) * sum_k (beta * exp(beta*C_k) / sum(exp(beta*C_k))) * dC_k/dz_i
             = sum_k P_k * dC_k/dz_i
    where P_k = exp(beta*C_k) / sum(exp(beta*C_k))
    
    dC_k/dz_i = f_i * (dC_k/df_i)
    
    So dM/dz_i = f_i * sum_k P_k * (dC_k/df_i)
    
    Note that sum_k P_k * (dC_k/df_i) is the gradient of the LSE wrt f_i.
    Let G_f = grad_LSE(C, beta). Shape (N,).
    Then dM/dz_i = f_i * G_f[i].
    
    So dJ/dz_i = f_i * G_f[i] / S^2 - 2 * M * f_i / S^3
               = (f_i / S^2) * (G_f[i] - 2*M/S)
    
    We need to compute G_f efficiently.
    C_k = sum_{i+j=k} f_i f_j.
    dC_k/df_i = sum_{j} f_j delta_{i+j, k} = f_{k-i} (if 0 <= k-i < N, else 0).
    This is effectively the correlation of f with itself, but weighted by P_k.
    
    G_f[i] = sum_k P_k * f_{k-i}
    This is a convolution of P with f!
    G_f = conv(P, f) (linear convolution, truncated to N).
    
    So:
    1. f = exp(z)
    2. S = sum(f)
    3. C = linear_conv(f, f)
    4. P = softmax(beta * C)
    5. M = dot(P, C)  # This is the LSE value * beta? No.
       LSE = (1/beta) log(sum exp(beta C)).
       Derivative of LSE wrt C is P.
       So dM/dC = P.
       M = LSE(C, beta).
       
    6. G_f = linear_conv(P, f)  # Length 2N-1, take first N?
       Wait. dC_k/df_i = f_{k-i}.
       G_f[i] = sum_k P_k f_{k-i}.
       Let j = k-i => k = i+j.
       G_f[i] = sum_j P_{i+j} f_j.
       This is the cross-correlation of P and f.
       Specifically, if we define P_ext of length 2N-1, and f of length N.
       G_f[i] = sum_{j=0}^{N-1} f_j P_{i+j}.
       
    7. J = M / S^2
    8. dJ/dz_i = f_i * (G_f[i] - 2*M/S) / S^2
    
    Let's implement this.
    """
    beta = 10.0 # Sharpness of LSE. Higher is better (closer to max).
    
    f = np.exp(z)
    S = np.sum(f)
    
    # Linear convolution f * f
    # Use FFT
    n_fft = 2 * N
    F = np.fft.rfft(f, n=n_fft)
    C_freq = F * F
    C_full = np.fft.irfft(C_freq, n=n_fft)
    C = C_full[:2*N-1] # Length 2N-1
    
    # LSE and Probabilities
    # Numerical stability
    C_max = np.max(C)
    exp_beta_C = np.exp(beta * (C - C_max))
    sum_exp = np.sum(exp_beta_C)
    P = exp_beta_C / sum_exp
    M = C_max + (1.0/beta) * np.log(sum_exp)
    
    # Gradient of M wrt f: G_f
    # G_f[i] = sum_j f_j P_{i+j}
    # This is the cross-correlation.
    # We can compute this via FFT as well.
    # corr = ifft( conj(rfft(f)) * rfft(P) )?
    # P has length 2N-1. f has length N.
    # Let's pad P to length 2N-1? No, P is already length 2N-1.
    # We want G_f of length N.
    
    # Let's use direct convolution for simplicity if N is small, but N=600 is small enough for O(N^2)?
    # 600^2 = 360,000. Very fast.
    # But we are inside an optimization loop. L-BFGS might call this 100-1000 times.
    # 360k * 1000 = 360M ops. Python loops are slow. Numpy vectorized is fast.
    # Let's use FFT for G_f too.
    
    # G_f[i] = sum_{j=0}^{N-1} f_j P_{i+j}
    # This is the 'valid' part of the correlation?
    # Let's construct an array Q of length 2N-1 where Q[k] = P[k].
    # We want the convolution of f and P?
    # (f * P)[i] = sum_j f_j P[i-j]. Not quite.
    # We have indices i+j.
    # Let's flip P. P_flip[k] = P[2N-2-k].
    # Then sum_j f_j P[i+j] = sum_j f_j P_flip[2N-2-(i+j)] = sum_j f_j P_flip[(2N-2-i)-j].
    # This is the convolution (f * P_flip)[2N-2-i].
    
    P_flip = P[::-1]
    n_fft2 = 2 * N # Need length at least (N-1) + (2N-1) = 3N-2?
    # Actually, conv of length N and 2N-1 has length 3N-2.
    # But we only need indices 0 to N-1 of G_f.
    # G_f[i] corresponds to index 2N-2-i in the full conv.
    # So we need the "middle" part of the convolution.
    
    # Let's just do the FFT conv of f and P_flip.
    n_fft_conv = 4 * N # Safe padding
    F_f = np.fft.rfft(f, n=n_fft_conv)
    F_Pf = np.fft.rfft(P_flip, n=n_fft_conv)
    conv_full = np.fft.irfft(F_f * F_Pf, n=n_fft_conv)
    
    # Extract G_f
    # Index in conv_full for G_f[i] is 2N-2-i.
    # i ranges 0..N-1.
    # Indices range 2N-2 down to N-1.
    G_f = conv_full[N-1:3*N-2][::-1] # Slice and reverse
    
    # Calculate J and Gradient
    J = M / (S * S)
    
    # dJ/dz_i = f_i * (G_f[i] - 2*M/S) / S^2
    term = G_f - (2.0 * M / S)
    grad_z = f * term / (S * S)
    
    return J, grad_z

def construct():
    start_time = time.time()
    
    # Initial guess: Uniform distribution
    # f = 1/N. z = log(1/N)
    z0 = np.full(N, np.log(1.0/N))
    
    # Try a few random initializations in log-space
    # But uniform is a strong baseline.
    # Maybe a Gaussian-like distribution?
    # Let's stick to uniform first, then maybe perturb.
    
    best_z = z0.copy()
    best_score = np.inf
    
    # Run L-BFGS-B
    # Bounds: z can be anything, but exp(z) should be reasonable.
    # If z is too large, exp(z) overflows. If too small, underflows.
    # Let's bound z to [-20, 20].
    bounds = [(-20.0, 20.0)] * N
    
    try:
        res = minimize(
            objective_and_grad,
            z0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={
                'maxiter': 1000,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50,
                'disp': False
            }
        )
        
        if res.fun < best_score:
            best_score = res.fun
            best_z = res.x
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to initial
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 5:
        # Not enough time for more
        f_best = np.exp(best_z)
        f_best = np.maximum(f_best, 0.0)
        s = np.sum(f_best)
        if s > 1e-15:
            f_best = f_best / s
        return [float(x) for x in f_best]
    
    # Try a second run with a different initialization: Gaussian
    # x in [-1/4, 1/4]
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25/N
    gauss = np.exp(-50 * x**2)
    gauss = gauss / np.sum(gauss)
    z1 = np.log(gauss + 1e-15)
    
    try:
        res2 = minimize(
            objective_and_grad,
            z1,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={
                'maxiter': 1000,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50,
                'disp': False
            }
        )
        
        if res2.fun < best_score:
            best_score = res2.fun
            best_z = res2.x
            
    except Exception as e:
        print(f"Second optimization failed: {e}")
        
    # Try a third run: Cosine bump
    # f = 0.5 * (1 + cos(2*pi*x)) ? No, support is [-1/4, 1/4].
    # Let's try a polynomial bump.
    # Just use the best found so far.
    
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 5:
        # Try one more random restart
        z2 = z0 + np.random.normal(0, 0.1, N)
        try:
            res3 = minimize(
                objective_and_grad,
                z2,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50,
                    'disp': False
                }
            )
            if res3.fun < best_score:
                best_score = res3.fun
                best_z = res3.x
        except Exception as e:
            pass

    # Convert back to f
    f_best = np.exp(best_z)
    f_best = np.maximum(f_best, 0.0)
    s = np.sum(f_best)
    if s > 1e-15:
        f_best = f_best / s
        
    return [float(x) for x in f_best]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
