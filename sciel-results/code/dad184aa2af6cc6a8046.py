"""Reformulated spectral-design approach: flat-spectrum via magnitude-phase decomposition."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2


def _autoconv_max_abs(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))


def _integral(x):
    return float(np.sum(x))


def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _autoconv_values(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]


def _soft_max_abs(conv, beta=1000.0):
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta


def _smooth_metric(x, beta=500.0):
    conv = _autoconv_values(x)
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _soft_max_abs(conv, beta) / (s * s)


def _spectral_design_optimize(x_init, deadline, max_iters=5000):
    """
    Optimize in the frequency domain: fix magnitudes of FFT to be flat,
    optimize phases to minimize off-zero-lag autoconvolution peaks.
    """
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    # Initial FFT
    fx = np.fft.rfft(best_x, _FFT_SIZE)
    
    # Target: flat magnitude spectrum
    # |F[k]| should be constant for all k
    # We optimize phases only, keeping magnitudes fixed at their initial values
    mags = np.abs(fx).copy()
    # Avoid zero magnitudes
    mags[mags < 1e-10] = 1e-10
    
    # Normalize so that integral (sum) is preserved
    # sum(x) = F[0], so we fix F[0] = sum(x)
    target_sum = _integral(best_x)
    mags[0] = abs(target_sum)
    phase0 = np.angle(fx[0]) if target_sum != 0 else 0.0
    
    # Use L-BFGS to optimize phases
    def objective_and_grad(phase_vec):
        # Reconstruct FFT from fixed magnitudes and optimized phases
        fx_new = mags.copy()
        fx_new[0] = target_sum  # Fix DC component to preserve integral
        
        # Apply phases
        for k in range(1, len(fx)):
            fx_new[k] = mags[k] * np.exp(1j * phase_vec[k - 1])
        
        # Inverse FFT to get time domain
        x_new = np.fft.irfft(fx_new, _FFT_SIZE)[:N]
        
        # Compute metric
        conv = _autoconv_values(x_new)
        s = _integral(x_new)
        if abs(s) < 1e-12:
            return 1e10, np.zeros_like(phase_vec)
        
        # Use smooth surrogate for gradient
        beta = 200.0
        abs_conv = np.abs(conv)
        max_val = np.max(abs_conv)
        shifted = abs_conv - max_val
        soft_max = max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
        
        val = soft_max / (s * s)
        
        # Gradient computation (approximate via finite differences for simplicity in spectral domain)
        # For better performance, use automatic differentiation or analytical gradient
        # Here we use a simplified gradient estimate
        
        eps = 1e-6
        grad = np.zeros(len(phase_vec))
        for j in range(len(phase_vec)):
            phase_vec[j] += eps
            fx_temp = mags.copy()
            fx_temp[0] = target_sum
            for k in range(1, len(fx)):
                fx_temp[k] = mags[k] * np.exp(1j * phase_vec[k - 1])
            x_temp = np.fft.irfft(fx_temp, _FFT_SIZE)[:N]
            conv_temp = _autoconv_values(x_temp)
            s_temp = _integral(x_temp)
            if abs(s_temp) < 1e-12:
                grad[j] = 0.0
            else:
                val_temp = _soft_max_abs(conv_temp, beta) / (s_temp * s_temp)
            phase_vec[j] -= eps
            grad[j] = (val_temp - val) / eps
        
        return val, grad
    
    # Initialize phases from current solution
    phases_init = np.zeros(len(fx) - 1)
    for k in range(1, len(fx)):
        if mags[k] > 1e-10:
            phases_init[k - 1] = np.angle(fx[k])
    
    # Optimize
    try:
        result = minimize(
            objective_and_grad,
            phases_init,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success:
            # Reconstruct from optimized phases
            fx_final = mags.copy()
            fx_final[0] = target_sum
            for k in range(1, len(fx)):
                fx_final[k] = mags[k] * np.exp(1j * result.x[k - 1])
            
            x_final = np.fft.irfft(fx_final, _FFT_SIZE)[:N]
            val_final = _metric(x_final)
            
            if val_final < best_val:
                best_val = val_final
                best_x = x_final.copy()
    except Exception:
        pass
    
    return best_x, best_val


def _try_gradient_descent_smooth(x_init, deadline, rng, max_iters=20000):
    """Gradient descent using smooth log-sum-exp surrogate."""
    best_x = x_init.copy()
    best_val = _metric(best_x)
    current_x = x_init.copy()
    
    lr = 0.005
    patience = 0
    best_smooth_val = _smooth_metric(current_x, beta=300.0)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        conv = _autoconv_values(current_x)
        abs_conv = np.abs(conv)
        s = _integral(current_x)
        
        if abs(s) < 1e-12:
            break
        
        beta = 300.0
        shifted = abs_conv - np.max(abs_conv)
        weights = np.exp(beta * shifted)
        weights /= np.sum(weights)
        
        grad = np.zeros(N)
        for k in range(2 * N - 1):
            if weights[k] < 1e-10:
                continue
            sign_k = np.sign(conv[k]) if conv[k] != 0 else 1.0
            for j in range(N):
                contrib = 0.0
                if j + k < N:
                    contrib += current_x[j + k]
                if j - k >= 0:
                    contrib += current_x[j - k]
                grad[j] += 2.0 * sign_k * weights[k] * contrib
        
        soft_max_val = _soft_max_abs(conv, beta)
        grad *= -2.0 * soft_max_val / (s ** 3)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-15:
            grad /= grad_norm
        
        x_new = current_x - lr * grad
        v_new = _metric(x_new)
        if v_new < best_val:
            best_val = v_new
            best_x = x_new.copy()
            current_x = x_new
            patience = 0
            lr = min(lr * 1.05, 0.05)
        else:
            patience += 1
            if patience > 30:
                lr *= 0.7
                patience = 0
                if lr < 1e-7:
                    break
    
    return best_x, best_val


def _try_local_search_fast(x, deadline, rng, max_rounds=100000):
    """Fast local search with in-place modifications."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    step_scale = 0.2
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 150:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
                break
        
        k = int(rng.integers(1, 5))
        indices = rng.choice(N, size=k, replace=False)
        
        best_delta = 0
        best_idx = None
        for trial in range(2):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_idx = x_test
        
        if best_idx is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_idx
            no_improve = 0
        else:
            no_improve += 1
    
    return best_x, best_val


def _final_polish(x, deadline, rng):
    """High-precision final polish with coordinate descent."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    for sweep in range(30):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            lo, hi = -0.05, 0.05
            best_delta = 0.0
            best_v = best_val
            
            for _ in range(10):
                if time.time() > deadline:
                    break
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                
                x1 = best_x.copy()
                x1[idx] += m1
                v1 = _metric(x1)
                
                x2 = best_x.copy()
                x2[idx] += m2
                v2 = _metric(x2)
                
                if v1 < v2:
                    hi = m2
                else:
                    lo = m1
                
                if v1 < best_v:
                    best_v = v1
                    best_delta = m1
                if v2 < best_v:
                    best_v = v2
                    best_delta = m2
            
            if best_delta != 0.0 and best_v < best_val:
                best_x[idx] += best_delta
                best_val = best_v
                improved = True
        
        if not improved:
            break
    
    return best_x, best_val


def construct():
    deadline = time.time() + 145.0
    
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    rng = np.random.default_rng(2024)
    
    # Phase 1: Spectral design optimization (main innovation)
    if time.time() < deadline - 80:
        x_spec, val_spec = _spectral_design_optimize(best_x, deadline - 80, max_iters=3000)
        if val_spec < best_val:
            best_val = val_spec
            best_x = x_spec
    
    # Phase 2: Gradient descent with smooth surrogate
    if time.time() < deadline - 40:
        x_grad, val_grad = _try_gradient_descent_smooth(best_x, deadline - 40, rng, max_iters=10000)
        if val_grad < best_val:
            best_val = val_grad
            best_x = x_grad
    
    # Phase 3: Local search
    if time.time() < deadline - 15:
        best_x, best_val = _try_local_search_fast(best_x, deadline - 15, rng, max_rounds=50000)
    
    # Phase 4: Final polish
    if time.time() < deadline - 2:
        best_x, best_val = _final_polish(best_x, deadline - 2, rng)
    
    return best_x
# EVOLVE-BLOCK-END
