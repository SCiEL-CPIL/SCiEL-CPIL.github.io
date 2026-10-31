"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def construct():
    """
    Construct step heights h in [0,1] with sum = N/2 that minimize
    the maximum overlap of h with shifted (1-h).
    
    The overlap at shift k is sum_i h[i] * (1-h[(i+k) % N]).
    We minimize the maximum over all shifts k.
    
    Strategy: Use SLSQP with log-sum-exp smoothing of the max overlap,
    with explicit deadline guard.
    """
    deadline = time.time() + 120
    
    # Initial guess: alternating 0 and 1 pattern (satisfies sum = N/2, in [0,1])
    x0 = np.zeros(N)
    x0[::2] = 1.0
    # Ensure sum is exactly N/2
    x0 = x0.copy()
    
    def max_overlap_shifts(h):
        """Compute overlap at each shift: sum_i h[i] * (1 - h[(i+k)%N])"""
        # 1-h
        one_minus_h = 1.0 - h
        # For each shift k, compute sum_i h[i] * one_minus_h[(i+k) % N]
        # This is a circular cross-correlation-like computation
        # overlap[k] = sum_i h[i] * (1 - h[(i+k) % N])
        #             = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
        #             = sum(h) - circular_corr(h, h)[k]
        
        # Circular correlation: corr[k] = sum_i h[i] * h[(i+k) % N]
        # Using FFT for efficiency
        H = np.fft.fft(h)
        corr = np.fft.ifft(H * np.conj(H)).real
        # corr[k] = sum_i h[i] * h[(i - k) % N]  (depending on convention)
        # Let's be careful: np.fft.ifft(fft(a) * conj(fft(b))) gives circular cross-correlation
        # corr[k] = sum_i a[i] * b[(i-k) % N]
        # So corr[k] = sum_i h[i] * h[(i-k) % N]
        # We want sum_i h[i] * h[(i+k) % N] = corr[-k % N] = corr[N-k] for k>0
        # Actually let's just compute it directly for clarity, or use the right indexing
        
        # sum_i h[i] * h[(i+k) % N] for k = 0, 1, ..., N-1
        # This is the circular autocorrelation with positive lag k
        # corr_pos[k] = sum_i h[i] * h[(i+k) % N]
        # From FFT: ifft(|H|^2)[m] = sum_i h[i] * h[(i-m) % N]
        # So corr_pos[k] = ifft(|H|^2)[-k % N] = ifft(|H|^2)[(N-k) % N]
        
        corr_neg = np.fft.ifft(np.abs(H)**2).real  # corr_neg[m] = sum_i h[i]*h[(i-m)%N]
        # corr_pos[k] = corr_neg[-k % N] = corr_neg[(N-k) % N]
        corr_pos = corr_neg[np.arange(N)[::-1] % N]  # This might be off, let's verify
        
        # Actually simpler: corr_pos[k] = sum_i h[i] * h[(i+k)%N]
        # Let's just use np.roll for verification in a small case, but for speed use FFT
        # corr_neg[m] = sum_i h[i] * h[(i-m) % N]
        # Set j = i-m, then i = j+m: corr_neg[m] = sum_j h[j+m] * h[j]
        # corr_pos[k] = sum_i h[i] * h[(i+k)%N], let j = i+k, i = j-k:
        # corr_pos[k] = sum_j h[j-k] * h[j] = corr_neg[k]  (by symmetry of autocorrelation)
        # Wait: corr_neg[k] = sum_i h[i]*h[(i-k)%N]
        # corr_pos[k] = sum_i h[i]*h[(i+k)%N]
        # For real signals, autocorrelation is even: corr_neg[k] = corr_neg[-k] = corr_pos[k]
        # So corr_pos[k] = corr_neg[k] for real h (autocorrelation is symmetric)
        
        # Actually for autocorrelation of a real signal, R[k] = R[-k], so:
        # sum_i h[i]*h[(i+k)%N] = sum_i h[i]*h[(i-k)%N]
        # Therefore corr_pos = corr_neg
        
        sum_h = np.sum(h)
        overlaps = sum_h - corr_neg  # overlap[k] = sum(h) - corr[k]
        
        return overlaps
    
    def objective_and_grad(x):
        """Objective: log-sum-exp of overlaps (smooth max), plus gradient."""
        h = x
        overlaps = max_overlap_shifts(h)
        
        # Log-sum-exp for smooth max
        beta = 10.0  # Temperature for log-sum-exp
        m = np.max(overlaps)
        lse = m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta
        
        # Gradient of lse w.r.t. h
        # d(lse)/d(overlaps_k) = exp(beta*(overlaps_k - m)) / sum(exp(beta*(overlaps_j - m)))
        weights = np.exp(beta * (overlaps - m))
        weights /= np.sum(weights)
        
        # Gradient of overlaps w.r.t. h
        # overlaps[k] = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
        # d(overlaps[k])/d(h[j]) = 1 - h[(j-k)%N] - h[(j+k)%N]
        # (from the two terms where j appears: h[j] in first sum, and in product)
        
        # Let's compute grad_overlaps[k, j] = d(overlaps[k])/d(h[j])
        # = 1 - h[(j-k) % N] - h[(j+k) % N]
        
        # grad_lse[j] = sum_k weights[k] * (1 - h[(j-k)%N] - h[(j+k)%N])
        
        # For each j: sum_k weights[k] * h[(j-k)%N] = circular_conv(weights, h)[j]
        # sum_k weights[k] * h[(j+k)%N] = circular_conv(weights, h)[j] (by symmetry)
        
        # Actually: sum_k w[k] * h[(j-k)%N] is the circular convolution (w * h)[j]
        # And sum_k w[k] * h[(j+k)%N] = sum_k w[k] * h[(-k+j)%N]... 
        # Let m = -k, then = sum_m w[-m] * h[(j+m)%N] = (w_rev * h)[j] where w_rev[m] = w[-m]
        
        W = np.fft.fft(weights)
        H = np.fft.fft(h)
        conv1 = np.fft.ifft(W * H).real  # conv1[j] = sum_k w[k]*h[(j-k)%N]
        
        # For the second term: sum_k w[k]*h[(j+k)%N]
        # = sum_k w[k]*h[(-(-k)+j)%N]
        # Let l = -k mod N, k = -l: sum_l w[-l]*h[(j-l)%N]
        # = (w_neg * h)[j] where w_neg[l] = w[-l % N]
        w_neg = weights[::-1].copy()
        w_neg[0] = weights[0]  # w_neg[0] = w[0]
        W_neg = np.fft.fft(w_neg)
        conv2 = np.fft.ifft(W_neg * H).real  # conv2[j] = sum_l w_neg[l]*h[(j-l)%N] = sum_k w[k]*h[(j+k)%N]
        
        grad_lse = np.sum(weights) - conv1 - conv2  # = 1 - conv1 - conv2 since sum(weights)=1
        
        # Add small regularization to keep in [0,1]
        reg = 1e-4
        reg_grad = reg * (2*h - 1)  # gradient of reg*(h-0.5)^2 summed... actually let's just use barrier
        
        return lse, grad_lse
    
    def objective(x):
        val, _ = objective_and_grad(x)
        return val
    
    def gradient(x):
        _, grad = objective_and_grad(x)
        return grad
    
    # Constraints: sum(h) = N/2, 0 <= h <= 1
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2, 'jac': lambda x: np.ones(N)}
    ]
    bounds = [(0.0, 1.0)] * N
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Try SLSQP
    result = minimize(
        objective, x0, jac=gradient,
        method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-8, 'disp': False}
    )
    
    if time.time() > deadline - 10:
        return best_x
    
    if result.success or result.fun < best_val:
        best_x = result.x.copy()
        best_val = result.fun
    
    # Project to feasible set: clip to [0,1] and adjust sum
    best_x = np.clip(best_x, 0.0, 1.0)
    # Adjust to make sum exactly N/2
    diff = N/2 - np.sum(best_x)
    if abs(diff) > 1e-10:
        # Distribute difference evenly, respecting bounds
        if diff > 0:
            # Need to increase sum
            room = 1.0 - best_x
            total_room = np.sum(room)
            if total_room > 0:
                best_x += diff * room / total_room
        else:
            # Need to decrease sum
            room = best_x
            total_room = np.sum(room)
            if total_room > 0:
                best_x += diff * room / total_room
    
    best_x = np.clip(best_x, 0.0, 1.0)
    
    # Final adjustment: ensure sum is exactly N/2 by adjusting the element with most room
    diff = N/2 - np.sum(best_x)
    if abs(diff) > 1e-12:
        if diff > 0:
            idx = np.argmax(1.0 - best_x)
            add = min(diff, 1.0 - best_x[idx])
            best_x[idx] += add
            remaining = diff - add
            if abs(remaining) > 1e-12:
                for i in range(N):
                    if abs(remaining) < 1e-12:
                        break
                    room = 1.0 - best_x[i]
                    if room > 0:
                        adj = min(remaining, room)
                        best_x[i] += adj
                        remaining -= adj
        else:
            idx = np.argmin(best_x)
            sub = min(-diff, best_x[idx])
            best_x[idx] -= sub
            remaining = -diff - sub
            if abs(remaining) > 1e-12:
                for i in range(N):
                    if abs(remaining) < 1e-12:
                        break
                    room = best_x[i]
                    if room > 0:
                        adj = min(remaining, room)
                        best_x[i] -= adj
                        remaining -= adj
    
    # Verify sum
    best_x = np.clip(best_x, 0.0, 1.0)
    
    return best_x

# EVOLVE-BLOCK-END
