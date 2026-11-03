"""Rewritten program for Erdos minimum overlap task."""
import numpy as np
import time
import sys

N = 100
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0  # seconds to leave before hard limit

def _compute_max_overlap(h):
    """Compute the maximum overlap C_k = sum_i h[i] * (1 - h[(i+k)%N]) for k=1..N-1.
    
    Note: The overlap of h with shifted (1-h) at shift k is:
    C_k = sum_i h[i] * (1 - h[(i+k)%N])
    = sum_i h[i] - sum_i h[i] * h[(i+k)%N]
    = N/2 - autocorr[k]
    
    where autocorr[k] = sum_i h[i] * h[(i+k)%N]
    
    So maximizing the minimum autocorrelation minimizes the maximum overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    # Compute circular autocorrelation
    # autocorr[k] = sum_i h[i] * h[(i+k)%n]
    # Use FFT for efficiency
    H = np.fft.rfft(h, n=2*n)
    # autocorr[k] = ifft(|H|^2)[k] for k=0..n-1, but need circular
    # Actually, circular autocorrelation:
    # C[k] = sum_{i=0}^{n-1} h[i] * h[(i+k) % n]
    # This is the circular cross-correlation of h with itself
    # Using FFT: ifft(fft(h) * conj(fft(h))) gives circular autocorrelation
    H = np.fft.fft(h)
    power = H * np.conj(H)
    autocorr = np.fft.ifft(power).real
    # autocorr[k] for k=0..n-1
    # autocorr[0] = sum h[i]^2
    # We want max over k=1..n-1 of (N/2 - autocorr[k])
    # = N/2 - min_{k=1..n-1} autocorr[k]
    min_ac = np.min(autocorr[1:])
    return n / 2.0 - min_ac

def _compute_all_overlaps(h):
    """Return array of overlaps C_k for k=1..N-1."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    H = np.fft.fft(h)
    power = H * np.conj(H)
    autocorr = np.fft.ifft(power).real
    overlaps = n / 2.0 - autocorr[1:]
    return overlaps

def _project_to_feasible(h):
    """Project h onto the feasible set: 0 <= h[i] <= 1, sum(h) = N/2.
    Uses iterative projection (POCS-like) with a few iterations.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    n = len(h)
    target_sum = n / 2.0
    
    for _ in range(50):
        # Clip to [0, 1]
        h = np.clip(h, 0.0, 1.0)
        # Adjust sum
        diff = target_sum - np.sum(h)
        if abs(diff) < 1e-15:
            break
        # Distribute difference proportionally, respecting bounds
        h += diff / n
        h = np.clip(h, 0.0, 1.0)
        # If clipping changed the sum, iterate again
        diff = target_sum - np.sum(h)
        if abs(diff) < 1e-12:
            break
    
    # Final correction: if sum is still off, adjust elements with slack
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-10:
        if diff > 0:
            # Need to increase; find elements < 1
            indices = np.where(h < 1.0 - 1e-12)[0]
            if len(indices) > 0:
                per = diff / len(indices)
                h[indices] = np.minimum(h[indices] + per, 1.0)
        else:
            # Need to decrease; find elements > 0
            indices = np.where(h > 1e-12)[0]
            if len(indices) > 0:
                per = diff / len(indices)
                h[indices] = np.maximum(h[indices] + per, 0.0)
    
    return h

def _soft_max(x, beta):
    """Soft-max approximation: (1/beta) * log(sum(exp(beta * x)))."""
    x = np.asarray(x, dtype=np.float64)
    # Avoid overflow
    max_x = np.max(x)
    if max_x > 500 / beta:
        max_x = 500 / beta
    return (1.0 / beta) * np.log(np.sum(np.exp(beta * (x - max_x)))) + max_x

def _soft_max_grad(x, beta):
    """Gradient of soft-max w.r.t. x."""
    x = np.asarray(x, dtype=np.float64)
    max_x = np.max(x)
    exp_vals = np.exp(beta * (x - max_x))
    weights = exp_vals / np.sum(exp_vals)
    return weights

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    n = N
    target_sum = n / 2.0
    
    # Best solution found
    best_h = np.full(n, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    # Initial candidates
    candidates = []
    
    # 1. Constant 0.5
    candidates.append(np.full(n, 0.5))
    
    # 2. Alternating 0 and 1 pattern
    alt = np.zeros(n)
    alt[::2] = 1.0
    candidates.append(alt.copy())
    
    # 3. Alternating with slight offset
    alt2 = np.zeros(n)
    alt2[::2] = 0.8
    alt2[1::2] = 0.2
    candidates.append(alt2.copy())
    
    # 4. Random feasible points
    np.random.seed(42)
    for _ in range(50):
        h = np.random.uniform(0, 1, n)
        h = _project_to_feasible(h)
        candidates.append(h)
    
    # 5. Convex combinations of alternating patterns
    base1 = np.full(n, 0.5)
    base2 = np.zeros(n)
    base2[::2] = 1.0
    base2[1::2] = 0.0
    for w in np.linspace(0, 1, 20):
        h = (1 - w) * base1 + w * base2
        h = _project_to_feasible(h)
        candidates.append(h)
    
    # 6. Smooth periodic patterns
    for freq in [1, 2, 3, 5]:
        for phase in [0, 0.5, 1.0]:
            t = np.arange(n)
            h = 0.5 + 0.3 * np.sin(2 * np.pi * freq * t / n + phase * 2 * np.pi)
            h = _project_to_feasible(h)
            candidates.append(h)
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline:
            break
        h = _project_to_feasible(h)
        score = _compute_max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Optimization phase: gradient descent on soft-max objective
    # Objective: minimize soft_max(overlaps, beta)
    # which approximates max(overlaps)
    
    current_h = best_h.copy()
    
    # Multi-stage beta schedule
    betas = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        # For this beta, do gradient descent
        lr = 0.1 / beta  # Scale learning rate with beta
        max_iters = 200
        iters_per_beta = max(50, int((deadline - time.time()) * 2))
        max_iters = min(max_iters, iters_per_beta)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            overlaps = _compute_all_overlaps(current_h)
            obj = _soft_max(overlaps, beta)
            
            # Gradient of soft-max w.r.t. overlaps
            weights = _soft_max_grad(overlaps, beta)
            
            # Gradient of overlaps w.r.t. h
            # overlaps[k] = n/2 - autocorr[k]
            # autocorr[k] = sum_i h[i] * h[(i+k)%n]
            # d(autocorr[k])/dh[j] = h[(j-k)%n] + h[(j+k)%n]
            # So d(overlaps[k])/dh[j] = -(h[(j-k)%n] + h[(j+k)%n])
            
            grad_h = np.zeros(n)
            # Efficient computation using FFT
            # d(overlaps)/dh = -2 * ifft(weights_fft * conj(fft(h)))
            # More precisely:
            # overlaps[k] = n/2 - sum_i h[i] h[(i+k)%n]
            # d(overlaps[k])/dh[j] = -h[(j-k)%n] - h[(j+k)%n]
            # grad_h[j] = sum_{k=1}^{n-1} weights[k] * d(overlaps[k])/dh[j]
            #           = -sum_{k=1}^{n-1} weights[k] * (h[(j-k)%n] + h[(j+k)%n])
            #           = -2 * sum_{k=1}^{n-1} weights[k] * h[(j-k)%n]  (by symmetry if weights symmetric)
            # But weights may not be symmetric, so:
            # sum_{k=1}^{n-1} weights[k] * h[(j-k)%n] = circular cross-correlation of weights and h
            # Let's compute it directly using FFT
            W = np.fft.fft(weights)
            H = np.fft.fft(current_h)
            # cross_corr[j] = sum_k weights[k] * h[(j-k)%n]
            # = ifft(conj(W) * H) ... need to be careful
            # Actually: ifft(conj(W) * H)[j] = sum_k W[k] * conj(H[k]) ... no
            # Let me think again.
            # sum_{k} a[k] * b[(j-k)%n] = ifft(fft(a) * conj(fft(b)))[j]
            # So sum_k weights[k] * h[(j-k)%n] = ifft(fft(weights) * conj(fft(h)))[j]
            cross_corr = np.fft.ifft(W * np.conj(H)).real
            # Now sum_{k=1}^{n-1} weights[k] * h[(j+k)%n]
            # = sum_{k=1}^{n-1} weights[k] * h[(j-(-k))%n]
            # Let k' = -k, then sum_{k'=-1}^{-(n-1)} weights[-k'] * h[(j+k')%n]
            # = sum_{k'=1}^{n-1} weights[-k'] * h[(j+k')%n]
            # = sum_{k'=1}^{n-1} weights[(n-k')%n] * h[(j+k')%n]
            # Let w_rev[k] = weights[(n-k)%n] for k=1..n-1, w_rev[0]=0
            w_rev = np.zeros(n)
            w_rev[1:] = weights[:0:-1]  # weights[n-1], weights[n-2], ..., weights[1]
            W_rev = np.fft.fft(w_rev)
            cross_corr2 = np.fft.ifft(W_rev * np.conj(H)).real
            
            grad_h = -(cross_corr + cross_corr2)
            
            # Add regularization to keep sum = n/2 and bounds
            # Project gradient or use projected gradient descent
            current_h = current_h - lr * grad_h
            current_h = _project_to_feasible(current_h)
            
            # Check improvement
            new_score = _compute_max_overlap(current_h)
            if new_score < best_score:
                best_score = new_score
                best_h = current_h.copy()
    
    # Final polish: coordinate descent on the top few constraints
    if time.time() < deadline:
        current_h = best_h.copy()
        # Identify top constraints
        overlaps = _compute_all_overlaps(current_h)
        top_k = 5
        top_indices = np.argsort(overlaps)[-top_k:]
        
        # Coordinate descent on pairs
        polish_iters = int((deadline - time.time()) * 10)
        polish_iters = min(polish_iters, 5000)
        
        for it in range(polish_iters):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(n):
                for j in range(i+1, min(i+10, n)):
                    # Try small perturbations
                    for delta in [0.01, -0.01, 0.005, -0.005]:
                        trial = current_h.copy()
                        trial[i] += delta
                        trial[j] -= delta
                        trial = _project_to_feasible(trial)
                        score = _compute_max_overlap(trial)
                        if score < best_score:
                            best_score = score
                            best_h = trial.copy()
                            current_h = trial.copy()
                            improved = True
                            break
            
            if not improved and it > 100:
                break
    
    # Final verification
    best_h = _project_to_feasible(best_h)
    
    return best_h

if __name__ == "__main__":
    h = construct()
    print(f"Sum: {np.sum(h)}, Min: {np.min(h)}, Max: {np.max(h)}")
    print(f"Score: {_compute_max_overlap(h)}")
