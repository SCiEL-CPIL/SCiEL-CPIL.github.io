"""Erdos minimum overlap: minimize max overlap of h with shifted (1-h)."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 120.0  # seconds, evaluator kills at 150s, keep safety margin
START_TIME = time.time()

def _remaining():
    return DEADLINE - (time.time() - START_TIME)

def _overlap_profile(h):
    """Compute overlap O(k) = sum_i h[i]*(1-h[(i+k)%N]) for all shifts k."""
    # O(k) = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
    # sum_i h[i] = N/2 (by constraint)
    # So O(k) = N/2 - autocorrelation of h at lag k
    # autocorrelation: C(k) = sum_i h[i]*h[(i+k)%N]
    h_ext = np.concatenate([h, h])
    # C(k) = sum_{i=0}^{N-1} h[i] * h[i+k] for k in 0..N-1
    C = np.array([np.dot(h, h_ext[k:k+N]) for k in range(N)])
    return N / 2.0 - C

def _max_overlap(h):
    return np.max(_overlap_profile(h))

def _soft_max_overlap(h, beta):
    """Smooth surrogate for max overlap using log-sum-exp."""
    O = _overlap_profile(h)
    # log-sum-exp for numerical stability
    M = np.max(O)
    return M + np.log(np.sum(np.exp(beta * (O - M)))) / beta

def _project(h, target_sum):
    """Project h onto {x : 0 <= x[i] <= 1, sum(x) = target_sum}."""
    h = np.clip(h, 0.0, 1.0)
    # Adjust to meet sum constraint via water-filling
    s = np.sum(h)
    if abs(s - target_sum) < 1e-12:
        return h
    # We need to add (target_sum - s) distributed across entries
    diff = target_sum - s
    # If diff > 0, increase entries with most room (closest to 1)
    # If diff < 0, decrease entries with most room (closest to 0)
    # Use iterative projection
    for _ in range(50):
        s = np.sum(h)
        diff = target_sum - s
        if abs(diff) < 1e-10:
            break
        if diff > 0:
            # Increase: prioritize entries not at 1
            room = 1.0 - h
            mask = room > 1e-9
            if not np.any(mask):
                break
            # Distribute proportional to room
            add = diff / np.sum(room[mask]) * room[mask]
            h[mask] = np.minimum(1.0, h[mask] + add)
        else:
            # Decrease: prioritize entries not at 0
            room = h
            mask = room > 1e-9
            if not np.any(mask):
                break
            sub = (-diff) / np.sum(room[mask]) * room[mask]
            h[mask] = np.maximum(0.0, h[mask] - sub)
    # Final clip and fix
    h = np.clip(h, 0.0, 1.0)
    # Tiny residual fix
    s = np.sum(h)
    diff = target_sum - s
    if abs(diff) > 1e-12:
        # Distribute to entries with room
        if diff > 0:
            mask = h < 1.0 - 1e-12
            if np.any(mask):
                h[mask] += diff / np.sum(mask)
        else:
            mask = h > 1e-12
            if np.any(mask):
                h[mask] += diff / np.sum(mask)
    h = np.clip(h, 0.0, 1.0)
    return h

def _gradient_soft_max(h, beta):
    """Gradient of soft-max overlap w.r.t. h."""
    O = _overlap_profile(h)
    # d O(k)/d h[i] = (1 - h[(i+k)%N]) - h[(i+k)%N] ... let me recompute
    # O(k) = sum_j h[j] * (1 - h[(j+k)%N])
    # d O(k)/d h[i] = (1 - h[(i+k)%N]) - h[(i-k)%N]
    #   because h[i] appears in term j=i as h[i]*(1-h[(i+k)%N]) -> deriv = 1-h[(i+k)%N]
    #   and in term j=i-k as h[j]*(1-h[i]) -> deriv = -h[j] = -h[(i-k)%N]
    # So d O(k)/d h[i] = (1 - h[(i+k)%N]) - h[(i-k)%N]
    
    h_ext = np.concatenate([h, h])
    # For each i, k: term = 1 - h[(i+k)%N] - h[(i-k)%N]
    # This is a convolution-like structure
    # d O(k)/d h[i] = 1 - h[(i+k)%N] - h[(i-k)%N]
    
    # Soft max: f = (1/beta) log(sum_k exp(beta O(k)))
    # df/d O(k) = exp(beta O(k)) / sum_j exp(beta O(j))
    # df/d h[i] = sum_k df/dO(k) * dO(k)/dh[i]
    
    # For efficiency, compute weights
    M = np.max(O)
    exp_O = np.exp(beta * (O - M))
    weights = exp_O / np.sum(exp_O)  # softmax weights
    
    # df/dh[i] = sum_k weights[k] * (1 - h[(i+k)%N] - h[(i-k)%N])
    # = sum_k weights[k] - sum_k weights[k]*h[(i+k)%N] - sum_k weights[k]*h[(i-k)%N]
    # = 1 - (weights * h)_cyc[i] - (weights_rev * h)_cyc[i]
    # where (weights * h)_cyc[i] = sum_k weights[k] h[(i+k)%N]
    # and (weights_rev * h)_cyc[i] = sum_k weights[k] h[(i-k)%N] = sum_k weights[N-k] h[(i+k)%N]
    
    # These are circular convolutions
    # sum_k weights[k] h[(i+k)%N] = circular convolution of h and weights
    # Using FFT for speed
    W = np.fft.rfft(weights)
    H = np.fft.rfft(h)
    conv1 = np.fft.irfft(W * H, n=N)  # (weights * h)_cyc
    conv2 = np.fft.irfft(W * np.conj(np.fft.rfft(h[::-1])) * np.conj(np.fft.rfft(h[::-1])), n=N)  # not quite
    
    # Let me just do it directly for clarity since N=100 is small
    # sum_k weights[k] * h[(i+k)%N]
    # = sum_k weights[k] * h[(i+k)%N]
    # This is the circular cross-correlation
    # conv1[i] = sum_k weights[k] h[(i+k) % N]
    conv1 = np.array([np.dot(weights, h[(i + np.arange(N)) % N]) for i in range(N)])
    conv2 = np.array([np.dot(weights, h[(i - np.arange(N)) % N]) for i in range(N)])
    
    grad = 1.0 - conv1 - conv2
    return grad

def _construct_gradient():
    """Optimize soft-max overlap with projected gradient descent."""
    target_sum = N / 2.0
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Start from a perturbed initial point to escape constant minimum
    np.random.seed(42)
    
    # Try multiple initializations
    inits = [
        np.full(N, 0.5),
        0.5 + 0.1 * np.random.randn(N),
        0.5 + 0.2 * np.sin(2 * np.pi * np.arange(N) / N * 3),
        0.5 + 0.15 * np.cos(2 * np.pi * np.arange(N) / N * 2),
    ]
    
    for init in inits:
        if _remaining() < 5.0:
            break
        h = _project(init.copy(), target_sum)
        curr_score = _max_overlap(h)
        
        # Adaptive beta schedule
        betas = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        for beta in betas:
            if _remaining() < 3.0:
                break
            # Gradient descent with decreasing step size
            lr = 0.05 / beta
            for it in range(300):
                if _remaining() < 2.0:
                    break
                grad = _gradient_soft_max(h, beta)
                h_new = h - lr * grad
                h_new = _project(h_new, target_sum)
                
                # Check improvement
                score = _max_overlap(h_new)
                if score < curr_score - 1e-10:
                    h = h_new
                    curr_score = score
                    if curr_score < best_score:
                        best_score = curr_score
                        best_h = h.copy()
                else:
                    # Reduce step size
                    lr *= 0.5
                    if lr < 1e-10:
                        break
    
    return best_h, best_score

def _construct_local_search():
    """Local search: try small perturbations and keep improvements."""
    target_sum = N / 2.0
    best_h, best_score = _construct_gradient()
    
    if _remaining() < 5.0:
        return best_h, best_score
    
    # Coordinate descent / local search
    np.random.seed(123)
    for it in range(5000):
        if _remaining() < 3.0:
            break
        # Pick random pair to swap/adjust
        i = np.random.randint(N)
        j = np.random.randint(N)
        if i == j:
            continue
        # Try moving mass from j to i
        for delta in [0.01, 0.05, 0.1, 0.2]:
            h_new = best_h.copy()
            h_new[i] += delta
            h_new[j] -= delta
            h_new = _project(h_new, target_sum)
            score = _max_overlap(h_new)
            if score < best_score - 1e-9:
                best_h = h_new
                best_score = score
                break
    
    return best_h, best_score

def _polish(h, target_sum):
    """High-precision polish using scipy minimize on soft-max."""
    if _remaining() < 10.0:
        return h, _max_overlap(h)
    
    best_h = h.copy()
    best_score = _max_overlap(h)
    
    # Use scipy to polish with a high beta
    beta = 50.0
    
    def objective(x):
        x_proj = _project(x, target_sum)
        return _soft_max_overlap(x_proj, beta)
    
    # Parameterize to avoid constraints: use h = sigmoid(z) and adjust
    # Actually, let's just use the projected approach with scipy
    x0 = best_h.copy()
    
    res = minimize(objective, x0, method='L-BFGS-B', 
                   options={'maxiter': 200, 'ftol': 1e-12})
    
    if res.success or res.fun < objective(x0):
        h_new = _project(res.x, target_sum)
        score = _max_overlap(h_new)
        if score < best_score:
            best_h = h_new
            best_score = score
    
    return best_h, best_score

def construct():
    """Construct step heights minimizing max overlap with shifted 1-h."""
    target_sum = N / 2.0
    
    # Phase 1: Gradient-based optimization with soft-max surrogate
    h, score = _construct_gradient()
    
    # Phase 2: Local search refinement
    if _remaining() > 10.0:
        h, score = _construct_local_search()
    
    # Phase 3: High-precision polish
    if _remaining() > 15.0:
        h, score = _polish(h, target_sum)
    
    # Final: ensure constraints are exactly met
    h = _project(h, target_sum)
    
    # Verify
    assert np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12), "h out of [0,1]"
    assert abs(np.sum(h) - target_sum) < 1e-8, f"sum = {np.sum(h)}, expected {target_sum}"
    
    return h
# EVOLVE-BLOCK-END
