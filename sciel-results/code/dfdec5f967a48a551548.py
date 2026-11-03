"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np

def _linear_overlap_values(h):
    """Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_i h[i]*h[(i+k)%N]."""
    n = len(h)
    # Use FFT to compute circular autocorrelation
    H = np.fft.fft(h)
    auto = np.fft.ifft(np.abs(H)**2).real
    # auto[k] = sum_i h[i] * h[(i+k) % n]
    # overlap_k = sum_i h[i] * (1 - h[(i+k)%n]) = sum(h) - auto[k] = n/2 - auto[k]
    # C_k = (2/n) * overlap_k
    cs = (2.0 / n) * (n / 2.0 - auto)
    return cs

def _score(h):
    """Return the max overlap value (the quantity to minimize)."""
    cs = _linear_overlap_values(h)
    return np.max(cs)

def _project(h):
    """Project h onto the feasible set: 0 <= h_i <= 1, sum(h) = N/2."""
    h = np.clip(h, 0, 1)
    s = np.sum(h)
    target = N / 2.0
    if s == 0:
        h = np.full(N, 0.5)
        return h
    # If sum is too high, reduce; if too low, increase
    # Simple approach: scale, then clip, then adjust
    if abs(s - target) > 1e-12:
        # Iterative projection
        for _ in range(50):
            h = np.clip(h, 0, 1)
            s = np.sum(h)
            if abs(s - target) < 1e-12:
                break
            # Adjust by shifting mass
            if s > target:
                # Reduce elements that are not at 0
                mask = h > 1e-10
                if np.sum(mask) == 0:
                    break
                excess = s - target
                # Reduce proportionally among non-zero elements
                h[mask] -= excess * h[mask] / s
                h[mask] = np.clip(h[mask], 0, 1)
            else:
                # Increase elements that are not at 1
                mask = h < 1.0 - 1e-10
                if np.sum(mask) == 0:
                    break
                deficit = target - s
                # Increase proportionally among non-saturated elements
                h[mask] += deficit * (1.0 - h[mask]) / np.sum(1.0 - h[mask])
                h[mask] = np.clip(h[mask], 0, 1)
        # Final correction
        diff = target - np.sum(h)
        if abs(diff) > 1e-12:
            if diff > 0:
                idx = np.argmin(h)
            else:
                idx = np.argmax(h)
            h[idx] += diff
            h[idx] = np.clip(h[idx], 0, 1)
    return h

def _smooth_max_obj_grad(h, beta):
    """
    Compute smooth approximation of max(C_k) and its gradient.
    Uses log-sum-exp: f(h) = (1/beta) * log(sum_k exp(beta * C_k))
    Gradient: df/dh_i = sum_k (exp(beta*C_k) / sum_j exp(beta*C_j)) * dC_k/dh_i
    dC_k/dh_i = (2/N) * (1 - h[(i+k)%N]) ... wait, let me recompute.
    
    C_k = (2/N) * (N/2 - sum_i h[i]*h[(i+k)%N])
    dC_k/dh_i = (2/N) * (-h[(i+k)%N]) for the first term, and also from the second h[(i-k)%N]
    Actually: sum_i h[i]*h[(i+k)%N], derivative w.r.t. h[i] is h[(i+k)%N]
    And also h[i] appears as h[(j-k)%N] when j = i+k, so h[i] * h[j] where j-k=i, i.e., h[i]*h[(i+k)%N]
    Wait, the sum is over all pairs. d/dh_i of sum_j h[j]*h[(j+k)%N] = h[(i+k)%N] + h[(i-k)%N]
    (because h[i] appears once as h[j] with j=i, and once as h[(j+k)%N] with j=i-k)
    
    So dC_k/dh_i = (2/N) * (-(h[(i+k)%N] + h[(i-k)%N]))
    
    Actually wait: C_k = (2/N)*(N/2 - autocorr_k)
    autocorr_k = sum_i h[i] h[(i+k)%N]
    d(autocorr_k)/dh[i] = h[(i+k)%N] + h[(i-k)%N]
    (since h[i] appears in term i as h[i]*h[(i+k)%N] and in term (i-k) as h[(i-k)]*h[i])
    
    So dC_k/dh[i] = (2/N) * (-(h[(i+k)%N] + h[(i-k)%N]))
    """
    n = len(h)
    cs = _linear_overlap_values(h)
    
    # Compute softmax weights
    max_c = np.max(cs)
    exp_cs = np.exp(beta * (cs - max_c))
    sum_exp = np.sum(exp_cs)
    weights = exp_cs / sum_exp  # weights[k] = exp(beta*C_k) / sum_j exp(beta*C_j)
    
    # Objective value
    obj = (1.0 / beta) * (np.log(sum_exp) + beta * max_c)
    
    # Gradient: grad[i] = sum_k weights[k] * dC_k/dh[i]
    # dC_k/dh[i] = (2/N) * (-(h[(i+k)%N] + h[(i-k)%N]))
    # grad[i] = (2/N) * sum_k weights[k] * (-(h[(i+k)%N] + h[(i-k)%N]))
    
    # Efficient computation using convolution
    # sum_k weights[k] * h[(i+k)%N] = circular convolution of weights and h
    # sum_k weights[k] * h[(i-k)%N] = circular convolution of weights and flipped h
    
    # Use FFT
    W = np.fft.fft(weights)
    H = np.fft.fft(h)
    conv1 = np.fft.ifft(W * H).real  # conv1[i] = sum_k weights[k] * h[(i+k)%N]
    
    # For conv2[i] = sum_k weights[k] * h[(i-k)%N]
    # This is cross-correlation, which is convolution with flipped weights
    W_flip = np.fft.fft(weights[::-1]) if n > 1 else W
    # Actually, simpler: conv2[i] = sum_k weights[k] * h[(i-k)%N]
    # = sum_k weights[(i-k)%N] * h[k] ... let me think differently
    # conv2[i] = sum_k weights[k] * h[(i-k)%N]
    # Let j = i-k, then k = i-j, conv2[i] = sum_j weights[(i-j)%N] * h[j]
    # This is circular cross-correlation of weights and h
    # = ifft(fft(weights) * conj(fft(h))) ... no
    # Actually conv2[i] = sum_k weights[k] * h[(i-k)%N]
    # In FFT: this is ifft(fft(weights) * fft(h)) but with a flip
    # Let's just use: conv2[i] = sum_k weights[k] * h[(i-k)%N]
    # = (weights * h)[i] in circular correlation sense
    # Using FFT: ifft(conj(fft(weights)) * fft(h)) gives sum_k weights[(i-k)%N] * h[k]
    # That's not quite right either.
    
    # Simplest: just compute directly for N=100
    conv2 = np.zeros(n)
    for i in range(n):
        conv2[i] = np.sum(weights * np.roll(h, i))
    # Wait, np.roll(h, i)[k] = h[(k-i)%n], so np.roll(h, i)[(i-k)%n]... this is getting complicated.
    # Let me just do it directly:
    conv2 = np.zeros(n)
    for i in range(n):
        # conv2[i] = sum_k weights[k] * h[(i-k)%n]
        conv2[i] = np.dot(weights, np.roll(h, i))
        # np.roll(h, i)[k] = h[(k - i) % n], so np.roll(h, i)[(i-k)%n]... 
        # Actually np.roll(h, i) shifts h to the right by i, so element k of rolled is h[(k-i)%n]
        # We want sum_k weights[k] * h[(i-k)%n]
        # Let m = i-k, then k = i-m, sum_m weights[(i-m)%n] * h[m]
        # = sum_m h[m] * weights[(i-m)%n]
        # This is the circular cross-correlation of h and weights at lag i
        # = ifft(fft(h) * conj(fft(weights)))... let me verify
        # Actually let me just compute it properly:
    
    # Redo conv2 properly
    conv2 = np.zeros(n)
    for i in range(n):
        # conv2[i] = sum_k weights[k] * h[(i-k) % n]
        # Create array where index k has h[(i-k)%n]
        idx = (i - np.arange(n)) % n
        conv2[i] = np.dot(weights, h[idx])
    
    grad = (2.0 / n) * (-(conv1 + conv2))
    
    return obj, grad

def _nesterov_accelerated_gradient(h_init, beta_start, beta_end, n_iter, deadline):
    """Nesterov accelerated gradient descent with adaptive beta."""
    h = h_init.copy()
    h_prev = h_init.copy()
    best_h = h_init.copy()
    best_score = _score(h_init)
    
    lr = 1.0 / (beta_end * N)  # initial learning rate
    
    for t in range(n_iter):
        if time.time() > deadline:
            break
        
        # Adaptive beta
        beta = beta_start + (beta_end - beta_start) * (t / max(1, n_iter - 1))
        
        # Nesterov momentum
        y = h + (t / (t + 2.0)) * (h - h_prev)
        y = _project(y)
        
        obj, grad = _smooth_max_obj_grad(y, beta)
        
        # Backtracking line search
        alpha = lr
        for _ in range(20):
            h_new = y - alpha * grad
            h_new = _project(h_new)
            obj_new, _ = _smooth_max_obj_grad(h_new, beta)
            if obj_new <= obj - 0.5 * alpha * np.sum(grad**2):
                break
            alpha *= 0.5
        
        h_prev = h
        h = h_new
        
        score = _score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    return best_h, best_score

def _coordinate_descent_polish(h, n_rounds, deadline):
    """Coordinate descent to polish the solution."""
    h = h.copy()
    best_score = _score(h)
    
    for _ in range(n_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        cs = _linear_overlap_values(h)
        k_max = np.argmax(cs)
        
        # Try moving mass from high-contributing positions to low-contributing ones
        # For the worst shift k_max, overlap = sum_i h[i]*(1-h[(i+k_max)%N])
        # We want to reduce this.
        
        # Try small perturbations: for each pair (i, j), try moving epsilon from i to j
        # This is O(N^2) per round, which is 10000 for N=100, acceptable
        
        eps = 0.005
        for i in range(N):
            if time.time() > deadline:
                break
            if h[i] < eps:
                continue
            for j in range(N):
                if i == j:
                    continue
                if h[j] > 1.0 - eps:
                    continue
                
                h_new = h.copy()
                h_new[i] -= eps
                h_new[j] += eps
                h_new = _project(h_new)
                
                score_new = _score(h_new)
                if score_new < best_score - 1e-10:
                    h = h_new
                    best_score = score_new
                    improved = True
        
        if not improved:
            break
    
    return h, best_score

def construct():
    """
    Choose equal-width step heights h on [0,2], bounded between 0 and 1 with integral 1,
    to minimize the maximum overlap of h with shifted 1-h.
    """
    deadline = time.time() + 145  # safety margin
    
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    # Multiple diverse initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.full(N, 0.5))
    
    # 2. Cosine pattern
    h_cos = np.full(N, 0.5)
    for k in range(1, 6):
        h_cos += 0.1 * np.cos(2*np.pi*k*np.arange(N)/N)
    inits.append(_project(h_cos))
    
    # 3. Sine pattern
    h_sin = np.full(N, 0.5)
    for k in range(1, 6):
        h_sin += 0.1 * np.sin(2*np.pi*k*np.arange(N)/N)
    inits.append(_project(h_sin))
    
    # 4. Random
    h_rand = np.random.uniform(0, 1, N)
    inits.append(_project(h_rand))
    
    # 5. Alternating
    h_alt = np.array([0.5 + 0.3*(1 if i%2==0 else -1) for i in range(N)])
    inits.append(_project(h_alt))
    
    # 6. Block pattern
    h_block = np.zeros(N)
    h_block[:N//2] = 1.0
    inits.append(_project(h_block))
    
    # 7. Another cosine with different phases
    h_cos2 = np.full(N, 0.5)
    for k in range(1, 8):
        h_cos2 += 0.08 * np.cos(2*np.pi*k*np.arange(N)/N + np.pi/4)
    inits.append(_project(h_cos2))
    
    # 8. Random with structure
    h_struct = np.random.uniform(0.2, 0.8, N)
    inits.append(_project(h_struct))
    
    for init_idx, h_init in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        h_init = _project(h_init)
        score_init = _score(h_init)
        
        if score_init < best_score:
            best_score = score_init
            best_h = h_init.copy()
        
        # Run Nesterov AGD
        try:
            h_opt, score_opt = _nesterov_accelerated_gradient(
                h_init, 
                beta_start=50, 
                beta_end=500, 
                n_iter=300, 
                deadline=deadline - 5
            )
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass
        
        # Coordinate descent polish
        if time.time() < deadline - 3:
            try:
                h_polished, score_polished = _coordinate_descent_polish(
                    best_h, 
                    n_rounds=20, 
                    deadline=deadline - 2
                )
                if score_polished < best_score:
                    best_score = score_polished
                    best_h = h_polished.copy()
            except Exception:
                pass
    
    # Final projection to ensure feasibility
    best_h = _project(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
