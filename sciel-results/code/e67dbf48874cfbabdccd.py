import numpy as np
import time

N = 100
TARGET_SUM = N / 2.0

def compute_D(h):
    """Compute D_s(h) for all s in 1..N-1."""
    D = np.zeros(N-1)
    # h is shape (N,)
    # D_s = sum |h_i - h_{i-s}|
    # We can use np.roll
    for s in range(1, N):
        D[s-1] = np.sum(np.abs(h - np.roll(h, s)))
    return D

def grad_g(h, beta):
    """Gradient of g(h) = -log(sum exp(-beta D_s(h)))."""
    D = compute_D(h)
    # Avoid overflow/underflow in exp
    # w_s = exp(-beta D_s)
    # log sum exp trick: max_D = max(D)
    # w_s = exp(-beta (D_s - max_D))
    max_D = np.max(D)
    w = np.exp(-beta * (D - max_D))
    w_norm = w / np.sum(w)
    
    # grad D_s / dh_j = sgn(h_j - h_{j-s}) - sgn(h_{j+s} - h_j)
    # We need sum_s w_s * grad D_s
    # grad_g = -beta * sum_s w_norm_s * grad D_s
    
    grad = np.zeros(N)
    # To be efficient, we can accumulate contributions
    # For each s, grad D_s is a vector.
    # This is O(N^2).
    for s in range(1, N):
        # diff = h - roll(h, s)
        # sgn(diff)
        diff = h - np.roll(h, s)
        sgn = np.sign(diff)
        # grad D_s[j] = sgn(h_j - h_{j-s}) - sgn(h_{j+s} - h_j)
        # Note: sgn(h_{j+s} - h_j) = -sgn(h_j - h_{j+s})
        # So grad D_s[j] = sgn(h_j - h_{j-s}) + sgn(h_j - h_{j+s})
        # Wait, let's re-verify.
        # D_s = sum_i |h_i - h_{i-s}|
        # d/dh_j |h_j - h_{j-s}| = sgn(h_j - h_{j-s})
        # d/dh_j |h_{j+s} - h_j| = -sgn(h_{j+s} - h_j) = sgn(h_j - h_{j+s})
        # So grad D_s[j] = sgn(h_j - h_{j-s}) + sgn(h_j - h_{j+s})
        # In numpy:
        # sgn1 = sign(h - roll(h, s))
        # sgn2 = sign(h - roll(h, -s))  # which is sign(h_j - h_{j+s})? No.
        # roll(h, -s) shifts left by s. h_{j} in rolled is h_{j+s} in original?
        # np.roll(h, -s)[j] = h[(j+s)%N].
        # So h - np.roll(h, -s) is h_j - h_{j+s}.
        # So sgn2 = sign(h - np.roll(h, -s)).
        
        # We already computed diff = h - np.roll(h, s) which is h_j - h_{j-s}.
        # So sgn1 = sgn.
        # sgn2 = np.sign(h - np.roll(h, -s))
        
        grad += w_norm[s-1] * (sgn + np.sign(h - np.roll(h, -s)))
        
    return -beta * grad

def project(h):
    """Project h onto {x: sum x = TARGET_SUM, 0 <= x <= 1}."""
    # Binary search on mu
    # h_i' = clip(h_i + mu, 0, 1)
    # sum h_i' = TARGET_SUM
    
    lo = -np.max(h)
    hi = 1 - np.min(h)
    
    for _ in range(100):
        mu = (lo + hi) / 2.0
        h_proj = np.clip(h + mu, 0, 1)
        s = np.sum(h_proj)
        if s > TARGET_SUM:
            hi = mu
        else:
            lo = mu
            
    return np.clip(h + mu, 0, 1)

def construct():
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    best_h = None
    best_score = np.inf # We want to minimize max overlap, which is N/2 - 0.5 * min D_s
    # Min overlap = N/2 - 0.5 * min D_s
    # So maximizing min D_s is equivalent.
    
    # Initialize with random
    # Maybe multiple restarts
    for restart in range(10):
        if time.time() - start_time > time_limit:
            break
            
        h = np.random.uniform(0, 1, N)
        h = project(h)
        
        beta = 10.0
        lr = 0.01
        
        for step in range(500):
            if time.time() - start_time > time_limit:
                break
                
            D = compute_D(h)
            min_D = np.min(D)
            current_overlap = N/2.0 - 0.5 * min_D
            
            if current_overlap < best_score:
                best_score = current_overlap
                best_h = h.copy()
                
            # If min_D is very high, we can stop or increase beta
            # Max possible D_s is 2 * sum h_i = N. So min_D <= N.
            # Actually max |h_i - h_j| is 1, so D_s <= N.
            # If min_D is close to N, overlap is close to 0.
            
            grad = grad_g(h, beta)
            h = h + lr * grad
            h = project(h)
            
            # Adaptive learning rate or beta increase
            if step % 50 == 0 and step > 0:
                # Increase beta to sharpen the min approximation
                beta *= 1.5
                lr /= 1.5

    if best_h is None:
        return np.full(N, 0.5)
        
    # Final polish: local search
    # Try small perturbations
    for i in range(N):
        if time.time() - start_time > time_limit:
            break
        for delta in [-0.01, 0.01]:
            h_trial = best_h.copy()
            h_trial[i] += delta
            # Adjust another element to maintain sum?
            # Or just project?
            h_trial = project(h_trial)
            D = compute_D(h_trial)
            min_D = np.min(D)
            overlap = N/2.0 - 0.5 * min_D
            if overlap < best_score:
                best_score = overlap
                best_h = h_trial
    
    return best_h
