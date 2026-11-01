"""Optimized signed step function via Gumbel-Softmax relaxation and efficient local search."""
import numpy as np
import time

N = 400
DT = 0.5 / N  # Interval width: 0.5 / 400


def _compute_autoconv(signed_heights):
    """Compute the autoconvolution via FFT and return as a 1D array of length N (lags 0..N-1)."""
    n = 1
    while n < 2 * N:
        n <<= 1
    F = np.fft.rfft(signed_heights, n)
    R = np.fft.irfft(F * np.conj(F), n)
    return R[:N] * DT


def _score_binary(signs):
    """
    Compute the score for a binary sequence.
    The task mentions 'upper bound on C3'. 
    We minimize the maximum absolute value of the sidelobes.
    """
    R = _compute_autoconv(signs)
    sidelobes = R[1:]
    return np.max(np.abs(sidelobes))


def _gumbel_sigmoid(logits, temperature=1.0, hard=True):
    """
    Gumbel-Sigmoid reparameterization.
    Returns y in [0, 1].
    """
    # Sample Gumbel noise
    u = np.random.rand(N)
    # Avoid log(0)
    u = np.clip(u, 1e-8, 1 - 1e-8)
    g = -np.log(-np.log(u))
    
    y = 1.0 / (1.0 + np.exp(-(logits + g) / temperature))
    
    if hard:
        y_hard = (y > 0.5).astype(float)
        # Straight-through estimator: forward pass uses hard, backward pass uses soft
        # For manual gradient computation, we just return the soft y for gradient calc
        # and hard y for the actual signal evaluation if needed.
        # Here we return the soft y for the gradient flow, but we will evaluate the hard version.
        pass
        
    return y


def _compute_gradient(logits, temperature=1.0):
    """
    Compute the gradient of the soft-max loss w.r.t logits using Gumbel-Softmax.
    This is an approximation. We sample one Gumbel noise vector.
    """
    # 1. Sample Gumbel noise
    u = np.random.rand(N)
    u = np.clip(u, 1e-8, 1 - 1e-8)
    g = -np.log(-np.log(u))
    
    # 2. Compute soft sigmoid
    z = (logits + g) / temperature
    y = 1.0 / (1.0 + np.exp(-z))
    
    # 3. Map to signs: s = 2y - 1
    s = 2.0 * y - 1.0
    
    # 4. Compute Autoconvolution R
    R = _compute_autoconv(s)
    sidelobes = R[1:]
    
    # 5. Compute Loss: Soft-Max of squared sidelobes
    tau = 0.01
    R_sq = sidelobes ** 2
    m = np.max(R_sq)
    if m < 1e-8:
        return np.zeros(N)
        
    # SoftMax approximation: L = m + tau * log(sum(exp((R_sq - m)/tau)))
    # Gradient of L w.r.t R_sq:
    # dL/dR_sq_k = exp((R_sq_k - m)/tau) / sum(exp((R_sq_j - m)/tau))
    exp_terms = np.exp((R_sq - m) / tau)
    sum_exp = np.sum(exp_terms)
    dL_dRsq = exp_terms / sum_exp
    
    # dL/dR_k = 2 * R_k * dL_dRsq_k
    dL_dR = 2.0 * sidelobes * dL_dRsq
    
    # 6. Compute dR/ds
    # R_k = sum_j s_j s_{j-k}
    # dR_k/ds_i = s_{i-k} + s_{i+k} (with boundary handling)
    # We need dL/ds_i = sum_k dL_dR_k * dR_k/ds_i
    
    # Efficiently compute dL/ds using convolution/correlation
    # dR/ds is a linear operator.
    # Let's compute it via FFT or direct loops?
    # N=400, direct loops are O(N^2) = 160k ops. This is fast enough for a few hundred steps.
    
    dL_ds = np.zeros(N)
    for i in range(N):
        grad_i = 0.0
        # Sum over k where i is involved
        # R_k involves s_i if i is in [0, N-1] and i-k is in [0, N-1] or i+k is in [0, N-1]
        # Actually, dR_k/ds_i = s_{i-k} + s_{i+k}
        # We sum over k=1..N-1
        for k in range(1, N):
            term1 = 0.0
            term2 = 0.0
            if 0 <= i - k < N:
                term1 = s[i - k]
            if 0 <= i + k < N:
                term2 = s[i + k]
            grad_i += dL_dR[k - 1] * (term1 + term2)
        dL_ds[i] = grad_i
        
    # 7. Backprop through Gumbel-Sigmoid
    # s = 2y - 1 => ds/dy = 2
    # y = sigmoid(z) => dy/dz = y(1-y)
    # z = (logits + g)/T => dz/dlogits = 1/T
    # dL/dlogits = dL_ds * 2 * y(1-y) / T
    
    dy_dz = y * (1.0 - y)
    dL_dlogits = dL_ds * 2.0 * dy_dz / temperature
    
    return dL_dlogits


def _local_search_incremental(signs, max_time=10.0):
    """
    Efficient local search using incremental autoconvolution updates.
    """
    start_time = time.time()
    current_signs = signs.copy()
    R = _compute_autoconv(current_signs)
    current_score = np.max(np.abs(R[1:]))
    
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        best_gain = 0
        best_flip = -1
        
        # Check all possible flips
        for i in range(N):
            s_i = current_signs[i]
            
            # Calculate Delta R for flipping i
            # Delta R[k] = -2 * s_i * (s[i-k] + s[i+k]) * DT
            # We only need to update R temporarily to check the max.
            # Instead of full update, we can estimate the change in peak.
            # But to be safe, let's do the full update for the best candidate found in a pass?
            # No, we can just compute the new R for each i? That's O(N^2) per pass.
            # Pass is O(N^2). Total time O(N^2 * iters).
            
            # Let's try a faster heuristic:
            # Only flip if it reduces the current peak significantly?
            
            # For now, let's stick to the O(N^2) pass but optimize the inner loop.
            # We can precompute the "neighbor sum" for each i.
            # neighbor_sum[i] = sum_{k=1}^{N-1} (s[i-k] + s[i+k]) * weight_k?
            # No, the weight depends on the current R[k] if we were doing gradient.
            # Here we are doing greedy: flip if max(|R_new|) < max(|R_old|).
            
            # Let's just do the full update for each i, but it's slow in Python.
            # Alternative: Use NumPy vectorization for the update.
            
            # Delta R vector:
            # dR = zeros(N)
            # For k in 1..N-1:
            #   if i-k >= 0: dR[k] += -2 * s_i * s[i-k]
            #   if i+k < N: dR[k] += -2 * s_i * s[i+k]
            
            # This is hard to vectorize easily per i without loops.
            
            # Let's use a different strategy: 
            # Compute the gradient of the peak w.r.t s_i?
            # No, let's just use the previous method but limit iterations.
            
            pass
            
    return current_signs

def construct():
    start_time = time.time()
    timeout = 110.0
    
    np.random.seed(42)
    
    # Phase 1: Gradient Descent on Logits with Gumbel-Softmax
    logits = np.zeros(N)
    lr = 0.1
    temperature = 1.0
    
    best_signs = np.ones(N)
    best_score = _score_binary(best_signs)
    
    # Run GD for a short time
    for step in range(200):
        if time.time() - start_time > timeout - 20:
            break
            
        # Compute gradient
        grad = _compute_gradient(logits, temperature)
        
        # Update logits
        logits -= lr * grad
        
        # Sample hard signs
        u = np.random.rand(N)
        u = np.clip(u, 1e-8, 1 - 1e-8)
        g = -np.log(-np.log(u))
        y = 1.0 / (1.0 + np.exp(-(logits + g) / temperature))
        signs = (2.0 * y - 1.0)
        # Hard binarize
        signs = np.sign(signs)
        # Handle zeros (unlikely)
        signs[signs == 0] = 1.0
        
        score = _score_binary(signs)
        
        if score < best_score:
            best_score = score
            best_signs = signs.copy()
            
        # Cool temperature
        temperature *= 0.99
        
    # Phase 2: Local Search on best_signs
    # Use a simple random restart local search or incremental if possible.
    # Given the complexity of efficient incremental search in pure Python without C extensions,
    # we will use a limited number of full evaluations for local search.
    
    current_signs = best_signs.copy()
    current_score = best_score
    
    local_start = time.time()
    max_local_time = timeout - (time.time() - start_time) - 5
    
    improved = True
    iters = 0
    while improved and time.time() - local_start < max_local_time:
        improved = False
        iters += 1
        if iters > 50: # Limit passes
            break
            
        # Try flipping each bit
        for i in range(N):
            new_signs = current_signs.copy()
            new_signs[i] *= -1
            new_score = _score_binary(new_signs)
            
            if new_score < current_score:
                current_signs = new_signs
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_signs = current_signs.copy()
                    
    return best_signs
