import numpy as np
import time

N = 100
TIME_LIMIT = 110.0  # Safety margin

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

_START_TIME = time.time()

def _compute_max_overlap(h):
    """
    Computes the maximum overlap of h with shifted (1-h).
    Assumes circular shifts.
    Overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
    """
    g = 1.0 - h
    # Compute all circular cross-correlations via FFT
    # overlap_k = sum_i h[i] * g[(i+k) % N]
    # This is equivalent to the circular cross-correlation.
    # Using FFT:
    H = np.fft.fft(h)
    G = np.fft.fft(g)
    # Correlation of h and g: sum_i h[i] conj(g[i-k])?
    # Standard definition: (f * g)_n = sum_k f_k g_{n-k}
    # We want C_k = sum_i h_i g_{i+k}
    # Let g' = g reversed?
    # Actually, cross-correlation of a and b is sum a_i b_{i+k}.
    # In Fourier domain, if A = fft(a), B = fft(b), then
    # C = ifft(A * conj(B)) gives sum a_i b_{i-k} ?
    # Let's check indices.
    # fft(a)_n = sum_k a_k e^{-2pi i k n / N}
    # ifft(A * conj(B))_k = (1/N) sum_n A_n conj(B_n) e^{2pi i k n / N}
    # = (1/N) sum_n (sum_j a_j e^{-2pi i j n / N}) (sum_l conj(b_l) e^{2pi i l n / N}) e^{2pi i k n / N}
    # = sum_j sum_l a_j conj(b_l) (1/N) sum_n e^{2pi i n (l - j + k) / N}
    # = sum_j sum_l a_j conj(b_l) delta_{l-j+k, 0}
    # = sum_j a_j conj(b_{j-k})
    # Since h and g are real, conj(b_{j-k}) = b_{j-k}.
    # So this computes sum_j h_j g_{j-k}.
    # We want sum_j h_j g_{j+k}. This is just the same but with index -k.
    # So C_k = (1/N) * ifft(fft(h) * conj(fft(g))) at index -k?
    # Or simply, let C = ifft(fft(h) * conj(fft(g))). Then C_k = sum_j h_j g_{j-k}.
    # We want sum_j h_j g_{j+k} = sum_j h_j g_{j-(-k)} = C_{-k} = C_{N-k} (since real, symmetric).
    # Actually, if we define overlap_k = sum_i h_i (1-h)_{i+k}.
    # Let's just use the direct definition if N is small, or check FFT.
    # For N=100, direct O(N^2) = 10000 ops is fast.
    # FFT is O(N log N) ~ 700 ops.
    # Let's use FFT for gradient computation later. For max, direct is fine for speed check?
    # No, let's use FFT for consistency.
    
    H = np.fft.fft(h)
    G = np.fft.fft(g)
    # Correlation sum_i h_i g_{i+k}
    # This is circular correlation.
    # np.correlate? No, circular.
    # C = ifft(fft(h) * conj(fft(g))) -> sum_i h_i g_{i-k}
    # We want sum_i h_i g_{i+k} = sum_i h_i g_{i-(-k)}
    # So it's the element at index -k in the standard correlation?
    # Let's compute corr = ifft(fft(h) * conj(fft(g)))
    # corr[k] = sum_i h_i g_{i-k}
    # We want O[k] = sum_i h_i g_{i+k} = corr[-k] = corr[N-k] (for k!=0)
    # Since h, g real, corr is symmetric?
    # corr[-k] = sum_i h_i g_{i+k}.
    # corr[k] = sum_i h_i g_{i-k}.
    # If we swap h and g?
    # Let's just compute the vector of overlaps directly using FFT.
    
    # overlap_k = sum_i h_i g_{i+k}
    # This is the cross-correlation of h and g with lag k.
    # In numpy, no built-in circular correlation.
    # But we can do:
    # O = ifft(fft(h) * conj(fft(g)))
    # Then O[k] = sum_i h_i g_{i-k}.
    # So overlap_k (defined as shift k of g) corresponds to O[N-k] ?
    # Let's test with small N.
    # Actually, let's just use the direct loop for N=100, it's fast enough for scoring,
    # but for gradient we need it many times.
    # Let's assume the evaluator uses a specific definition.
    # "overlap of h with shifted 1-h"
    # Usually means sum h_i (1-h)_{i+k}.
    
    # Let's compute all overlaps via FFT.
    # H = fft(h)
    # G = fft(g)
    # Prod = H * conj(G)
    # Corrs = ifft(Prod)
    # Corrs[k] = sum_i h_i g_{i-k}
    # We want Overlaps[k] = sum_i h_i g_{i+k}
    # Note that Overlaps[k] = Corrs[-k] (circular) = Corrs[N-k] for k>0.
    # And since h,g real, Corrs is real and symmetric?
    # Corrs[k] = sum h_i g_{i-k}
    # Corrs[N-k] = sum h_i g_{i-(N-k)} = sum h_i g_{i+k}
    # Yes.
    # So the set of overlaps is just the set of values in Corrs.
    # The max overlap is max(Corrs).
    
    H = np.fft.fft(h)
    G = np.fft.fft(g)
    Corrs = np.fft.ifft(H * np.conj(G)).real
    return np.max(Corrs)

def _soft_max(obj, beta=10.0):
    """
    Compute a smooth approximation of the maximum of a vector.
    """
    # To avoid overflow
    m = np.max(obj)
    if m == -np.inf:
        return 0.0
    return m + np.log(np.sum(np.exp((obj - m) * beta))) / beta

def _gradient_soft_max(h, beta=10.0):
    """
    Compute the gradient of the soft max of the overlaps w.r.t. h.
    Overlaps_k = sum_i h_i g_{i+k} where g = 1-h.
    Overlaps_k = sum_i h_i (1-h)_{i+k} = sum_i h_i - sum_i h_i h_{i+k}
    Let S = sum h_i = N/2 (constant).
    Overlaps_k = S - sum_i h_i h_{i+k}
    Let Q_k = sum_i h_i h_{i+k} (circular auto-correlation of h with shift k).
    Objective = max_k (S - Q_k) = S - min_k Q_k.
    Minimizing max overlap is equivalent to maximizing min Q_k.
    Or minimizing soft_max(Overlaps) = soft_max(S - Q).
    
    Let's stick to the previous formulation for gradient.
    Overlaps = S - Q.
    SoftMax(Overlaps) = SoftMax(S - Q).
    Gradient w.r.t h.
    
    Alternatively, just compute the overlaps, get weights, and backprop.
    """
    g = 1.0 - h
    H = np.fft.fft(h)
    G = np.fft.fft(g)
    # Overlaps_k = sum_i h_i g_{i+k}
    # As established, if Corrs = ifft(H * conj(G)), then Corrs[k] = sum_i h_i g_{i-k}.
    # So Overlaps (shift k) = Corrs[N-k].
    # Let's compute Corrs.
    Corrs = np.fft.ifft(H * np.conj(G)).real
    # Overlaps is just a permutation of Corrs.
    # Specifically Overlaps_k = Corrs[-k % N]
    # The set of values is the same. The max is the same.
    # The soft max is the same.
    # The gradient:
    # L = soft_max(Overlaps)
    # dL/dh_i = sum_k w_k (d Overlaps_k / d h_i)
    # Overlaps_k = sum_j h_j g_{j+k} = sum_j h_j (1 - h_{j+k})
    # d Overlaps_k / d h_i = g_{i+k} - h_{i+k}
    # Wait, derivative of h_j (1-h_{j+k}) w.r.t h_i.
    # If j == i, term is (1 - h_{i+k}).
    # If j+k == i, i.e., j = i-k, term is -h_{i-k}.
    # So d Overlaps_k / d h_i = (1 - h_{i+k}) - h_{i-k}
    # = g_{i+k} - h_{i-k}
    
    # We need sum_k w_k * (g_{i+k} - h_{i-k})
    # Let w be the weights from the soft max of Overlaps.
    # Note: w_k corresponds to Overlaps_k.
    # Corrs index mapping: Overlaps_k is at index -k in Corrs?
    # Let's be careful.
    # Corrs[k] = sum_j h_j g_{j-k} = Overlaps_{-k}.
    # So if we compute weights based on Corrs values, say u_k = weight(Corrs[k]).
    # Then u_k is weight for Overlaps_{-k}.
    # We need sum_k weight(Overlaps_k) * grad_k.
    # Let v_k = weight(Overlaps_k).
    # Then v_k = u_{-k} = u_{N-k} (for k!=0).
    # So we need to permute the weights.
    
    # 1. Compute Corrs
    # 2. Compute weights u = softmax(beta * Corrs)
    # 3. Permute u to get v. v[k] = u[N-k % N]. Note v[0]=u[0].
    # 4. Compute grad = sum_k v_k * (g_{i+k} - h_{i-k})
    #    This is a convolution/correlation.
    #    Term 1: sum_k v_k g_{i+k} = (v * g)_i ?
    #    (v * g)_i = sum_k v_k g_{i-k}.
    #    We have g_{i+k}. Let j = -k. sum_j v_{-j} g_{i-j}.
    #    If v is symmetric, this is v * g.
    #    Is v symmetric?
    #    Corrs[k] = Overlaps_{-k}.
    #    Overlaps_{-k} = sum h_j g_{j-k} = sum h_j g_{j+k} = Overlaps_k?
    #    Yes, Overlaps is symmetric (auto-correlation like property for real h, g? No, cross).
    #    Overlaps_k = <h, shift_k g>.
    #    Overlaps_{-k} = <h, shift_{-k} g> = <shift_k h, g> = <g, shift_{-k} h>?
    #    For real vectors, <h, shift_k g> = <shift_{-k} h, g> = <g, shift_k h>.
    #    Also Overlaps_k = sum h_i g_{i+k}.
    #    Overlaps_{-k} = sum h_i g_{i-k}.
    #    Are they equal?
    #    sum h_i g_{i+k} = sum h_{j-k} g_j.
    #    sum h_i g_{i-k} = sum h_{j+k} g_j.
    #    Generally not equal unless h and g are symmetric.
    #    So Corrs is not symmetric.
    #    So v is not symmetric.
    
    # We need to compute:
    # Grad_i = sum_k v_k (g_{i+k} - h_{i-k})
    # = sum_k v_k g_{i+k} - sum_k v_k h_{i-k}
    # Term 2: sum_k v_k h_{i-k} = (v * h)_i. This is standard convolution.
    # Term 1: sum_k v_k g_{i+k}.
    # Let w = v reversed? w_j = v_{-j}.
    # Then sum_k v_k g_{i+k} = sum_k w_{-k} g_{i+k}. Let m = -k.
    # = sum_m w_m g_{i-m} = (w * g)_i.
    # So we need to reverse v to get w.
    
    # Steps:
    # 1. Corrs = ifft(fft(h) * conj(fft(g))).real
    # 2. u = softmax(beta * Corrs)
    # 3. v = u[[-0, -1, -2...]]?
    #    v[k] = u[N-k % N].
    #    v[0] = u[0]
    #    v[1] = u[N-1]
    #    v[2] = u[N-2]
    #    So v = u[0] concat u[N-1::-1]
    # 4. w = v reversed?
    #    w[j] = v[-j] = v[N-j]
    #    w[0] = v[0] = u[0]
    #    w[1] = v[N-1] = u[1]
    #    w[2] = v[N-2] = u[2]
    #    So w = u?
    #    Let's re-verify.
    #    v[k] = weight of Overlaps_k.
    #    Overlaps_k = Corrs[-k].
    #    So v[k] = u[-k].
    #    Grad_i = sum_k v_k (g_{i+k} - h_{i-k})
    #    = sum_k u[-k] g_{i+k} - sum_k u[-k] h_{i-k}
    #    Let m = -k.
    #    Term 1: sum_m u[m] g_{i-m} = (u * g)_i
    #    Term 2: sum_m u[m] h_{i-m} = (u * h)_i
    #    So Grad = u * g - u * h = u * (g - h) = u * (1 - 2h).
    #    
    #    This is beautiful! The gradient is simply the convolution of the weights u (from Corrs) with (1 - 2h).
    
    # 1. Compute Corrs
    # 2. Compute u = softmax(beta * Corrs)
    # 3. Compute diff = 1 - 2h
    # 4. Compute Grad = ifft(fft(u) * fft(diff)).real
    #    Wait, convolution in time domain is multiplication in freq domain.
    #    (u * diff)_i = sum_k u_k diff_{i-k}
    #    This matches the derivation.
    
    # Check scaling.
    # L = soft_max(Overlaps)
    # dL/dh_i = sum_k (e^{beta O_k} / sum e^{beta O_j}) * dO_k/dh_i
    # This is exactly the softmax weights.
    # The FFT convolution computes sum_k u_k diff_{i-k}.
    # Our derivation: Grad_i = sum_m u_m (g_{i-m} - h_{i-m}).
    # diff_j = g_j - h_j = 1 - 2h_j.
    # So yes.
    
    # One subtlety: The "shift" direction in Overlaps_k = sum h_i g_{i+k}.
    # Corrs[k] = sum h_i g_{i-k} = Overlaps_{-k}.
    # So u_k is the weight for Overlaps_{-k}.
    # In the gradient sum, we summed over k (Overlaps index).
    # dL = sum_k u_{Overlaps, k} dO_k
    # u_{Overlaps, k} = u_Corrs, -k.
    # So the vector of weights in the Overlaps basis is v[k] = u[-k].
    # And we derived Grad = v * (g-h) ?
    # Let's re-check Term 1 with v.
    # Term 1 = sum_k v_k g_{i+k}
    # v_k = u_{-k}
    # = sum_k u_{-k} g_{i+k}
    # Let m = -k.
    # = sum_m u_m g_{i-m}
    # This is (u * g)_i.
    # Term 2 = sum_k v_k h_{i-k}
    # = sum_k u_{-k} h_{i-k}
    # Let m = -k.
    # = sum_m u_m h_{i-m}
    # This is (u * h)_i.
    # So Grad = u * g - u * h.
    # Correct.
    
    H = np.fft.fft(h)
    G = np.fft.fft(g)
    Corrs = np.fft.ifft(H * np.conj(G)).real
    
    # Compute weights
    m = np.max(Corrs)
    exps = np.exp(beta * (Corrs - m))
    sum_exps = np.sum(exps)
    u = exps / sum_exps
    
    diff = 1.0 - 2.0 * h
    U = np.fft.fft(u)
    D = np.fft.fft(diff)
    Grad = np.fft.ifft(U * D).real
    
    return Grad, _soft_max(Overlaps_from_Corrs(Corrs), beta)

def Overlaps_from_Corrs(Corrs):
    # The values are the same, just permuted.
    # For soft max, the value is the same.
    return Corrs

def _project_to_simplex(v, target_sum):
    """
    Project v onto {x | 0 <= x <= 1, sum(x) = target_sum}.
    Simplex projection with bounds.
    """
    n = len(v)
    if target_sum <= 0:
        return np.zeros(n)
    if target_sum >= n:
        return np.ones(n)
    
    # Use the algorithm for simplex projection
    # 1. Sort v in descending order
    # 2. Find the largest k such that v[k] + (target_sum - sum(v[:k]))/k > 0
    # 3. Set the first k elements to v[i] + (target_sum - sum(v[:k]))/k
    # 4. Set the rest to 0
    # Then clip to [0,1].
    # Actually, standard simplex projection assumes sum=1 and x>=0.
    # Here we have sum=target_sum and 0<=x<=1.
    # This is equivalent to projecting y = v/target_sum onto simplex sum=1, then scaling?
    # No, bounds are also scaled.
    
    # Efficient algorithm:
    # 1. Sort v descending.
    # 2. Compute prefix sums.
    # 3. Find k such that v[k-1] > (target_sum - prefix[k-1])/k >= v[k] (with 1-based indexing care).
    # Let's use the standard method.
    
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0][-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    # Clip to 1
    w = np.clip(w, 0, 1)
    # Adjust for clipping to satisfy sum exactly
    # If sum(w) != target_sum, we need to adjust.
    # Usually projection onto intersection of simplex and box is more complex.
    # For small N, we can do a few iterations of projection or just use the above and hope it's close.
    # Or use scipy.optimize.linprog? Too slow.
    # Let's refine:
    # The formula above is for x >= 0. The clipping to 1 might break the sum.
    # We can iterate:
    # 1. Project to x >= 0, sum = S.
    # 2. Clip to 1.
    # 3. If sum > S, reduce large elements?
    # If sum < S, increase small elements?
    
    # Better:
    # The set is {x | 0 <= x <= 1, sum x = S}.
    # This is a polytope.
    # For N=100, we can just use the simple projection and then fix the sum.
    
    # Let's try a different approach:
    # Start with w from the non-negative projection.
    # If sum(w) > S:
    #    We have too much mass. We must reduce some x_i.
    #    The ones clipped at 1 are "full". The others are "partial".
    #    We should reduce the partial ones?
    #    Actually, the projection is the closest point in Euclidean distance.
    #    
    # Let's just use the standard simplex projection and then clip, and hope the error is small.
    # Or, since N=100, we can solve for theta again considering the clipped values?
    
    # Let's try a robust method:
    # 1. Get w0 = proj_simplex(v, S) (assuming 0<=w)
    # 2. If any w0[i] > 1, set w0[i] = 1. Let I be indices where clipped.
    # 3. Remaining mass S_rem = S - sum(w0[I]).
    # 4. Project v[~I] to simplex with sum S_rem and lower bound 0, upper bound 1.
    #    Repeat until convergence.
    
    w = np.zeros(n)
    active = np.ones(n, dtype=bool)
    v_active = v.copy()
    S_active = target_sum
    
    for _ in range(n): # Max n iterations
        if not np.any(active):
            break
        v_act = v_active[active]
        S_act = S_active
        
        # Project v_act to simplex sum=S_act, x>=0
        # Reuse the logic
        u = np.sort(v_act)[::-1]
        cssv = np.cumsum(u)
        rho_arr = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - S_act))[0]
        if len(rho_arr) == 0:
             rho = -1
        else:
             rho = rho_arr[-1]
        
        if rho < 0:
            theta = 0
        else:
            theta = (cssv[rho] - S_act) / (rho + 1.0)
            
        w_act = np.maximum(v_act - theta, 0)
        
        # Check for upper bound violation
        viol = w_act > 1.0
        if np.any(viol):
            # Set these to 1
            w[active] = w_act
            # Identify which indices in 'active' correspond to 'viol'
            idx_active = np.where(active)[0]
            idx_viol = idx_active[viol]
            w[idx_viol] = 1.0
            active[idx_viol] = False
            S_active -= np.sum(w_act[viol]) # Subtract the mass we assigned (which is 1 for each)
            # Actually, we assigned 1 to each. So subtract count of viol.
            # Wait, w_act[viol] might be > 1. But we are setting them to 1.
            # So we subtract 1 for each.
            S_active -= np.sum(1.0 * viol) # No, sum of ones.
            S_active -= np.count_nonzero(viol)
            v_active[idx_viol] = 0 # Not used anymore
        else:
            w[active] = w_act
            break
            
    # Final check
    # If sum is not exactly S due to floating point, adjust.
    diff = target_sum - np.sum(w)
    if abs(diff) > 1e-9:
        # Distribute diff among non-clipped elements
        non_clipped = (w > 0) & (w < 1)
        if np.any(non_clipped):
            count = np.sum(non_clipped)
            w[non_clipped] += diff / count
            w = np.clip(w, 0, 1)
            
    return w

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    # Multi-restart gradient descent on soft max surrogate
    
    num_restarts = 20
    max_iter = 500
    beta = 10.0
    
    for r in range(num_restarts):
        if time.time() - _START_TIME > TIME_LIMIT - 10:
            break
            
        # Random initialization
        h = np.random.uniform(0, 1, N)
        # Project to sum N/2
        h = _project_to_simplex(h, N / 2.0)
        
        current_score = _compute_max_overlap(h)
        
        # Gradient Descent
        lr = 0.05
        for i in range(max_iter):
            if time.time() - _START_TIME > TIME_LIMIT - 5:
                break
                
            grad, soft_obj = _gradient_soft_max(h, beta)
            
            # Update
            h_new = h - lr * grad
            h_new = _project_to_simplex(h_new, N / 2.0)
            
            # Check improvement
            score_new = _compute_max_overlap(h_new)
            
            if score_new < current_score:
                h = h_new
                current_score = score_new
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
            else:
                # Reduce learning rate
                lr *= 0.9
                if lr < 1e-6:
                    break
                    
    return best_h
