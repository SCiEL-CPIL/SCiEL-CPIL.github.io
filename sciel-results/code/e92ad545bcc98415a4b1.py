"""
Second autocorrelation inequality: choose 50 nonnegative heights to maximize
the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution.

Strategy:
- The score is S(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf).
- This objective is scale-invariant: S(ah) = S(h) for any a > 0.
- We can fix ||h||_1 = 1 without loss of generality. Then S(h) = ||h||_2^2 / ||h*h||_inf.
- Let g = ||h||_2^2 and d = ||h*h||_inf. We want to maximize g/d.
- We perform gradient ascent on log(g) - log(d) = 2*log(g) - log(d) subject to h >= 0 and sum(h) = 1.
- The gradient of g = sum h_i^2 is 2*h.
- The gradient of d = max_k conv[k] is a vector that is 1 at the index of the max conv value and 0 elsewhere.
  Specifically, if conv = h*h, then d = max(conv). The subgradient of d with respect to h is:
  grad_d[h] = 2 * (e_m * h) where m is the index of the max in conv, and * is convolution.
  Wait, conv[k] = sum_i h[i] h[k-i].
  d/dh[j] conv[k] = h[k-j] + h[k-j] = 2 h[k-j] (if k-j is in range, considering symmetry? No, just h[k-j] + h[k-j] is wrong if we treat them as independent, but h is the same vector).
  Actually, conv[k] = sum_i h[i] h[k-i].
  d conv[k] / d h[j] = h[k-j] + (if j is the other index? No, just sum over i. If i=j, term is h[k-j]. If k-i=j => i=k-j, term is h[j]. So derivative is h[k-j] + h[j] if k-j != j, else 2h[j].
  Basically, the gradient of d with respect to h is 2 * (e_m * h), where e_m is a spike at m (the argmax of conv).
  
- We use projected gradient ascent on the simplex.
- We try multiple initializations: spike, uniform, blocks, Gaussians.
"""

import numpy as np
import time

N = 50

def _autoconvolution(h):
    """Compute the full autoconvolution of h with itself."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * N - 1]

def _score(h):
    """Compute the score: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h ** 2)
    conv = _autoconvolution(h)
    l_inf = np.max(np.abs(conv))
    if l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def _project_to_simplex(v):
    """Project vector v onto the probability simplex {x >= 0, sum(x) = 1}."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

def _gradient_step(h, lr):
    """
    Perform one step of projected gradient ascent on log(||h||_2^2) - log(||h*h||_inf).
    h is assumed to be on the simplex.
    """
    l2_sq = np.sum(h ** 2)
    conv = _autoconvolution(h)
    l_inf = np.max(conv)
    
    if l2_sq == 0 or l_inf == 0:
        return h.copy()
    
    # Gradient of log(l2_sq) = 2*h / l2_sq
    grad_g = 2.0 * h / l2_sq
    
    # Gradient of log(l_inf) = grad(l_inf) / l_inf
    # grad(l_inf) = 2 * (e_m * h) where m is argmax(conv)
    m = np.argmax(conv)
    # e_m is a vector of zeros with 1 at m.
    # conv of e_m and h is just h shifted by m?
    # (e_m * h)[k] = sum_i e_m[i] h[k-i] = h[k-m] (if k-m in range).
    # So e_m * h is h shifted.
    # Let's compute g = e_m * h.
    g = np.zeros(N)
    # h[k-m] is non-zero when 0 <= k-m < N => m <= k < m+N.
    # So g[m : m+N] = h.
    start = m
    end = m + N
    # But g has length N? No, the gradient vector must have length N.
    # The derivative of d = conv[m] with respect to h[j] is:
    # d conv[m] / d h[j] = h[m-j] + (if m-j != j? no, just h[m-j] + h[m-j] is wrong.
    # conv[m] = sum_i h[i] h[m-i].
    # d/dh[j] = h[m-j] (from i=j) + h[j] (from m-i=j => i=m-j).
    # So grad_d[j] = h[m-j] + h[j] (with bounds checks).
    # This is exactly 2 * (e_m * h)[j]?
    # (e_m * h)[j] = sum_i e_m[i] h[j-i] = h[j-m].
    # So 2 * (e_m * h)[j] = 2 h[j-m].
    # This is NOT h[m-j] + h[j].
    
    # Let's re-evaluate.
    # conv[k] = sum_i h[i] h[k-i].
    # d conv[m] / d h[j] = h[m-j] + h[j] (if indices valid).
    # This is the convolution of h with (e_m + e_-m)? No.
    # It is (h * (e_m + e_-m))[j]?
    # (h * e_m)[j] = h[j-m].
    # (h * e_-m)[j] = h[j+m].
    # So grad_d = h shifted by -m + h shifted by m.
    
    grad_d = np.zeros(N)
    # Term 1: h[m-j] valid for j in [0, m] (if m < N)
    # Actually j ranges 0..N-1. m-j ranges 0..N-1 => j ranges m-N+1 .. m.
    # Intersection with [0, N-1] is [max(0, m-N+1), min(m, N-1)].
    idx1_start = max(0, m - N + 1)
    idx1_end = min(m, N - 1)
    if idx1_start <= idx1_end:
        # j in [idx1_start, idx1_end]
        # h[m-j]
        # If j=0, h[m]. If j=m, h[0].
        # This is h[m : idx1_start-1 : -1]? No.
        # h[m-j] for j=idx1_start ... idx1_end
        # m-idx1_start ... m-idx1_end
        # This is h[m-idx1_start : m-idx1_end+1 : -1]
        grad_d[idx1_start:idx1_end+1] += h[m-idx1_start : m-idx1_end+1 : -1]
        
    # Term 2: h[j] valid for m-j in [0, N-1] => j in [m-N+1, m].
    # Same range.
    # So we just add h[j] for j in that range.
    if idx1_start <= idx1_end:
        grad_d[idx1_start:idx1_end+1] += h[idx1_start:idx1_end+1]
        
    # Wait, is it h[m-j] + h[j]?
    # Let's test N=3, h=[a,b,c]. m=1.
    # conv[1] = h[0]h[1] + h[1]h[0] + h[2]h[-1](0) = 2ab.
    # d/dh[0] = 2b. Formula: h[1-0]+h[0] = b+a. WRONG.
    # Ah, the derivative of sum_i h[i]h[m-i] with respect to h[j]:
    # Terms where i=j: h[j]h[m-j]. Deriv: h[m-j].
    # Terms where m-i=j => i=m-j: h[m-j]h[j]. Deriv: h[j].
    # If j = m-j (i.e., 2j=m), these are the same term h[j]^2, deriv is 2h[j].
    # In the sum formula, if j != m-j, we get h[m-j] + h[j].
    # If j == m-j, we get h[j] + h[j] = 2h[j].
    # So the formula h[m-j] + h[j] works IF we treat the indices correctly.
    # In my test: m=1, j=0. h[1]+h[0] = b+a. But actual deriv is 2b.
    # Why? conv[1] = h[0]h[1] + h[1]h[0].
    # d/dh[0] = 1*h[1] + h[1]*1 = 2h[1] = 2b.
    # My formula gave h[1]+h[0] = b+a.
    # The error is that h[0]h[1] and h[1]h[0] are the SAME term in the sum if we consider i and m-i.
    # The sum is over i. For m=1, i=0 gives h[0]h[1]. i=1 gives h[1]h[0].
    # These are distinct terms in the summation index i, but they involve the same variables.
    # d/dh[0] of h[0]h[1] is h[1].
    # d/dh[0] of h[1]h[0] is h[1].
    # Total 2h[1].
    # So the derivative is sum over i of d/dh[j] (h[i]h[m-i]).
    # Non-zero if i=j (gives h[m-j]) or m-i=j (gives h[j]).
    # So it IS h[m-j] + h[j].
    # Why did my test fail?
    # h=[a,b,c]. m=1.
    # i=0: h[0]h[1] = ab. d/dh[0] = b.
    # i=1: h[1]h[0] = ba. d/dh[0] = b.
    # Sum = 2b.
    # Formula: h[m-j] + h[j] = h[1] + h[0] = b + a.
    # The formula h[m-j] + h[j] assumes that the term where i=j is h[j]h[m-j] and the term where m-i=j is h[m-j]h[j].
    # If i=j, the term is h[j]h[m-j]. Deriv w.r.t h[j] is h[m-j].
    # If m-i=j, the term is h[m-j]h[j]. Deriv w.r.t h[j] is h[m-j].
    # So it should be h[m-j] + h[m-j] = 2h[m-j]?
    # No, if i=j, the variable is h[j]. The other factor is h[m-j]. Deriv is h[m-j].
    # If m-i=j, i is fixed to m-j. The term is h[m-j]h[j]. Deriv w.r.t h[j] is h[m-j].
    # So both contributions are h[m-j].
    # Total 2h[m-j].
    # UNLESS j = m-j, in which case i=j=m/2. The term is h[j]^2. Deriv is 2h[j].
    # So the gradient is 2 * h[m-j] (with bounds).
    # Let's re-verify.
    # conv[m] = sum_i h[i] h[m-i].
    # d/dh[j] conv[m] = sum_i (delta_{ij} h[m-i] + h[i] delta_{i, m-j}).
    # = h[m-j] (from first term, i=j) + h[m-j] (from second term, i=m-j).
    # = 2 h[m-j].
    # So grad_d[j] = 2 h[m-j] (if m-j is in range).
    
    # My previous calculation of grad_d was wrong. Let's fix it.
    grad_d = np.zeros(N)
    # j such that 0 <= m-j < N => m-N < j <= m.
    # j in [max(0, m-N+1), min(m, N-1)].
    j_start = max(0, m - N + 1)
    j_end = min(m, N - 1)
    if j_start <= j_end:
        # h[m-j] for j in j_start..j_end
        # m-j goes from m-j_start down to m-j_end.
        grad_d[j_start:j_end+1] = 2.0 * h[m-j_start : m-j_end+1 : -1]
        
    grad_log_d = grad_d / l_inf
    
    grad = grad_g - grad_log_d
    
    h_new = h + lr * grad
    h_new = _project_to_simplex(h_new)
    
    return h_new

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_h = None
    best_score = 0.0
    
    # Initializers
    inits = []
    
    # 1. Spike
    h_sp = np.zeros(N); h_sp[N//2] = 1.0
    inits.append(h_sp)
    
    # 2. Uniform
    inits.append(np.ones(N) / N)
    
    # 3. Blocks
    for k in [2, 5, 10, 20]:
        h_b = np.zeros(N)
        h_b[:k] = 1.0 / k
        inits.append(h_b)
        h_b2 = np.zeros(N)
        h_b2[N-k:] = 1.0 / k
        inits.append(h_b2)
        h_b3 = np.zeros(N)
        start = (N-k)//2
        h_b3[start:start+k] = 1.0 / k
        inits.append(h_b3)
        
    # 4. Gaussians
    x = np.arange(N)
    for sigma in [1, 2, 5, 10]:
        h_g = np.exp(-((x - N//2)**2) / (2*sigma**2))
        h_g /= np.sum(h_g)
        inits.append(h_g)
        
    # 5. Geometric
    for r in [0.9, 0.95, 0.99]:
        h_g = r ** np.arange(N)
        h_g /= np.sum(h_g)
        inits.append(h_g)
        h_g2 = r ** (N-1-np.arange(N))
        h_g2 /= np.sum(h_g2)
        inits.append(h_g2)

    # Run gradient ascent from each init
    for h0 in inits:
        if time.time() > deadline - 10:
            break
            
        h = h0.copy()
        # Adaptive learning rate
        lr = 0.01
        for step in range(2000):
            if time.time() > deadline - 5:
                break
                
            h_new = _gradient_step(h, lr)
            s_new = _score(h_new)
            s_old = _score(h)
            
            if s_new > s_old:
                h = h_new
                if s_new > best_score:
                    best_score = s_new
                    best_h = h.copy()
                # Increase lr slightly if improving
                lr *= 1.05
            else:
                # Decrease lr
                lr *= 0.5
                # If lr is too small, break
                if lr < 1e-6:
                    break
                    
    # Final polish with small steps
    if best_h is not None:
        h = best_h.copy()
        lr = 1e-4
        for step in range(500):
            if time.time() > deadline - 1:
                break
            h_new = _gradient_step(h, lr)
            s_new = _score(h_new)
            s_old = _score(h)
            if s_new > s_old:
                h = h_new
                if s_new > best_score:
                    best_score = s_new
                    best_h = h.copy()
            else:
                lr *= 0.5

    if best_h is None:
        return list(np.ones(N) / N)
        
    return list(best_h)

if __name__ == "__main__":
    # Quick test
    uniform = np.ones(N) / N
    print("Uniform score:", _score(uniform))
    
    spike = np.zeros(N)
    spike[N//2] = 1.0
    print("Spike score:", _score(spike))
    
    result = construct()
    print("Best score:", _score(np.array(result)))
    print("Result sum:", sum(result))
