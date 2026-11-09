import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv(f):
    """Computes the full linear autoconvolution of f."""
    return np.convolve(f, f, mode='full')

def _compute_ratio(f):
    """Compute max|ac| / (sum f)^2"""
    ac = _autoconv(f)
    s = np.sum(f)
    if abs(s) < 1e-10:
        return np.inf
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_abs_ac(f, beta):
    """
    Smooth approximation of max |ac| using log-sum-exp.
    Returns: (soft_max_val, gradient w.r.t f)
    """
    ac = _autoconv(f)
    abs_ac = np.abs(ac)
    
    # To avoid overflow in log-sum-exp, subtract max
    max_ac = np.max(abs_ac)
    if max_ac == 0:
        return 0.0, np.zeros_like(f)
    
    # soft_max = max + (1/beta) * log(sum(exp(beta * (abs_ac - max))))
    # We want gradient of this.
    # Let u = beta * (abs_ac - max)
    # sum_u = sum(exp(u))
    # soft_max = max + (1/beta) * log(sum_u)
    # d soft_max / df_i = (1/beta) * (1/sum_u) * sum_j (exp(u_j) * d abs_ac[j] / df_i)
    
    u = beta * (abs_ac - max_ac)
    # Clip u to prevent overflow if beta is large, though max_ac subtraction helps.
    u = np.clip(u, -50, 50)
    exp_u = np.exp(u)
    sum_u = np.sum(exp_u)
    
    soft_max_val = max_ac + (1.0 / beta) * np.log(sum_u)
    
    # Gradient computation
    # d abs_ac[j] / df_i = sign(ac[j]) * d ac[j] / df_i
    # d ac[j] / df_i = 2 * f[j-i] if 0 <= j-i < N else 0
    
    # We need: grad_i = (1/beta) * (1/sum_u) * sum_j (exp_u[j] * sign(ac[j]) * 2 * f[j-i] * I(0<=j-i<N))
    
    # Let's compute this efficiently.
    # For each i, grad_i = (2 * sgn_sum_i) / (beta * sum_u)
    # where sgn_sum_i = sum_j (exp_u[j] * sign(ac[j]) * f[j-i] * I(0<=j-i<N))
    
    # Note that the sum over j is effectively a correlation/convolution between the weighted sign vector and f.
    # Let w[j] = exp_u[j] * sign(ac[j])
    # Then sgn_sum_i = sum_j w[j] * f[j-i] * I(0<=j-i<N)
    
    # This is the cross-correlation of w and f?
    # Let's check indices.
    # Term is non-zero if j-i = k where 0<=k<N. So j = i+k.
    # sgn_sum_i = sum_{k=0}^{N-1} w[i+k] * f[k] * I(0<=i+k<2N-1)
    
    # This looks like a cross-correlation.
    # np.correlate(w, f, mode='full')?
    # np.correlate(a, v) computes sum_k a[k] * conj(v[k+i])?
    # Documentation: c_k = sum_n a_{n+k} * conj(v_n)
    # So c_k = sum_n w[n+k] * f[n] (assuming real)
    # This matches sgn_sum_i if we identify i with k.
    # But the range of k in correlate is different.
    # np.correlate(w, f, mode='full') has length N_w + N_f - 1 = (2N-1) + N - 1 = 3N-2.
    # We need values for i in [0, N-1].
    
    # Let's verify with small example.
    # N=2. f=[a,b]. ac=[a^2, 2ab, b^2].
    # i=0: sgn_sum_0 = w[0]f[0] + w[1]f[1] (j=0,1 valid? j-i=k. k=0->j=0, k=1->j=1. Yes.)
    #       = exp_u[0]sgn(ac0)a + exp_u[1]sgn(ac1)b
    # i=1: sgn_sum_1 = w[1]f[0] + w[2]f[1] (j=1,2 valid? k=0->j=1, k=1->j=2. Yes.)
    #       = exp_u[1]sgn(ac1)a + exp_u[2]sgn(ac2)b
    
    # Let's use np.correlate(w, f, mode='full')
    # w length 2N-1. f length N.
    # Result length 3N-2.
    # Index k in result corresponds to shift.
    # c_k = sum_n w[n+k] f[n]
    # We want sgn_sum_i = sum_n w[n+i] f[n] (where n is k in my previous derivation? No, n is index of f)
    # My previous derivation: sgn_sum_i = sum_{k=0}^{N-1} w[i+k] f[k]
    # This is exactly c_i from np.correlate(w, f, mode='full') if the indexing aligns.
    # Let's check np.correlate definition again.
    # c_k = sum_n a_{n+k} conj(v_n)
    # So c_i = sum_n w[n+i] f[n].
    # This matches.
    
    # However, we need to be careful with the valid range of n.
    # f[n] is 0 for n<0 or n>=N.
    # w[n+i] is 0 for n+i < 0 or n+i >= 2N-1.
    # The sum in np.correlate handles this naturally if we pad or if the function handles boundaries.
    # np.correlate does NOT pad by default? It computes full correlation.
    # Actually, np.correlate(a, v, mode='full') computes:
    # c_k = sum_n a_{n+k} v_n for k such that the sum is defined.
    # The output length is len(a) + len(v) - 1.
    # The indices of the output correspond to k from -(len(v)-1) to len(a)-1.
    # Wait, the order might be reversed or shifted.
    
    # Let's stick to a safer, slightly less optimized but correct approach for gradient.
    # Or use FFT.
    # Let's try FFT approach for the weighted sum.
    # sgn_sum = ifft(fft(w) * conj(fft(f_padded)))?
    # Correlation theorem: corr(w, f) = ifft(fft(w) * conj(fft(f)))
    # We need to pad to length at least (2N-1) + N - 1 = 3N-2 to avoid circular aliasing.
    
    size = 1 << (len(w) + len(f) - 1).bit_length()
    W = np.fft.fft(w, size)
    F = np.fft.fft(f, size)
    corr = np.fft.ifft(W * np.conj(F)).real
    
    # Now we need to map the indices of 'corr' to our 'i' in [0, N-1].
    # In linear correlation c_k = sum_n w[n+k] f[n].
    # The output of ifft(fft(w)*conj(fft(f))) gives circular correlation.
    # With zero padding, it gives linear correlation.
    # The linear correlation sequence c_k for k = -(N-1) ... (2N-2).
    # The ifft output is arranged as c_{N-1}, c_N, ..., c_{2N-2}, c_0, c_{-(N-1)}, ..., c_{-1}?
    # Or c_{-(N-1)}, ..., c_0, ..., c_{2N-2}?
    # Usually, the zero index is at the start.
    # So index 0 corresponds to k=0?
    # Let's check np.correlate behavior.
    # np.correlate([1,2], [3,4], 'full') -> [4, 11, 6]
    # c_{-1} = 1*4 = 4
    # c_0 = 1*3 + 2*4 = 11
    # c_1 = 2*3 = 6
    # So the output is [c_{-(len(v)-1)}, ..., c_0, ..., c_{len(a)-1}]
    
    # In our case, len(w) = 2N-1, len(f) = N.
    # Output length = 3N-2.
    # Indices 0 to N-2 correspond to k = -(N-1) to -1.
    # Index N-1 corresponds to k = 0.
    # Index N-1 + i corresponds to k = i.
    
    # We need sgn_sum_i for i in [0, N-1].
    # So we need k = 0 to N-1.
    # These correspond to indices N-1 to N-1 + N-1 = 2N-2 in the corr array.
    
    grad = np.zeros_like(f)
    if sum_u > 0:
        # Extract the relevant part of the correlation
        # corr has length 3N-2
        # We want indices N-1 to 2N-2
        start_idx = N - 1
        end_idx = 2 * N - 1  # exclusive
        sgn_sum_i = corr[start_idx:end_idx]
        
        # grad_i = (2 / (beta * sum_u)) * sgn_sum_i[i]
        grad[:] = (2.0 / (beta * sum_u)) * sgn_sum_i
        
    return soft_max_val, grad

def _objective_and_grad(f, beta):
    """
    Objective: log(soft_max |ac|) - 2 log(|sum f|)
    Gradient: (1/soft_max) * grad(soft_max) - 2/sum_f * ones
    """
    soft_max_val, grad_soft = _soft_max_abs_ac(f, beta)
    
    s = np.sum(f)
    if s == 0 or soft_max_val == 0:
        return 1e10, np.zeros_like(f)
    
    obj_val = np.log(soft_max_val) - 2 * np.log(np.abs(s))
    
    # d log(soft_max) / df = (1/soft_max) * grad(soft_max)
    g1 = grad_soft / soft_max_val
    
    # d -2 log(|s|) / df = -2/s * ones
    g2 = -2.0 * np.ones_like(f) / s
    
    grad_obj = g1 + g2
    
    return obj_val, grad_obj

def _sign_flip_search(f, max_steps=100):
    """
    Local search: try flipping sign of individual elements or pairs to reduce the ratio.
    """
    current_f = f.copy()
    current_ratio = _compute_ratio(current_f)
    
    for _ in range(max_steps):
        improved = False
        # Try flipping single signs
        # To save time, only try a subset or random subset if N is large?
        # N=400 is small enough to try all single flips in one pass? 400 convolutions is fast.
        for i in range(N):
            new_f = current_f.copy()
            new_f[i] = -new_f[i]
            new_ratio = _compute_ratio(new_f)
            if new_ratio < current_ratio:
                current_f = new_f
                current_ratio = new_ratio
                improved = True
        if not improved:
            break
            
    return current_f, current_ratio

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_f = None
    best_score = np.inf
    
    # Initial guesses
    n_restarts = 5
    np.random.seed(42)
    
    # Use a moderate beta for smoothing.
    # Too small: not smooth enough. Too large: close to max, but numerical issues?
    # max_abs is O(1). beta=10 or 20 should be fine.
    beta = 10.0
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        if r == 0:
            f0 = np.ones(N)
        elif r == 1:
            f0 = np.random.randn(N)
        elif r == 2:
            f0 = np.random.choice([-1.0, 1.0], size=N)
            if np.sum(f0) < 0: f0 = -f0
        elif r == 3:
            # Chirp
            t = np.linspace(0, 10, N)
            f0 = np.exp(1j * t**2)
            f0 = np.real(f0)
        else:
            f0 = np.random.randn(N)
        
        # Optimization
        try:
            # L-BFGS-B is good for this.
            # We need to pass beta to the objective and grad functions.
            # We can use a closure or a wrapper.
            
            def obj_wrapper(f):
                val, _ = _objective_and_grad(f, beta)
                return val
            
            def grad_wrapper(f):
                _, grad = _objective_and_grad(f, beta)
                return grad
            
            res = minimize(
                obj_wrapper, 
                f0, 
                method='L-BFGS-B', 
                jac=grad_wrapper,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            candidate_f = res.x
            
            # Polish with sign flipping
            if time.time() < deadline - 10:
                candidate_f, candidate_ratio = _sign_flip_search(candidate_f)
            else:
                candidate_ratio = _compute_ratio(candidate_f)
            
            if candidate_ratio < best_score:
                best_score = candidate_ratio
                best_f = candidate_f
                
        except Exception as e:
            pass
            
    # If no valid result, return ones
    if best_f is None:
        best_f = np.ones(N)
        
    return best_f
