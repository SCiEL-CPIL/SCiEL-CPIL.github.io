import numpy as np
import time

N = 400

def _calc_acf(s):
    """Compute full autocorrelation array A[0..N-1]"""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    P = F * np.conj(F)
    acf = np.fft.irfft(P, n=n_fft)
    return np.real(acf[:N])

def _max_abs_acf_from_array(A):
    return float(np.max(np.abs(A[1:])))

def _flip_update(s, i, A):
    """
    Apply flip of s[i] in-place to both s and A.
    Returns the old value of s[i].
    """
    old_si = s[i]
    s[i] = -old_si
    
    # A[k] = sum_{j=0}^{N-1-k} s[j+k]*s[j]
    # After flipping s[i], the change in A[k] is:
    # -2 * old_si * (s[i+k] + s[i-k]) with appropriate bounds
    
    # For each k where the flip affects A[k]:
    # The term s[i] appears in A[k] when:
    #   i = j+k for some j, i.e., j = i-k, valid if 0 <= i-k <= N-1-k
    #   i = j for some j, i.e., j = i, valid if i <= N-1-k
    
    # More directly:
    # A[k] = sum_{j=0}^{N-1-k} s[j+k]*s[j]
    # s[i] contributes when:
    #   j+k = i -> j = i-k, valid if 0 <= i-k and i-k <= N-1-k -> k <= i and k >= i-(N-1-k) -> k >= i-N+1+k -> always true for k>=0 if i-k>=0
    #   j = i -> valid if i <= N-1-k -> k <= N-1-i
    
    # So s[i] appears in A[k] for:
    #   k from 0 to min(i, N-1-i)
    #   Specifically:
    #   - When j = i: k such that i <= N-1-k -> k <= N-1-i
    #   - When j+k = i: k such that i-k >= 0 -> k <= i, and j=i-k <= N-1-k -> i-k <= N-1-k -> i <= N-1 -> always true
    #   
    # Wait, let me think again.
    # A[k] = sum_{j=0}^{N-1-k} s[j+k]*s[j]
    # The index i appears as:
    #   s[j+k] where j+k=i -> j=i-k, valid if 0<=i-k<=N-1-k
    #   s[j] where j=i, valid if 0<=i<=N-1-k
    #
    # For s[j+k] with j=i-k:
    #   0 <= i-k -> k <= i
    #   i-k <= N-1-k -> i <= N-1 -> always true
    #   So k from 0 to i
    #
    # For s[j] with j=i:
    #   0 <= i -> always true
    #   i <= N-1-k -> k <= N-1-i
    #   So k from 0 to N-1-i
    #
    # Combined: k from 0 to min(i, N-1-i)
    # But we need to be careful about double-counting when both conditions are met for the same k.
    # When does s[i] appear twice in A[k]?
    #   When j+k=i AND j=i, which means i+k=i -> k=0.
    #   So for k=0, s[i] appears in s[i]*s[i] (once as s[j+k] and once as s[j], but it's the same product term)
    #   Actually, A[0] = sum_{j=0}^{N-1} s[j]*s[j] = sum s[j]^2 = N, since s[j]=±1.
    #   So A[0] is always N, regardless of flips. The flip doesn't change A[0].
    #
    # Let me reconsider. A[k] = sum_{j=0}^{N-1-k} s[j+k]*s[j]
    # For a given k, which terms involve s[i]?
    #   Term 1: s[j+k] = s[i] -> j = i-k, valid if 0 <= i-k <= N-1-k
    #   Term 2: s[j] = s[i] -> j = i, valid if 0 <= i <= N-1-k
    #
    # For k=0:
    #   Term 1: j=i, s[i]*s[i] = 1
    #   Term 2: j=i, s[i]*s[i] = 1
    #   But these are the same term! A[0] = sum s[j]^2, each s[j] appears once.
    #   So s[i] contributes 1 to A[0], not 2.
    #
    # For k>0:
    #   Term 1: j=i-k, contributes s[i]*s[i-k]
    #   Term 2: j=i, contributes s[i+k]*s[i]
    #   These are different terms (unless i-k=i, which means k=0)
    #
    # So for k>0, s[i] appears in at most 2 terms:
    #   - s[i]*s[i-k] (from j=i-k, requires i-k>=0 -> k<=i)
    #   - s[i+k]*s[i] (from j=i, requires i+k<N -> k<N-1-i)
    #
    # The change when flipping s[i] from old_si to -old_si:
    #   Delta A[k] = -2 * old_si * (contribution from terms involving s[i])
    #   For k>0:
    #     If k <= i: term s[i]*s[i-k] changes by -2*old_si*s[i-k]
    #     If k < N-1-i: term s[i+k]*s[i] changes by -2*old_si*s[i+k]
    #   For k=0: A[0] is invariant, delta=0
    #
    # So:
    #   Delta A[k] = -2 * old_si * (s[i-k] * [k<=i] + s[i+k] * [k<N-1-i]) for k>0
    #   Delta A[0] = 0
    
    # Apply delta to A
    # For k=1 to N-1:
    #   delta_k = -2 * old_si * (s[i-k] if k<=i else 0) + (-2 * old_si * (s[i+k] if k<N-1-i else 0))
    #   = -2 * old_si * (s[i-k] * (k<=i) + s[i+k] * (k<N-1-i))
    
    # Vectorized update for k=1..N-1
    # Create mask for valid indices
    k_vals = np.arange(1, N)
    
    # s[i-k] for k=1..N-1: valid when i-k >= 0 -> k <= i
    # s[i-k] = s[i-1::-1][:N-1] but we need s[i-k] for k=1..N-1
    # s[i-k] for k=1 is s[i-1], for k=2 is s[i-2], ..., for k=i is s[0]
    # So s[i-k] for k=1..min(i, N-1) = s[i-1::-1][:min(i, N-1)]
    
    # Let's construct the delta array
    delta = np.zeros(N)
    
    # Part 1: s[i-k] contribution for k=1..min(i, N-1)
    if i >= 1:
        max_k1 = min(i, N-1)
        # s[i-k] for k=1..max_k1 = s[i-1], s[i-2], ..., s[i-max_k1]
        # This is s[i-1::-1][:max_k1]
        vals1 = s[i-1::-1][:max_k1]
        delta[1:1+max_k1] += -2 * old_si * vals1
    
    # Part 2: s[i+k] contribution for k=1..min(N-1-i, N-1)
    if i < N-1:
        max_k2 = min(N-1-i, N-1)
        # s[i+k] for k=1..max_k2 = s[i+1], s[i+2], ..., s[i+max_k2]
        # This is s[i+1:i+1+max_k2]
        vals2 = s[i+1:i+1+max_k2]
        delta[1:1+max_k2] += -2 * old_si * vals2
    
    A += delta
    return old_si

def _smooth_seed(seed_val=42):
    """
    Generate a high-quality initial binary sequence by minimizing
    the L2 norm of the autocorrelation sidelobes using gradient descent
    on a continuous variable, then binarizing.
    """
    np.random.seed(seed_val)
    # Start with random continuous values in [-1, 1]
    x = np.random.uniform(-1, 1, size=N)
    
    # Gradient descent parameters
    lr = 0.05
    steps = 500
    momentum = 0.9
    v = np.zeros_like(x)
    
    for step in range(steps):
        # Compute AC
        A = _calc_acf(x)
        
        # Objective: sum(A[1:]**2)
        # Gradient of sum(A[k]**2) w.r.t x[i]
        # A[k] = sum_j x[j+k]x[j]
        # d(A[k])/dx[i] = x[i-k] (if k<=i) + x[i+k] (if k<N-1-i)
        # d(Obj)/dx[i] = sum_{k=1}^{N-1} 2*A[k] * (d(A[k])/dx[i])
        
        grad = np.zeros(N)
        
        # We can compute this efficiently or just loop over k for simplicity in seed gen
        # Since N=400 and steps=500, O(N^2 * steps) = 400*400*500 = 80M ops, which is fast enough in Python if vectorized per k, 
        # but even a simple loop over k might be ok. Let's try vectorized accumulation.
        
        # For each k, add 2*A[k] * (x[i-k] + x[i+k]) to grad[i]
        # This is essentially a convolution-like operation.
        # grad = 2 * (A[1:] * x) * something? No, it's not simple convolution because A depends on x.
        # But for a fixed x, A is fixed.
        
        # Let's just compute the gradient term by term.
        # For efficiency, we can precompute the contributions.
        # Actually, let's just use the same logic as _flip_update but for gradient.
        # d(Obj)/dx[i] = 2 * sum_{k=1}^{N-1} A[k] * ( [k<=i]*x[i-k] + [k<N-1-i]*x[i+k] )
        
        # This is equivalent to:
        # For each i, sum over k.
        # Let's swap sums:
        # sum_k A[k] x[i-k] [k<=i] + sum_k A[k] x[i+k] [k<N-1-i]
        
        # Term 1: sum_{k=1}^i A[k] x[i-k]
        # Let j = i-k. k=i-j. Sum_{j=0}^{i-1} A[i-j] x[j]
        # This is a convolution of A and x.
        
        # Term 2: sum_{k=1}^{N-1-i} A[k] x[i+k]
        # Let j = i+k. k=j-i. Sum_{j=i+1}^{N-1} A[j-i] x[j]
        # This is a cross-correlation.
        
        # We can compute these using FFT or direct loops.
        # Given N=400, direct loops in Python might be slow if done naively 500 times.
        # But we can vectorize the gradient calculation per step.
        
        # Let's construct the gradient using vectorized operations.
        # g1[i] = sum_{j=0}^{i-1} A[i-j] x[j]
        # g2[i] = sum_{j=i+1}^{N-1} A[j-i] x[j]
        
        # We can compute g1 and g2 using np.convolve or similar?
        # g1 is like a causal convolution of A and x.
        # g2 is like an anti-causal convolution.
        
        # Let's just use a loop over i for clarity and correctness, as N is small.
        # 400 iterations * 500 steps = 200,000 inner loop iterations.
        # Each inner loop is O(N). Total O(N^2 * steps) = 80M.
        # Python loop might take ~1-2 seconds. Acceptable.
        
        for i in range(N):
            g_val = 0.0
            # k from 1 to min(i, N-1)
            # x[i-k] term
            if i > 0:
                # k=1..i -> x[i-1]..x[0]
                # A[k] -> A[1]..A[i]
                g_val += np.dot(A[1:i+1], x[i-1::-1][:i])
            if i < N-1:
                # k=1..N-1-i -> x[i+1]..x[N-1]
                # A[k] -> A[1]..A[N-1-i]
                g_val += np.dot(A[1:N-i], x[i+1:])
            
            grad[i] = 2.0 * g_val
        
        # Update with momentum
        v = momentum * v - lr * grad
        x = x + v
        
        # Optional: project to sphere or just keep as is
        # Binarization happens at the end
        
    # Binarize
    s = np.sign(x)
    # Handle zeros
    s[s == 0] = 1.0
    
    return s.astype(np.int8)

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin
    
    best_val = float('inf')
    best_seq = np.ones(N, dtype=np.int8)
    
    # List of seeds to try
    # 1. Smooth seed (high quality)
    # 2. Random seeds (fallback/diversity)
    
    seeds_to_try = []
    
    # Generate smooth seeds with different random inits
    for i in range(3):
        seeds_to_try.append(('smooth', i))
        
    # Generate random seeds
    for i in range(10):
        seeds_to_try.append(('random', i))
        
    # Shuffle to mix them up? Or just run smooth first?
    # Running smooth first is better because it's likely to find a good solution quickly.
    
    for seed_type, seed_val in seeds_to_try:
        if time.time() > deadline:
            break
            
        if seed_type == 'smooth':
            s = _smooth_seed(seed_val)
        else:
            np.random.seed(seed_val)
            s = np.random.randint(0, 2, size=N) * 2 - 1
            s = s.astype(np.int8)
            
        A = _calc_acf(s)
        current_val = _max_abs_acf_from_array(A)
        
        # Local search with in-place delta updates
        improved = True
        passes = 0
        max_passes = 100  # Reduced passes for more restarts
        no_improve_count = 0
        max_no_improve = 200
        
        while improved and passes < max_passes and no_improve_count < max_no_improve and time.time() < deadline:
            improved = False
            passes += 1
            no_improve_count = 0
            
            # Random order to avoid bias
            indices = np.random.permutation(N)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                # Try flipping s[i]
                old_si = _flip_update(s, i, A)
                new_val = _max_abs_acf_from_array(A)
                
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                else:
                    # Revert the flip
                    _flip_update(s, i, A)
                    no_improve_count += 1
                    
        if current_val < best_val:
            best_val = current_val
            best_seq = s.copy()
            
        # If we've found a very good solution, we can stop early
        if best_val < 0.02:
            break

    return best_seq
