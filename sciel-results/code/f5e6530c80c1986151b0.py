"""
Strategy: Discrete Coordinate Descent on Binary Sequences
The task asks for "signed step heights". While continuous values might be allowed, 
binary sequences ($\pm 1$) are a strong candidate for "signed" step functions in combinatorial contexts. 
Even if continuous is allowed, binary sequences often provide good baselines. 
The previous approach used continuous gradient descent on an L2 proxy. 
This approach uses a purely discrete search: we start with a random binary sequence 
and repeatedly flip the bit that most reduces the objective function 
$max(|A|) / S^2$. Since $S = \sum x_i$ is constant for a fixed number of +1s and -1s 
(if we flip, S changes by 2 or -2, but we can just recompute), 
and $A$ is the autoconvolution, we can evaluate the effect of flipping any bit $i$ 
by updating the FFT efficiently or just recomputing the score for candidate flips. 
Given N=400, recomputing FFT for each of 400 candidates per step is $400 \times O(N \log N)$. 
One full pass over all bits is $N \times N \log N \approx 400 \times 400 \times 9 \approx 1.4 \times 10^6$ ops. 
This is fast. We can do many passes.
"""

import numpy as np
from numpy.fft import fft, ifft
import time

N = 400

def _autoconv(x):
    """Compute linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    F = fft(x, size2)
    A = ifft(F * np.conj(F)).real
    return A[:size]

def _score(x):
    """Compute max(|autoconv|) / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    A = _autoconv(x)
    m = np.max(np.abs(A))
    return m / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    # Initialize with a random binary sequence
    # We want to avoid S=0. A random sequence has S ~ sqrt(N).
    x = np.random.choice([-1, 1], size=N).astype(float)
    
    best_x = x.copy()
    best_score = _score(best_x)
    
    # Coordinate Descent
    # We iterate through indices. For each index i, we check if flipping x[i] improves the score.
    # To speed this up, we can precompute the current autoconvolution A and sum S.
    # When we flip x[i] from v to -v, the new sum S' = S - 2v.
    # The new autoconvolution A' = A + 2v * (x shifted by i) + 2v * (x shifted by -i) + 4v^2 * delta_i
    # Actually, it's easier to just recompute FFT if we do it smartly, or just recompute the score 
    # if the number of flips per pass is small. 
    # However, N=400 is small. Let's try a simple local search.
    
    # Optimization: 
    # The objective is f(x) = max_k |A_k| / S^2.
    # This is non-smooth. 
    # Let's use a stochastic local search or just a greedy coordinate descent.
    
    # To make it faster, we can evaluate the effect of a flip more cheaply?
    # A_k = sum_j x_j x_{k-j}.
    # If we flip x_i to -x_i:
    # A'_k = A_k - 2 x_i (x_{k-i} + x_{i-k}) + 4 x_i^2 delta_{k,i}
    # Note x_i^2 = 1.
    # So A'_k = A_k - 2 x_i (x_{k-i} + x_{i-k}) + 4 delta_{k,i}.
    # We can update A in O(N) for a flip.
    # Then we need to find max |A'_k|. That is O(N).
    # So one flip evaluation is O(N).
    # One full pass over N indices is O(N^2) = 160,000 ops.
    # This is very fast. We can do thousands of passes.
    
    def _flip_cost(x, A, S, i):
        """
        Estimate the score if we flip x[i].
        x: current array
        A: current autoconvolution array (length 2N-1)
        S: current sum
        i: index to flip
        Returns: new score
        """
        v = x[i]
        # S'
        S_new = S - 2 * v
        
        # A' update
        # A_k' = A_k - 2 v (x_{k-i} + x_{i-k}) + 4 delta_{k,i}
        # We need to handle indices carefully for linear convolution.
        # x is length N. A is length 2N-1.
        # x_j is defined for 0 <= j < N.
        # Term x_{k-i}: valid if 0 <= k-i < N.
        # Term x_{i-k}: valid if 0 <= i-k < N.
        
        # We can't easily update the whole A array in O(N) without vectorized ops that are messy.
        # Given N=400, O(N^2) per pass is fine, but O(N) per flip check inside a loop is also fine.
        # Let's just recompute the score for the candidate flip using the updated A?
        # No, recomputing A from scratch is O(N log N). 
        # Updating A is O(N). 
        # Let's implement the O(N) update.
        
        A_new = A.copy()
        
        # We need to subtract 2*v*x_{k-i} and 2*v*x_{i-k} and add 4*delta.
        # This is effectively adding a shifted version of x.
        # Let's construct the delta_A array.
        # delta_A[k] = -2*v*x_{k-i} - 2*v*x_{i-k} + 4*delta_{k,i}
        
        # Vectorized approach:
        # Create an array d of zeros of length 2N-1.
        # d[k] gets contribution from x_{k-i} if valid.
        # This is just shifting x.
        
        # x_{k-i}: index j = k-i. 0 <= j < N => i <= k < i+N.
        # x_{i-k}: index j = i-k. 0 <= j < N => k <= i and i-k < N => i-N < k <= i.
        
        # Let's just do it with numpy slicing.
        # Part 1: -2*v * x_{k-i}
        # k ranges from i to i+N-1.
        # x_{k-i} corresponds to x[0:N].
        # So d[i : i+N] -= 2*v * x
        
        # Part 2: -2*v * x_{i-k}
        # k ranges from max(0, i-N+1) to i.
        # x_{i-k} corresponds to x[i-k].
        # Let j = i-k. When k=0, j=i. When k=i, j=0.
        # So x[i-k] is x reversed? No.
        # x_{i-k} for k=0 is x_i. For k=i is x_0.
        # So the sequence is x[i], x[i-1], ..., x[0].
        # This is x[i::-1].
        # The length is i+1.
        # So d[0 : i+1] -= 2*v * x[i::-1]
        
        # Part 3: +4*delta_{k,i}
        # d[i] += 4
        
        d = np.zeros_like(A)
        
        # Apply Part 1
        start1 = i
        end1 = i + N
        # Ensure bounds
        s1 = max(0, start1)
        e1 = min(len(d), end1)
        if s1 < e1:
            # x indices: k - i. k goes from s1 to e1-1.
            # x_idx starts at s1 - i, ends at e1 - 1 - i.
            x_start = s1 - i
            x_end = e1 - i
            d[s1:e1] -= 2 * v * x[x_start:x_end]
            
        # Apply Part 2
        start2 = 0
        end2 = i + 1
        s2 = max(0, start2)
        e2 = min(len(d), end2)
        if s2 < e2:
            # x indices: i - k. k goes from s2 to e2-1.
            # x_idx = i - k.
            # When k=s2, x_idx = i - s2.
            # When k=e2-1, x_idx = i - (e2-1).
            # We need x[i-s2 : i-(e2-1)-1 : -1]? 
            # It's easier to just loop or use a reversed slice carefully.
            # x[i-k] for k in [s2, e2-1] is x[i-s2], x[i-s2-1], ...
            # So it's x[i-s2 : i-e2+1 : -1]
            x_idx_start = i - s2
            x_idx_end = i - (e2 - 1)
            # Slice x[x_idx_start : x_idx_end - 1 : -1] ? 
            # If x_idx_end is 0, we want x[x_idx_start : -1 : -1].
            # If x_idx_end > 0, we want x[x_idx_start : x_idx_end - 1 : -1].
            # Actually, just constructing the array is safer.
            ks = np.arange(s2, e2)
            xs = x[i - ks]
            d[s2:e2] -= 2 * v * xs
            
        # Apply Part 3
        d[i] += 4.0
        
        A_new += d
        
        m_new = np.max(np.abs(A_new))
        
        if abs(S_new) < 1e-12:
            return np.inf, A_new, S_new
            
        score_new = m_new / (S_new * S_new)
        
        return score_new, A_new, S_new

    # Main Loop
    # We will perform multiple passes.
    # In each pass, we try to flip each index.
    # If a flip improves the score, we accept it and update x, A, S.
    
    A = _autoconv(x)
    S = np.sum(x)
    
    improved = True
    pass_count = 0
    
    while improved and time.time() < deadline:
        improved = False
        pass_count += 1
        
        # Randomize order to avoid local minima traps? 
        # Or just fixed order. Fixed order is faster.
        # Let's use fixed order first.
        
        for i in range(N):
            if time.time() > deadline - 5:
                break
                
            sc_new, A_new, S_new = _flip_cost(x, A, S, i)
            
            if sc_new < best_score:
                # Accept flip
                x[i] *= -1
                A = A_new
                S = S_new
                best_score = sc_new
                best_x = x.copy()
                improved = True
            elif sc_new < _score(x): # This is redundant if we track best_score correctly, 
                                     # but _score(x) is current. 
                                     # Actually, we should compare to current score, not best score, 
                                     # to allow non-monotonic paths? No, greedy is safer.
                                     # Greedy: accept if better than current.
                                     # But we stored best_x. 
                                     # Let's stick to greedy: accept if better than current state's score.
                                     # Current score is best_score only if we always update best_score on improvement.
                                     # Yes.
                                     pass
        
        # If no improvement in a full pass, we might be stuck.
        # We can try a random restart or a few random flips to escape.
        if not improved:
            # Try 10 random flips to escape local minimum
            escaped = False
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                i = np.random.randint(N)
                sc_new, A_new, S_new = _flip_cost(x, A, S, i)
                if sc_new < best_score:
                    x[i] *= -1
                    A = A_new
                    S = S_new
                    best_score = sc_new
                    best_x = x.copy()
                    escaped = True
                    improved = True
                    break
            if not escaped:
                # Break if stuck and no time left for restarts?
                # Or just break.
                break

    # Final check
    # The evaluator measures raw value.
    # Ensure we return the best found.
    return best_x

# EVOLVE-BLOCK-END
