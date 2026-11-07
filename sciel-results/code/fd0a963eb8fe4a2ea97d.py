"""
Strategy: Discrete Combinatorial Optimization via Simulated Annealing.

Rationale:
The objective function is V(h) = max_k (sum_i h_i (1 - h_{i+k})) / N * 2.
For a fixed lag k, the term C_k(h) = sum_i h_i - sum_i h_i h_{i+k} is a quadratic
pseudo-Boolean function. However, since the objective is the maximum of such terms,
and the feasible region (sum h_i = N/2, 0 <= h_i <= 1) is a convex polytope,
the minimum of a convex function over a convex polytope is attained at a vertex.
The vertices of this polytope (intersection of hypercube and hyperplane) are binary
vectors with exactly N/2 ones.

Thus, we can restrict the search to binary vectors b in {0, 1}^N with sum(b) = N/2.
We use Simulated Annealing to find a binary configuration that minimizes the maximum
correlation peak. The "swap" move preserves the sum constraint automatically.
"""

N = 100
import numpy as np
import time
import math
import random

# EVOLVE-BLOCK-START
def compute_corrs_fast(b):
    """
    Compute the correlation terms for a binary vector b.
    We need max(np.correlate(h, 1 - h, mode="full")).
    For binary h, 1-h is just the bitwise NOT (0s become 1s, 1s become 0s).
    Let z = 1 - b.
    corr[k] = sum_i b[i] * z[i+k] (with appropriate boundary handling for 'full' mode).
    
    Actually, let's use the property:
    C_k = sum_i b_i (1 - b_{i+k}) = sum_i b_i - sum_i b_i b_{i+k}.
    sum_i b_i is constant (N/2).
    So minimizing max C_k is equivalent to maximizing min (sum_i b_i b_{i+k})?
    No, we want to minimize the MAX of C_k.
    C_k = K - D_k, where K = N/2 and D_k = sum_i b_i b_{i+k}.
    Minimizing max (K - D_k) is equivalent to maximizing min D_k.
    
    So we want to find a binary sequence with N/2 ones such that the minimum
    autocorrelation side-lobe (overlap of ones) is maximized.
    This is a known problem related to Golay sequences or Barker sequences, but for
    fixed weight.
    
    We can compute D_k efficiently.
    D_k = sum_{i} b_i b_{i+k}.
    This is the autocorrelation of b at lag k.
    """
    # We want to compute the max of C_k = K - D_k.
    # This is K - min(D_k).
    # We need to be careful with the definition of np.correlate(h, 1-h, 'full').
    # np.correlate(a, v, 'full') at index k (0 to 2N-2) corresponds to lag.
    # Let's just compute the raw correlation as defined by numpy to be safe, 
    # but optimize the inner loop if possible. 
    # For N=100, np.correlate is fast enough (~microseconds).
    
    h = b.astype(float)
    # 1 - h is just 1.0 - h
    corr = np.correlate(h, 1.0 - h, mode="full")
    return corr

def objective_binary(b):
    """
    Compute the evaluation value for a binary vector b.
    """
    corr = compute_corrs_fast(b)
    max_corr = np.max(corr)
    return max_corr / N * 2.0

def construct():
    """
    Uses Simulated Annealing on the binary hypercube vertices.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

    # Initial state: Random binary vector with 50 ones
    def random_binary():
        b = np.zeros(N, dtype=np.int8)
        indices = np.random.choice(N, N // 2, replace=False)
        b[indices] = 1
        return b

    best_b = random_binary()
    best_val = objective_binary(best_b)
    
    # Current state
    curr_b = best_b.copy()
    curr_val = best_val

    # Simulated Annealing Parameters
    T = 1.0  # Initial temperature
    T_min = 1e-4
    alpha = 0.999  # Cooling rate
    
    # We need to evaluate changes efficiently.
    # Swapping a 1 at i and a 0 at j changes the correlation.
    # Instead of recalculating full correlation every time (O(N^2) or O(N log N)),
    # we can just call objective_binary which uses np.correlate (O(N^2) for small N, highly optimized in C).
    # For N=100, np.correlate is extremely fast. Let's estimate: ~10000 ops. 
    # 1 million iterations * 10000 ops = 10^10 ops. Might be too slow in Python loop overhead?
    # np.correlate is C-level. 1M calls might take ~10-20s. 
    # Let's aim for 500k iterations.
    
    max_iters = 1000000
    
    # Pre-compute list of 1s and 0s for fast swap selection
    ones = np.where(curr_b == 1)[0]
    zeros = np.where(curr_b == 0)[0]
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            # Optional: restart with higher T if stuck? 
            # For now, just continue at T_min (greedy hill climbing with rare escapes)
            
        # Choose random 1 and random 0 to swap
        idx_1 = np.random.choice(ones)
        idx_0 = np.random.choice(zeros)
        
        # Perform swap
        curr_b[idx_1] = 0
        curr_b[idx_0] = 1
        
        # Update sets
        # Removing idx_1 from ones, adding idx_0
        # Adding idx_0 to zeros, removing idx_1
        # For efficiency, we can just recompute indices if sets get small, 
        # but for N=100, finding indices is cheap. 
        # However, maintaining arrays of indices requires care.
        # Simpler: just use np.where again? No, that's O(N).
        # Let's just use the arrays and update them manually or just pick from the full vector 
        # by checking values? 
        # Actually, for N=100, np.where(curr_b==1) is very fast.
        # But calling it every iteration is overhead.
        # Let's maintain the lists.
        
        # This part is tricky to maintain perfectly without bugs. 
        # Alternative: Just pick two random indices i, j.
        # If b[i]==1 and b[j]==0, swap.
        # If b[i]==0 and b[j]==1, swap.
        # If same, try again or just skip? 
        # Probability of picking a valid swap pair is 50/100 * 50/99 ~ 0.25.
        # So we do ~4 random picks per valid move. 
        # This is robust and avoids maintaining index sets.
        
        # Let's revert the manual swap approach and use the robust random pair approach 
        # inside the loop for correctness, accepting the slight overhead.
        
        # Undo the swap we just did to test the robust approach
        curr_b[idx_1] = 1
        curr_b[idx_0] = 0
        
        # --- Robust Move Generation ---
        i_idx = np.random.randint(N)
        j_idx = np.random.randint(N)
        
        if curr_b[i_idx] == curr_b[j_idx]:
            continue # Invalid move, skip
        
        # Swap
        curr_b[i_idx] = 0
        curr_b[j_idx] = 1
        
        # Evaluate new state
        new_val = objective_binary(curr_b)
        
        # Acceptance criterion
        if new_val < curr_val or random.random() < math.exp(-(new_val - curr_val) / T):
            curr_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_b = curr_b.copy()
        else:
            # Reject, swap back
            curr_b[i_idx] = 1
            curr_b[j_idx] = 0
            
    # Final Result
    # The problem requires h in [0,1]. Binary is a subset.
    # We return the best binary vector found.
    # It satisfies sum = 50 exactly.
    # It satisfies 0 <= h <= 1.
    
    # Verify validity
    if not np.all(np.isfinite(best_b)) or np.abs(np.sum(best_b) - N/2.0) > 1e-9 * N:
        return np.full(N, 0.5)
        
    return best_b.astype(float)

# EVOLVE-BLOCK-END
