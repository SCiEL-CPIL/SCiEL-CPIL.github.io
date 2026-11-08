import numpy as np
import time

N = 400
# Domain is [-1/4, 1/4], length 0.5
# Each interval has width w = 0.5 / N
w = 0.5 / N

# The autoconvolution g(t) = (f*f)(t) is supported on [-0.5, 0.5].
# We discretize this on a grid.
# To capture the piecewise linear nature, we can use a grid of size M = 4*N or 8*N.
# Let's use M = 4*N for the internal representation of g to balance speed and accuracy.
# The grid points for g are from -0.5 to 0.5.
M = 4 * N
dx = 1.0 / M  # Width of each bin in the autoconvolution grid
# Note: The actual convolution integral scales with w.
# f(s) is height h_i on interval i.
# g(t) = sum_i sum_j h_i h_j * overlap(I_i, t-I_j)
# The maximum value of g is bounded by sum |h_i| w = N * 1 * w = 0.5.

def _init_autoconv(heights):
    """
    Compute the initial autoconvolution on the grid of size M.
    heights: array of shape (N,) with values in {-1, 1}
    Returns: array of shape (M,) representing g(t)
    """
    # Use FFT for initialization as it is O(N log N) and we only do it once (or few times for restarts)
    # Map heights to fine grid of size M
    # Each height covers M/N = 4 points
    f_fine = np.repeat(heights.astype(float), M // N)
    if len(f_fine) < M:
        f_fine = np.pad(f_fine, (0, M - len(f_fine)))
    elif len(f_fine) > M:
        f_fine = f_fine[:M]
        
    # Linear convolution via FFT
    # Zero pad to 2*M
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real
    # Scale by dx? 
    # The FFT convolution computes sum f_i g_j. 
    # The integral is sum f_i g_j dx.
    # So we multiply by dx.
    autoconv *= dx
    
    # The result is on a grid of length 2*M, representing [-0.5, 0.5] shifted?
    # np.fft.ifft returns values in order 0, 1, ..., 2M-1.
    # The convolution of two signals supported on [0, M) is supported on [0, 2M).
    # Our f is defined on indices 0..M-1 corresponding to x in [-1/4, 1/4].
    # So the autoconvolution is on indices 0..2M-1 corresponding to t in [-1/4, 1/4] + [-1/4, 1/4] = [-1/2, 1/2].
    # We need to shift it so that index 0 corresponds to t = -0.5?
    # Actually, for finding the max absolute value, the shift doesn't matter.
    # But for incremental updates, we need to know which index corresponds to which t.
    # Let's assume index k in autoconv corresponds to t_k = (k - M) * dx? 
    # If f is at 0..M-1, conv is at 0..2M-1.
    # Center is at M.
    # t = -0.5 corresponds to index 0? 
    # Let's check: f(s) is non-zero for s in [0, M) in grid units (representing [-0.25, 0.25]).
    # t = s + s'. Range of t is [0, 2M).
    # So index 0 is t=0 (which is -0.25 + -0.25 = -0.5 in physical units).
    # Index M is t=M (which is 0).
    # So the array `autoconv` is already aligned such that index 0 is the leftmost point.
    
    return autoconv

def _get_score(g):
    """
    Compute the score (upper bound on C3) from the autoconvolution array.
    The task asks for the "upper bound on C3". 
    Based on the context "Third autocorrelation inequality", this is likely the L-infinity norm 
    of the autoconvolution, possibly normalized.
    The seed program used np.max(np.abs(autoconv)).
    """
    return np.max(np.abs(g))

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin
    
    # Precompute prefix sums for fast integral queries
    # We need to maintain the current heights `h` and the current autoconvolution `g`.
    # We also need a way to quickly compute the integral of `f` over any interval.
    # Let F[k] be the prefix sum of the fine-grid representation of f?
    # Or just the prefix sum of the step heights?
    # Integral of f over [a, b] in physical units.
    # f is constant on intervals I_i = [x_i, x_i + w).
    # We can maintain a prefix sum array P of the heights h.
    # P[i] = sum_{j=0}^{i-1} h_j.
    # Then sum_{j=i}^{k} h_j = P[k+1] - P[i].
    # The integral of f over a physical interval [u, v] is w * (sum of h_j for intervals fully inside + partials).
    
    # For incremental update of g when flipping bit i:
    # delta_g(t) = 2 * h_i * ( Integral_{s in I_i} f(t-s) ds + Integral_{s} f(s) 1_{I_i}(t-s) ds )
    # Let I_i be the interval [L_i, R_i] in physical coordinates.
    # Term 1: Int_{L_i}^{R_i} f(t-s) ds. Let u = t-s. ds = -du.
    # Limits: s=L_i -> u=t-L_i. s=R_i -> u=t-R_i.
    # Int = Int_{t-R_i}^{t-L_i} f(u) du.
    # This is the integral of f over the interval [t-R_i, t-L_i].
    # Term 2: Int f(s) 1_{I_i}(t-s) ds. Condition t-s in I_i => s in [t-R_i, t-L_i].
    # So Term 2 is Int_{t-R_i}^{t-L_i} f(s) ds.
    # They are identical!
    # So delta_g(t) = 4 * h_i * Int_{t-R_i}^{t-L_i} f(u) du.
    
    # We need to compute this integral for all t in the autoconvolution grid.
    # The autoconvolution grid has M points? No, the support is 2*M points in the FFT result.
    # Let's work with the array `g` of length 2*M.
    # Index k in `g` corresponds to physical time t_k = (k - M) * dx? 
    # Wait, earlier I said index 0 is t=-0.5.
    # t_k = -0.5 + k * dx.
    
    # We need to compute I_k = Int_{t_k - R_i}^{t_k - L_i} f(u) du for all k.
    # This is a "sliding window integral" of f.
    # We can precompute the prefix sum of the fine-grid f to make this O(1) per k.
    # Let f_fine be the array of length M representing f on the fine grid.
    # Let P_fine be the prefix sum of f_fine.
    # Integral of f over [a, b] in physical units:
    # Map a, b to fine grid indices.
    # idx_a = (a - (-0.25)) / w_fine ?
    # w_fine = dx.
    # a_idx = (a + 0.25) / dx.
    # b_idx = (b + 0.25) / dx.
    # Integral = (P_fine[b_idx] - P_fine[a_idx]) * dx.
    
    # However, updating P_fine when h_i changes is O(M) if we do it naively, or O(1) if we just update the range?
    # P_fine is a prefix sum. Changing h_i affects P_fine[j] for all j >= start_idx_i.
    # That is O(M).
    # So updating the prefix sum is O(M).
    # Then computing delta_g for all k is O(M).
    # Total O(M) per move. M = 1600.
    # This is much faster than FFT (O(M log M) ~ 1600*11 ~ 17600 ops).
    # So ~10x speedup.
    
    # Let's implement this.
    
    best_h = np.random.choice([-1, 1], N).astype(np.int8)
    best_g = _init_autoconv(best_h)
    best_score = _get_score(best_g)
    
    current_h = best_h.copy()
    current_g = best_g.copy()
    current_score = best_score
    
    # Fine grid representation of f
    # f_fine length M.
    # h_i is repeated M//N times.
    f_fine = np.repeat(current_h.astype(float), M // N)
    if len(f_fine) < M:
        f_fine = np.pad(f_fine, (0, M - len(f_fine)))
        
    # Prefix sum of f_fine
    # P_fine[0] = 0
    # P_fine[k] = sum_{j=0}^{k-1} f_fine[j]
    P_fine = np.zeros(M + 1)
    np.cumsum(f_fine, out=P_fine[1:])
    
    # Tabu List
    tabu_len = 100
    tabu_list = np.full(N, -1, dtype=np.int32) # Stores iteration count when item became tabu
    iteration = 0
    
    # Physical coordinates of intervals
    # I_i = [ -0.25 + i*w, -0.25 + (i+1)*w ]
    # L_i = -0.25 + i*w
    # R_i = -0.25 + (i+1)*w
    L_phys = -0.25 + np.arange(N) * w
    R_phys = -0.25 + (np.arange(N) + 1) * w
    
    # Grid points for t
    t_grid = -0.5 + np.arange(2*M) * dx
    
    while time.time() - start_time < DEADLINE:
        iteration += 1
        
        # Evaluate all possible moves (single bit flips)
        # We can vectorize the evaluation of delta_g for all i?
        # delta_g_i(k) = 4 * h_i * Int_{t_k - R_i}^{t_k - L_i} f(u) du
        # This depends on i and k.
        # Computing this for all i and k is O(N*M).
        # N=400, M=1600. 640,000 operations.
        # This is feasible in NumPy if vectorized.
        
        # Let's try to vectorize.
        # We need to compute the integral of f over [t_k - R_i, t_k - L_i] for all i, k.
        # Let A[i, k] = t_k - R_i.
        # Let B[i, k] = t_k - L_i.
        # We need Int(A[i,k], B[i,k]) for all i,k.
        
        # This might be memory heavy (400 x 3200 floats = 1.2M floats = 10MB). OK.
        
        # However, we only need the MAX of |g + delta_g| for each i to find the best move.
        # g is fixed. delta_g depends on i.
        
        # For each i:
        # 1. Compute delta_g_i (vector of length 2*M).
        # 2. Compute new_g = g + delta_g_i.
        # 3. Compute score_i = max(abs(new_g)).
        # 4. Check tabu status of i.
        
        # Doing this in a Python loop over i (400 iterations) where each iteration does O(M) NumPy ops is fast.
        # 400 * (O(M) vector ops) ~ 400 * 1600 ~ 640k ops per full neighborhood eval.
        # With 115s budget, we can do many iterations.
        
        best_move_i = -1
        best_move_score = current_score
        best_new_g = None
        
        for i in range(N):
            # Check tabu
            if tabu_list[i] > iteration - tabu_len and current_score > best_score:
                # If it's tabu and doesn't improve global best, skip
                # (Aspiration criterion: allow if it improves global best)
                if not (current_score <= best_score): 
                    continue
            
            # Compute delta_g for flipping bit i
            # h_i is current_h[i]
            # delta_g(k) = 4 * h_i * Integral_{t_k - R_i}^{t_k - L_i} f(u) du
            
            # We need the integral of f over [t_k - R_i, t_k - L_i] for all k.
            # Let start_phys = t_grid - R_i
            # Let end_phys = t_grid - L_i
            
            start_phys = t_grid - R_phys[i]
            end_phys = t_grid - L_phys[i]
            
            # Map to fine grid indices
            # f is defined on [-0.25, 0.25].
            # If the interval [start_phys, end_phys] is outside, integral is 0.
            # Clip to [-0.25, 0.25]
            s_clipped = np.clip(start_phys, -0.25, 0.25)
            e_clipped = np.clip(end_phys, -0.25, 0.25)
            
            # If s_clipped >= e_clipped, integral is 0
            # Else, compute integral using P_fine
            
            # Map physical to index
            # idx = (phys - (-0.25)) / dx
            s_idx = (s_clipped + 0.25) / dx
            e_idx = (e_clipped + 0.25) / dx
            
            # P_fine is defined on integers 0..M.
            # We need to handle partial intervals.
            # Since f is a step function on the fine grid (constant in each fine bin),
            # and our fine grid matches the step boundaries (M is multiple of N),
            # the integral over [s, e] is simply:
            # sum of f_fine[j] * dx for j where bin [j*dx, (j+1)*dx] intersects [s, e]?
            # No, f is constant on the fine grid bins.
            # So Integral = dx * (sum of f_fine values for bins fully or partially covered?
            # Actually, since s_idx and e_idx might not be integers, we need to be careful.
            # However, s_idx and e_idx are derived from t_grid (which is on the dx grid) and R_i, L_i (which are on the w grid, and w is multiple of dx).
            # So s_idx and e_idx are integers!
            # Let's verify:
            # t_k = -0.5 + k*dx.
            # R_i = -0.25 + (i+1)*w.
            # w = M/N * dx.
            # R_i = -0.25 + (i+1)*(M/N)*dx.
            # start_phys = -0.5 + k*dx - (-0.25 + (i+1)*(M/N)*dx)
            #            = -0.25 + (k - (i+1)*(M/N))*dx
            # s_idx = (start_phys + 0.25) / dx = k - (i+1)*(M/N).
            # This is an integer.
            
            s_idx_int = np.round(s_idx).astype(np.int32)
            e_idx_int = np.round(e_idx).astype(np.int32)
            
            # Calculate integral
            # Integral = P_fine[e_idx_int] - P_fine[s_idx_int]
            # But we must ensure s_idx_int, e_idx_int are within [0, M].
            # And if s_idx_int >= e_idx_int, integral is 0.
            
            s_idx_int = np.clip(s_idx_int, 0, M)
            e_idx_int = np.clip(e_idx_int, 0, M)
            
            # P_fine is length M+1.
            # We need to handle the case where the interval is invalid (start > end)
            valid = s_idx_int < e_idx_int
            # Initialize integral to 0
            integrals = np.zeros(2*M)
            # For valid entries:
            # integrals[valid] = P_fine[e_idx_int[valid]] - P_fine[s_idx_int[valid]]
            # Note: P_fine is an array. We can use advanced indexing.
            integrals[valid] = P_fine[e_idx_int[valid]] - P_fine[s_idx_int[valid]]
            
            # Multiply by dx and 4 * h_i
            delta_g = 4.0 * current_h[i] * integrals * dx
            
            # New g
            new_g = current_g + delta_g
            
            # Score
            score_i = np.max(np.abs(new_g))
            
            # Update best move
            if score_i < best_move_score:
                best_move_score = score_i
                best_move_i = i
                best_new_g = new_g
                # Also track if this improves global best
                if score_i < best_score:
                    best_score = score_i
                    best_h = current_h.copy()
                    best_g = current_g.copy()
            
            # Mark i as tabu
            tabu_list[i] = iteration
        
        # Apply the best move
        if best_move_i != -1 and best_new_g is not None:
            current_h[best_move_i] *= -1
            current_g = best_new_g
            current_score = best_move_score
            
            # Update f_fine and P_fine
            # The fine grid indices corresponding to bit i
            start_fine = i * (M // N)
            end_fine = (i + 1) * (M // N)
            f_fine[start_fine:end_fine] = current_h[i]
            
            # Update P_fine
            # P_fine[j] for j > start_fine changes by delta = (current_h[i] - old_h[i]) * (end_fine - start_fine)
            # old_h[i] was -current_h[i]
            # delta = 2 * current_h[i] * (end_fine - start_fine)
            # But we only need to update P_fine[j] for j >= start_fine + 1?
            # P_fine[k] = sum_{j=0}^{k-1} f_fine[j].
            # If we change f_fine[start_fine:end_fine], then P_fine[k] changes for all k > start_fine.
            # Specifically, for k in (start_fine, end_fine], the change is proportional to how many of the changed bins are included.
            # This is O(M) to update.
            # Let's just recompute P_fine? O(M) cumsum.
            # M=1600. This is very fast.
            np.cumsum(f_fine, out=P_fine[1:])
            P_fine[0] = 0

    return best_h

# The function name must be construct()
# The return type should be a sequence of signed step heights (int8 or similar)
