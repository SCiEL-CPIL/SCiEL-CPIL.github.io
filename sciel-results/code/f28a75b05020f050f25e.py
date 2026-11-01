import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 118.0  # Safety margin before 120s/150s kill

def _compute_autocorr(h):
    """
    Compute linear aperiodic autocorrelation using FFT.
    Returns array of length 2*N.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    return ac

def _max_sidelobe(ac):
    """
    Compute max absolute value of sidelobes.
    """
    return np.max(np.abs(ac[1:N]))

def _objective_sum_sq(x):
    """
    Objective: Sum of squares of sidelobes.
    This is a smooth surrogate for the max sidelobe problem.
    """
    h = x
    ac = _compute_autocorr(h)
    sidelobes = ac[1:N]
    obj = np.sum(sidelobes**2)
    
    # Gradient calculation
    grad = np.zeros(N)
    for k in range(1, N):
        c_k = sidelobes[k-1]
        if c_k == 0:
            continue
        
        # Vectorized update
        grad[0:N-k] += 2.0 * c_k * h[k:N]
        grad[k:N] += 2.0 * c_k * h[0:N-k]
        
    return obj, grad

def _local_search_sa_fast(h, time_limit=5.0, T0=None, cooling_rate=0.9995):
    """
    Simulated Annealing local search with O(N) incremental autocorrelation updates.
    """
    start_time = time.time()
    # Initial autocorrelation
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    if T0 is None:
        T0 = max(current_val * 0.1, 0.01)
    
    h = h.copy()
    ac = ac.copy()
    
    # Pre-allocate delta array? No, we update ac in place.
    # We need to be careful with the update formula.
    # C_k = sum_{j=0}^{N-1-k} h_j h_{j+k}
    # Flip i: h_i -> -h_i
    # Change in C_k:
    # If i < N-k: term h_i h_{i+k} becomes -h_i h_{i+k}. Delta = -2 h_i h_{i+k}
    # If i-k >= 0: term h_{i-k} h_i becomes -h_{i-k} h_i. Delta = -2 h_{i-k} h_i
    # So Delta C_k = -2 * h_i * (h_{i+k} + h_{i-k}) with appropriate bounds.
    
    # To optimize, we can precompute the range of k affected by flipping i.
    # k ranges from 1 to N-1.
    # h_{i+k} is valid if i+k < N => k < N-i
    # h_{i-k} is valid if i-k >= 0 => k <= i
    
    for it in range(1000000): # Large cap
        if time.time() - start_time > time_limit:
            break
        
        i = np.random.randint(N)
        old_h_i = h[i]
        
        # Compute delta for all k
        # k = 1 .. N-1
        # delta_c = -2 * h[i] * (h[i+k] + h[i-k])
        
        # We can vectorize this update.
        # Create indices k = 1..N-1
        # k = np.arange(1, N)
        # h_ik = h[i+k] where i+k < N, else 0
        # h_ikm = h[i-k] where i-k >= 0, else 0
        
        # Let's just do the loop or vectorized slice ops.
        # Vectorized:
        k = np.arange(1, N)
        
        # h[i+k]
        # valid if i+k < N => k < N-i
        # indices: i+1 .. i+N-1, but only up to N-1
        # h[i+1 : min(N, i+N)] -> h[i+1 : N] if i+N > N
        # length of valid part: min(N-1-i, N-1) ?
        # k goes 1 to N-1.
        # h[i+k] exists for k in [1, N-1-i]
        # h[i-k] exists for k in [1, i]
        
        # Let's construct the delta array directly.
        # delta_c = np.zeros(N-1)
        # if i > 0: delta_c[0:i] += -2 * h[i] * h[i-1::-1]  # h[i-1], h[i-2]... h[0]
        # if i < N-1: delta_c[N-1-i-1:] += -2 * h[i] * h[i+1:] # h[i+1], h[i+2]... h[N-1]
        # Wait, index mapping:
        # delta_c[k-1] corresponds to C_k.
        # Term 1: h[i-k] for k=1..i. Indices in delta_c: 0..i-1. Values: h[i-1], h[i-2]...h[0].
        # Term 2: h[i+k] for k=1..N-1-i. Indices in delta_c: N-1-i .. N-2? No.
        # k=1 -> index 0. k=N-1-i -> index N-2-i.
        # So slice [0 : N-1-i] for h[i+1 : N].
        
        # Let's verify.
        # C_k involves h_j h_{j+k}.
        # If we flip h_i:
        # 1. j=i, term h_i h_{i+k}. Valid if i+k < N. k < N-i.
        #    Indices in C array (0-indexed for lags 1..N-1): k-1.
        #    So indices 0 to N-i-2.
        #    Values: h[i+1], h[i+2], ..., h[N-1].
        #    Delta: -2 * h_i * h[i+1:N]
        # 2. j+k=i => j=i-k. Term h_{i-k} h_i. Valid if i-k >= 0. k <= i.
        #    Indices in C array: k-1.
        #    So indices 0 to i-1.
        #    Values: h[i-1], h[i-2], ..., h[0].
        #    Delta: -2 * h_i * h[i-1::-1]
        
        # So:
        # delta_c = np.zeros(N-1)
        # if i > 0:
        #     delta_c[0:i] += -2.0 * h[i] * h[i-1::-1]
        # if i < N-1:
        #     delta_c[0:N-1-i] += -2.0 * h[i] * h[i+1:]
        
        # This is O(N) but with Python overhead.
        # Given N=400, this is very fast.
        
        # However, we can just update `ac` in place.
        # ac[1:N] holds C_1 .. C_{N-1}.
        
        # Let's create a temporary delta array to add to ac[1:N]
        # Actually, we can just do:
        
        # Save old ac values if we need to revert?
        # We can just copy ac to ac_new? No, that's O(N).
        # We can modify ac, check, and revert if needed.
        
        # Modify ac
        # We need to revert if rejected.
        # So we must save the modified parts or just recompute?
        # Recompute is O(N log N).
        # Saving/Restoring O(N) slices is fast.
        
        # Let's store the original ac[1:N] in a temp buffer?
        # Or just copy ac? ac is length 2N.
        # ac_copy = ac.copy() is O(N).
        
        # Better:
        # Apply delta to ac.
        # Compute new max.
        # If reject, subtract delta (or restore from copy).
        
        # Let's use a copy for simplicity and safety, N=400 is tiny.
        ac_orig = ac.copy()
        
        # Apply update
        if i > 0:
            ac[1 : i+1] += -2.0 * h[i] * h[i-1::-1]
        if i < N-1:
            ac[1 : N-i] += -2.0 * h[i] * h[i+1:]
            
        # Note: ac[1:N] corresponds to lags 1..N-1.
        # Slice 1:i+1 has length i. h[i-1::-1] has length i. Correct.
        # Slice 1:N-i has length N-1-i. h[i+1:] has length N-1-i. Correct.
        
        new_val = _max_sidelobe(ac)
        
        delta = new_val - current_val
        T = T0 * (cooling_rate ** it)
        
        if delta < 0 or (T > 1e-15 and np.random.random() < np.exp(-delta / T)):
            h[i] = -h[i]
            current_val = new_val
            # ac is already updated
        else:
            # Revert
            ac[:] = ac_orig
            # h[i] is already -h[i] in the check? No, we didn't change h[i] yet in the success branch?
            # Wait, in success branch I wrote `h[i] = -h[i]`.
            # In failure branch, h[i] is still old_h_i.
            # So no need to revert h.
            
    return h, current_val

def _greedy_polish(h, time_limit=2.0):
    """
    Strict greedy search using fast updates.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(N):
            ac_orig = ac.copy()
            
            # Apply update
            if i > 0:
                ac[1 : i+1] += -2.0 * h[i] * h[i-1::-1]
            if i < N-1:
                ac[1 : N-i] += -2.0 * h[i] * h[i+1:]
                
            new_val = _max_sidelobe(ac)
            
            if new_val < current_val:
                h[i] = -h[i]
                current_val = new_val
                improved = True
            else:
                ac[:] = ac_orig
                # h[i] unchanged
                
    return h, current_val

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous Relaxation
    num_continuous_starts = 5
    
    for i in range(num_continuous_starts):
        if time.time() >= deadline - 20.0:
            break
        
        x0 = np.random.choice([-1.0, 1.0], size=N)
        
        try:
            res = minimize(_objective_sum_sq, x0, method='L-BFGS-B', 
                           bounds=[(-1.0, 1.0)] * N, 
                           options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-6, 'maxls': 20})
        except Exception:
            continue
        
        h_bin = np.sign(res.x)
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
            
        ac = _compute_autocorr(h_bin)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
            
        if best_h is not None and time.time() < deadline - 10.0:
            time_for_sa = min(4.0, max(1.0, (deadline - 10.0 - time.time()) / max(1, num_continuous_starts - i)))
            
            h_sa, val_sa = _local_search_sa_fast(best_h.copy(), time_limit=time_for_sa)
            
            if val_sa < best_val:
                best_val = val_sa
                best_h = h_sa
    
    if best_h is not None and time.time() < deadline - 5.0:
        remaining_time = deadline - 5.0 - time.time()
        if remaining_time > 2.0:
            h_final, val_final = _local_search_sa_fast(best_h.copy(), time_limit=remaining_time)
            if val_final < best_val:
                best_val = val_final
                best_h = h_final
    
    if best_h is not None and time.time() < deadline - 1.0:
        h_polish, val_polish = _greedy_polish(best_h.copy(), time_limit=min(2.0, deadline - 1.0 - time.time()))
        if val_polish < best_val:
            best_val = val_polish
            best_h = h_polish
        
    if best_h is None:
        return np.ones(N)
        
    return best_h
