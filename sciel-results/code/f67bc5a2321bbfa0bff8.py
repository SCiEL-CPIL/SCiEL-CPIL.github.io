import numpy as np
import time

N = 50

def _compute_ratio(h):
    """Compute the ratio: squared L2 norm / (L1 norm * sup norm of autoconvolution)"""
    h = np.asarray(h, dtype=np.float64)
    if np.all(h == 0):
        return 0.0
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 == 0 or l2_sq == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    conv_max = np.max(conv)
    if conv_max == 0:
        return 0.0
    return l2_sq / (l1 * conv_max)

def _multiplicative_updates(h_init, max_iter=5000, tol=1e-10, time_budget=80.0):
    """
    Zero-sum game formulation:
    - Attacker selects lag k to maximize autoconvolution peak
    - Defender allocates nonnegative mass to minimize ratio denominator
    
    Uses iterative multiplicative updates (Sinkhorn-Knopp style)
    """
    h = np.array(h_init, dtype=np.float64)
    h = np.maximum(h, 1e-10)  # Ensure positivity for log updates
    h = h / np.sum(h)  # Normalize
    
    # Precompute the autoconvolution structure
    # For lag k, the autoconvolution at that lag is sum_i h[i]*h[i+k]
    # We want to find a saddle point where the attacker's optimal lag
    # and defender's optimal allocation are in equilibrium
    
    best_h = h.copy()
    best_ratio = _compute_ratio(best_h)
    
    start = time.time()
    
    for iteration in range(max_iter):
        if time.time() - start > time_budget:
            break
            
        # Compute current autoconvolution
        conv = np.convolve(h, h, mode='full')
        conv_max = np.max(conv)
        
        if conv_max <= 0:
            break
            
        # Attacker's perspective: which lag maximizes the peak?
        # The autoconvolution has lags from -(N-1) to (N-1)
        # conv[i] corresponds to lag i - (N-1)
        # Find the lag with maximum value
        peak_lag = np.argmax(conv)
        
        # Defender's update: minimize the denominator while keeping L1 fixed
        # The key insight: we want to reduce h[i]*h[i+lag] for the peak lag
        # while spreading mass to other positions
        
        # Multiplicative update rule:
        # h_new[i] = h[i] * exp(-alpha * gradient_i)
        # where gradient_i comes from the autoconvolution at the peak lag
        
        # For the peak lag, the contribution to autoconvolution is:
        # sum_i h[i] * h[i + peak_lag_offset]
        # where peak_lag_offset = peak_lag - (N-1)
        
        lag_offset = peak_lag - (N - 1)
        
        # Compute the "pressure" on each index from the peak lag
        pressure = np.zeros(N)
        if lag_offset >= 0:
            # h[i] * h[i + lag_offset] for i = 0 to N-1-lag_offset
            for i in range(N - lag_offset):
                pressure[i] += h[i + lag_offset]
                pressure[i + lag_offset] += h[i]
        else:
            # h[i] * h[i + lag_offset] where lag_offset < 0
            # This is h[i] * h[i - |lag_offset|] for i = |lag_offset| to N-1
            abs_lag = -lag_offset
            for i in range(abs_lag, N):
                pressure[i] += h[i - abs_lag]
                pressure[i - abs_lag] += h[i]
        
        # Normalize pressure
        if np.max(pressure) > 0:
            pressure = pressure / np.max(pressure)
        
        # Multiplicative update: reduce mass where pressure is high
        # Use a learning rate that ensures convergence
        alpha = 0.5
        h_new = h * np.exp(-alpha * pressure)
        
        # Enforce non-negativity and normalize
        h_new = np.maximum(h_new, 1e-15)
        h_new = h_new / np.sum(h_new)
        
        # Check convergence
        if np.max(np.abs(h_new - h)) < tol:
            h = h_new
            break
        
        h = h_new
        
        # Track best
        current_ratio = _compute_ratio(h)
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_h = h.copy()
    
    return best_h, best_ratio

def _local_polish(h, max_iter=2000, time_budget=30.0):
    """Local coordinate descent polish to improve the ratio"""
    h = np.array(h, dtype=np.float64)
    best_ratio = _compute_ratio(h)
    best_h = h.copy()
    
    start = time.time()
    
    for iteration in range(max_iter):
        if time.time() - start > time_budget:
            break
            
        improved = False
        for i in range(N):
            # Try adjusting h[i] slightly
            original = h[i]
            if original < 1e-15:
                continue
                
            # Try increasing and decreasing
            for delta in [0.1, 0.01, -0.1, -0.01]:
                h[i] = max(0, original + delta * original)
                # Renormalize to keep L1 constant
                l1 = np.sum(h)
                if l1 > 0:
                    h = h / l1
                    ratio = _compute_ratio(h)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_h = h.copy()
                        improved = True
                h[i] = original
        
        if not improved:
            break
    
    return best_h, best_ratio

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit
    
    # Strategy 1: Start with Sidon set (known good baseline)
    def is_sidon(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    best_h = None
    best_ratio = 0.0
    
    # Try to find a good Sidon set
    for k in range(12, 5, -1):
        if time.time() > deadline - 60:
            break
        for _ in range(500):
            if time.time() > deadline - 60:
                break
            S = np.random.choice(N, size=k, replace=False)
            if is_sidon(S.tolist()):
                h = np.zeros(N)
                for idx in S:
                    h[idx] = 1.0
                ratio = _compute_ratio(h)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_h = h.copy()
                break
    
    # If no Sidon set found, use uniform initialization
    if best_h is None:
        best_h = np.ones(N)
        best_h = best_h / np.sum(best_h)
        best_ratio = _compute_ratio(best_h)
    
    # Strategy 2: Multiplicative updates from the Sidon set
    if time.time() < deadline - 40:
        h_mu, ratio_mu = _multiplicative_updates(best_h, max_iter=3000, time_budget=40.0)
        if ratio_mu > best_ratio:
            best_ratio = ratio_mu
            best_h = h_mu
    
    # Strategy 3: Local polish
    if time.time() < deadline - 10:
        h_polished, ratio_polished = _local_polish(best_h, max_iter=1000, time_budget=20.0)
        if ratio_polished > best_ratio:
            best_ratio = ratio_polished
            best_h = h_polished
    
    # Convert to list of floats
    return list(best_h)
