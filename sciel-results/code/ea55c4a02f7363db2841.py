import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 145.0 # 150s limit, 5s margin

def compute_ratio(h):
    h = np.array(h)
    if np.any(h < -1e-9):
        return -1.0
    h = np.clip(h, 0, None)
    
    l1 = np.sum(h)
    if l1 < 1e-12:
        return -1.0
        
    l2_sq = np.sum(h**2)
    
    # Linear convolution
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(np.abs(h_conv))
    
    if max_conv < 1e-12:
        return -1.0
        
    return (l2_sq / (l1 * max_conv))

def smooth_max(x, beta=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

def is_sidon(arr):
    """Check if a set of integers is a Sidon set (B2 set)."""
    arr = sorted(set(arr))
    n = len(arr)
    sums = {}
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in sums:
                return False
            sums[s] = (i, j)
    return True

def find_sidon_set_greedy(n, max_val=1000):
    """Find a Sidon set of size n using a greedy algorithm."""
    sidon = []
    for i in range(max_val):
        candidate = sidon + [i]
        if is_sidon(candidate):
            sidon.append(i)
            if len(sidon) == n:
                break
    return sidon

def find_sidon_set_backtracking(n, max_val=100):
    """Find a Sidon set of size n using backtracking for better quality."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    
    best_set = None
    for start in range(max_val):
        current = [start]
        if _extend_sidon(current, n, max_val):
            if best_set is None or len(current) > len(best_set):
                best_set = current
            if len(best_set) == n:
                break
    
    if best_set is None:
        return find_sidon_set_greedy(n, max_val)
    return best_set[:n]

def _extend_sidon(current, target_size, max_val):
    """Try to extend current Sidon set to target_size."""
    if len(current) == target_size:
        return True
    if len(current) > target_size:
        return False
    
    last = current[-1] if current else -1
    for candidate in range(last + 1, max_val):
        new_set = current + [candidate]
        if is_sidon(new_set):
            if _extend_sidon(new_set, target_size, max_val):
                current.clear()
                current.extend(new_set)
                return True
    return False

def smooth_neg_log_ratio(params, beta=100.0):
    """Smooth version of neg_log_ratio using log-sum-exp for max."""
    h = np.exp(params)
    h_norm = h / np.sum(h)
    
    l1 = np.sum(h_norm)
    l2_sq = np.sum(h_norm**2)
    
    h_conv = np.convolve(h_norm, h_norm, mode='full')
    
    # Use smooth max
    max_conv = smooth_max(h_conv, beta)
    
    if max_conv < 1e-12:
        return 1000.0
        
    ratio = l2_sq / (l1 * max_conv)
    if ratio <= 0:
        return 1000.0
    return -np.log(ratio)

def construct():
    start_time = time.time()
    
    # Initial candidates
    candidates = []
    
    # 1. Uniform
    h_uni = np.ones(N) / N
    candidates.append(h_uni)
    
    # 2. Spike
    h_spike = np.zeros(N)
    h_spike[N//2] = 1.0
    candidates.append(h_spike)
    
    # 3. Gaussian-like
    x = np.linspace(-1, 1, N)
    h_gauss = np.exp(-5 * x**2)
    h_gauss /= np.sum(h_gauss)
    candidates.append(h_gauss)
    
    # 4. Cosine bump
    h_cos = 0.5 * (1 + np.cos(np.pi * x))
    h_cos /= np.sum(h_cos)
    candidates.append(h_cos)
    
    # 5. Sidon set based initialization - use backtracking for better quality
    sidon_indices = find_sidon_set_backtracking(8, N)
    if len(sidon_indices) < 5:
        # Fallback to greedy if backtracking fails
        sidon_indices = []
        for i in range(N):
            candidate = sidon_indices + [i]
            if is_sidon(candidate):
                sidon_indices.append(i)
    
    # Create h based on Sidon indices with optimized weights
    h_sidon = np.zeros(N)
    if sidon_indices:
        # Try different weight distributions
        for w_type in ['uniform', 'centered', 'edge']:
            h_test = np.zeros(N)
            if w_type == 'uniform':
                h_test[sidon_indices] = 1.0 / np.sqrt(len(sidon_indices))
            elif w_type == 'centered':
                # Heavier weight on center indices
                center = N / 2
                weights = np.exp(-0.5 * (np.array(sidon_indices) - center) ** 2 / (N / 4))
                h_test[sidon_indices] = weights / np.sum(weights)
            else:  # edge
                # Heavier weight on edge indices
                h_test[sidon_indices] = 1.0 / np.sqrt(len(sidon_indices))
                h_test[sidon_indices[0]] *= 1.5
                h_test[sidon_indices[-1]] *= 1.5
                h_test /= np.sum(h_test)
            candidates.append(h_test)
    
    # Also try a "sparse" random initialization
    for K in [5, 7, 10]:
        h_sparse = np.zeros(N)
        indices = np.random.choice(N, K, replace=False)
        h_sparse[indices] = 1.0 / np.sqrt(K)
        candidates.append(h_sparse)

    best_h = candidates[0]
    best_val = compute_ratio(best_h)
    
    for h in candidates[1:]:
        val = compute_ratio(h)
        if val > best_val:
            best_val = val
            best_h = h
            
    # Optimization
    # Minimize -log(ratio)
    # Use log-transform for positivity
    
    def neg_log_ratio(params):
        h = np.exp(params)
        h_norm = h / np.sum(h)
        r = compute_ratio(h_norm)
        if r <= 0:
            return 1000.0
        return -np.log(r)
        
    # Random restarts with smooth surrogate first, then exact refinement
    n_restarts = 40
    for i in range(n_restarts):
        if time.time() > DEADLINE - 15:  # Leave 15s for polish
            break
            
        if i == 0:
            x0 = np.log(best_h + 1e-6)
        elif i == 1 and sidon_indices:
            # Start from best Sidon variant
            h_sidon_best = np.zeros(N)
            h_sidon_best[sidon_indices] = 1.0 / np.sqrt(len(sidon_indices))
            x0 = np.log(h_sidon_best + 1e-6)
        elif i == 2 and sidon_indices:
            # Start from centered Sidon
            h_sidon_centered = np.zeros(N)
            center = N / 2
            weights = np.exp(-0.5 * (np.array(sidon_indices) - center) ** 2 / (N / 4))
            h_sidon_centered[sidon_indices] = weights / np.sum(weights)
            x0 = np.log(h_sidon_centered + 1e-6)
        else:
            # Random sparse start with varying sparsity
            K = np.random.choice([3, 5, 7, 10, 12])
            h_start = np.zeros(N)
            indices = np.random.choice(N, K, replace=False)
            h_start[indices] = 1.0 / np.sqrt(K)
            x0 = np.log(h_start + 1e-6)
            
        try:
            # First pass: smooth surrogate with moderate beta
            res_smooth = minimize(smooth_neg_log_ratio, x0, method='L-BFGS-B', 
                                  options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res_smooth.success:
                h_smooth = np.exp(res_smooth.x)
                h_smooth = h_smooth / np.sum(h_smooth)
                val_smooth = compute_ratio(h_smooth)
                
                if val_smooth > best_val:
                    best_val = val_smooth
                    best_h = h_smooth
                
                # Second pass: exact objective with tighter tolerance
                x0_exact = np.log(h_smooth + 1e-8)
                res_exact = minimize(neg_log_ratio, x0_exact, method='L-BFGS-B', 
                                     options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res_exact.success:
                    h_opt = np.exp(res_exact.x)
                    h_opt = h_opt / np.sum(h_opt)
                    val_opt = compute_ratio(h_opt)
                    if val_opt > best_val:
                        best_val = val_opt
                        best_h = h_opt
        except Exception:
            pass

    # Final high-precision polish pass with coordinate descent
    if time.time() < DEADLINE - 3:
        try:
            x0_polish = np.log(best_h + 1e-10)
            res_polish = minimize(neg_log_ratio, x0_polish, method='L-BFGS-B', 
                                   options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res_polish.success:
                h_polish = np.exp(res_polish.x)
                h_polish = h_polish / np.sum(h_polish)
                val_polish = compute_ratio(h_polish)
                if val_polish > best_val:
                    best_val = val_polish
                    best_h = h_polish
            
            # Coordinate descent polish
            if time.time() < DEADLINE - 1:
                h_cd = best_h.copy()
                for _ in range(50):
                    if time.time() > DEADLINE - 0.5:
                        break
                    improved = False
                    for j in range(N):
                        if h_cd[j] < 1e-12:
                            continue
                        # Try scaling h_cd[j]
                        for scale in [0.5, 0.8, 1.2, 1.5, 2.0, 0.1, 0.01]:
                            h_test = h_cd.copy()
                            h_test[j] *= scale
                            h_test = np.clip(h_test, 0, None)
                            if np.sum(h_test) < 1e-12:
                                continue
                            h_test /= np.sum(h_test)
                            val_test = compute_ratio(h_test)
                            if val_test > best_val:
                                best_val = val_test
                                best_h = h_test
                                h_cd = h_test
                                improved = True
                    if not improved:
                        break
        except Exception:
            pass

    # Ensure non-negative
    best_h = np.clip(best_h, 0, None)
    if np.sum(best_h) == 0:
         best_h = np.ones(N) / N
         
    return list(best_h)
