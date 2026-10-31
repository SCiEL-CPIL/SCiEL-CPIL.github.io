"""Improved: Sidon-set structured initialization + L-BFGS-B refinement with smooth max,
plus a local search swap pass to escape local minima."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # safety margin before 120s kill

def is_sidon(A):
    """Check if A is a Sidon set (all sums a+b with a<=b are distinct)."""
    sums = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def find_sidon_set_greedy(n):
    """Greedy construction of a Sidon set."""
    A = []
    used_sums = set()
    for x in range(n):
        valid = True
        new_sums = []
        for a in A:
            s = a + x
            if s in used_sums:
                valid = False
                break
            new_sums.append(s)
        if 2 * x in used_sums:
            valid = False
        
        if valid:
            A.append(x)
            used_sums.update(new_sums)
            used_sums.add(2 * x)
    return A

def local_search_swap(h, compute_ratio, max_iter=5000, time_limit=10.0, start_time=None):
    """Local search: try swapping pairs of non-zero entries or moving mass between entries.
    Accept if ratio improves."""
    if start_time is None:
        start_time = time.time()
    h = h.copy()
    current_val = compute_ratio(h)
    n_nonzero = np.sum(h > 1e-12)
    
    # Build list of indices with significant mass
    indices = np.where(h > 1e-12)[0]
    
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        if time.time() - start_time > time_limit:
            break
        
        # Try moving mass from one index to another
        for i in range(len(indices)):
            for j in range(len(indices)):
                if i == j:
                    continue
                a_idx = indices[i]
                b_idx = indices[j]
                
                # Try small transfer
                for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
                    delta = frac * h[a_idx]
                    if delta < 1e-15:
                        continue
                    new_h = h.copy()
                    new_h[a_idx] -= delta
                    new_h[b_idx] += delta
                    if new_h[a_idx] < -1e-15:
                        continue
                    new_h = np.maximum(new_h, 0.0)
                    val = compute_ratio(new_h)
                    if val > current_val + 1e-12:
                        h = new_h
                        current_val = val
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        # Also try adding a new non-zero entry
        if not improved:
            for k in range(N):
                if h[k] > 1e-12:
                    continue
                # Try adding a small amount
                for amt in [0.01, 0.05, 0.1]:
                    new_h = h.copy()
                    new_h[k] = amt
                    # Normalize
                    new_h /= np.sum(new_h)
                    val = compute_ratio(new_h)
                    if val > current_val + 1e-12:
                        h = new_h
                        current_val = val
                        improved = True
                        break
                if improved:
                    break
    
    return h, current_val

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    
    def elapsed():
        return time.time() - start
    
    def compute_ratio(h):
        """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
        h = np.asarray(h, dtype=np.float64)
        l1 = np.sum(h)
        if l1 < 1e-300:
            return 0.0
        l2_sq = np.sum(h * h)
        hh = np.convolve(h, h, mode='full')
        hh_max = np.max(hh)
        if hh_max < 1e-300:
            return 0.0
        return l2_sq / (l1 * hh_max)
    
    def neg_ratio(params, beta=500.0):
        """Negative ratio with smooth max approximation for better gradient."""
        h = np.exp(params)
        h = h / np.sum(h)
        l1 = np.sum(h)
        if l1 < 1e-300:
            return 0.0
        l2_sq = np.sum(h * h)
        hh = np.convolve(h, h, mode='full')
        # Use log-sum-exp for smooth max
        hh_shifted = hh - np.max(hh)
        hh_max_smooth = (1.0 / beta) * np.log(np.sum(np.exp(beta * hh_shifted))) + np.max(hh)
        if hh_max_smooth < 1e-300:
            return 0.0
        return -l2_sq / (l1 * hh_max_smooth)
    
    def neg_ratio_exact(params):
        """Negative ratio with exact max (for final polish)."""
        h = np.exp(params)
        h = h / np.sum(h)
        l1 = np.sum(h)
        if l1 < 1e-300:
            return 0.0
        l2_sq = np.sum(h * h)
        hh = np.convolve(h, h, mode='full')
        hh_max = np.max(hh)
        if hh_max < 1e-300:
            return 0.0
        return -l2_sq / (l1 * hh_max)
    
    best_h = np.ones(N) / N
    best_val = compute_ratio(best_h)
    
    candidates = []
    
    # 1. Single spike
    h1 = np.zeros(N)
    h1[N // 2] = 1.0
    candidates.append(h1)
    
    # 2. Two equal spikes
    h2 = np.zeros(N)
    h2[N // 4] = 0.5
    h2[3 * N // 4] = 0.5
    candidates.append(h2)
    
    # 3. Three equal spikes
    h3 = np.zeros(N)
    h3[N // 6] = 1.0 / 3
    h3[N // 2] = 1.0 / 3
    h3[5 * N // 6] = 1.0 / 3
    candidates.append(h3)
    
    # 4. Many equal spikes (10)
    h4 = np.zeros(N)
    positions = np.linspace(0, N - 1, 10).astype(int)
    h4[positions] = 1.0 / len(positions)
    candidates.append(h4)
    
    # 5. Uniform
    candidates.append(np.ones(N) / N)
    
    # 6. Two adjacent spikes
    h6 = np.zeros(N)
    h6[N // 2 - 1] = 0.5
    h6[N // 2] = 0.5
    candidates.append(h6)
    
    # 7. Wide plateau
    h7 = np.zeros(N)
    h7[N//4 : 3*N//4] = 1.0 / (N//2)
    candidates.append(h7)
    
    # 8. Sidon set based initialization
    A_greedy = find_sidon_set_greedy(N)
    h_sidon = np.zeros(N)
    for idx in A_greedy:
        h_sidon[idx] = 1.0
    if np.sum(h_sidon) > 0:
        h_sidon /= np.sum(h_sidon)
        candidates.append(h_sidon)
    
    # Try to find a larger Sidon set with random restarts
    best_A = A_greedy
    for target_size in range(len(A_greedy) + 1, 10):
        found = False
        for _ in range(300):
            if time.time() > start + DEADLINE - 20:
                break
            A_try = sorted(np.random.choice(N, target_size, replace=False))
            if is_sidon(A_try):
                if target_size > len(best_A):
                    best_A = A_try
                    found = True
                if found:
                    break
        if found:
            continue
        else:
            break
    
    if len(best_A) > len(A_greedy):
        h_sidon_large = np.zeros(N)
        for idx in best_A:
            h_sidon_large[idx] = 1.0
        h_sidon_large /= np.sum(h_sidon_large)
        candidates.append(h_sidon_large)
    
    # 9. Random sparse
    rng = np.random.default_rng(42)
    for _ in range(15):
        h_rand = np.zeros(N)
        k = rng.integers(1, 18)
        idx = rng.choice(N, size=k, replace=False)
        h_rand[idx] = 1.0 / k
        candidates.append(h_rand)
    
    # 10. Gaussian-like
    x = np.arange(N)
    h9 = np.exp(-0.5 * ((x - N/2) / (N/4))**2)
    h9 /= np.sum(h9)
    candidates.append(h9)
    
    # 11. Bimodal Gaussian
    h10 = 0.5 * np.exp(-0.5 * ((x - N/4) / (N/8))**2) + 0.5 * np.exp(-0.5 * ((x - 3*N/4) / (N/8))**2)
    h10 /= np.sum(h10)
    candidates.append(h10)
    
    # 12. Alternating pattern
    h12 = np.zeros(N)
    h12[::2] = 1.0
    h12 /= np.sum(h12)
    candidates.append(h12)
    
    # 13. Block pattern
    h13 = np.zeros(N)
    h13[0:10] = 1.0
    h13[40:50] = 1.0
    h13 /= np.sum(h13)
    candidates.append(h13)
    
    # 14. Sidon with unequal weights (try different weight distributions)
    if len(best_A) >= 2:
        for w_scheme in ['uniform', 'geometric', 'random']:
            h_w = np.zeros(N)
            if w_scheme == 'uniform':
                for idx in best_A:
                    h_w[idx] = 1.0
            elif w_scheme == 'geometric':
                weights = np.geomspace(1.0, 0.3, len(best_A))
                for i, idx in enumerate(best_A):
                    h_w[idx] = weights[i]
            else:
                weights = rng.random(len(best_A)) + 0.1
                for i, idx in enumerate(best_A):
                    h_w[idx] = weights[i]
            if np.sum(h_w) > 0:
                h_w /= np.sum(h_w)
                candidates.append(h_w)
    
    # Evaluate all candidates and keep best
    for h_cand in candidates:
        val = compute_ratio(h_cand)
        if val > best_val:
            best_val = val
            best_h = h_cand.copy()
    
    # Refine best candidates with L-BFGS-B using smooth objective
    scored = [(compute_ratio(h), h) for h in candidates]
    scored.sort(key=lambda x: -x[0])
    
    n_refine = min(15, len(scored))
    for i in range(n_refine):
        if elapsed() > DEADLINE - 20:
            break
        
        h_init = scored[i][1]
        h_pos = h_init + 1e-10
        x0 = np.log(h_pos)
        
        try:
            res = minimize(neg_ratio, x0, method='L-BFGS-B',
                          options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-14})
            if elapsed() > DEADLINE - 15:
                break
            h_opt = np.exp(res.x)
            h_opt = h_opt / np.sum(h_opt)
            val = compute_ratio(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Local search swap refinement on the best solution
    if elapsed() < DEADLINE - 10:
        h_ls, val_ls = local_search_swap(best_h, compute_ratio, max_iter=8000, time_limit=15.0, start_time=start)
        if val_ls > best_val:
            best_val = val_ls
            best_h = h_ls.copy()
    
    # Final polish from the best solution found with tighter tolerance
    if elapsed() < DEADLINE - 5:
        try:
            h_pos = best_h + 1e-10
            x0 = np.log(h_pos)
            res = minimize(neg_ratio, x0, method='L-BFGS-B',
                          options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-16})
            h_opt = np.exp(res.x)
            h_opt = h_opt / np.sum(h_opt)
            val = compute_ratio(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # One more local search pass after final polish
    if elapsed() < DEADLINE - 3:
        h_ls2, val_ls2 = local_search_swap(best_h, compute_ratio, max_iter=5000, time_limit=5.0, start_time=start)
        if val_ls2 > best_val:
            best_val = val_ls2
            best_h = h_ls2.copy()
    
    # Ensure non-negative and return as list
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
# EVOLVE-BLOCK-END
