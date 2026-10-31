"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    
    def compute_score(heights):
        """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
        h = np.array(heights)
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 0.0
        # Autoconvolution (linear, full mode)
        conv = np.convolve(h, h, mode='full')
        linf = np.max(conv)
        if linf < 1e-12:
            return 0.0
        return l2_sq / (l1 * linf)
    
    def neg_score(heights):
        """Negative score for minimization."""
        return -compute_score(heights)
    
    def is_sidon(lst):
        """Check if a list is a Sidon set (all pair sums are distinct)."""
        sums = set()
        n = len(lst)
        for i in range(n):
            for j in range(i, n):
                s = lst[i] + lst[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def find_sidon_set_greedy(n, target_size, start=0):
        """Find a Sidon set using greedy construction."""
        sidon = []
        for i in range(start, n):
            trial = sidon + [i]
            if is_sidon(trial):
                sidon.append(i)
                if len(sidon) >= target_size:
                    break
        return sidon
    
    def local_search_sidon(sidon_list, n, max_iters=5000):
        """Local search: swap elements to maximize score."""
        current = list(sidon_list)
        h = np.zeros(n)
        for i in current:
            h[i] = 1.0
        current_score = compute_score(h)
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 15:
            improved = False
            iters += 1
            # Try swapping each element with each non-element
            for idx in range(len(current)):
                for new_elem in range(n):
                    if new_elem in current:
                        continue
                    trial = list(current)
                    trial[idx] = new_elem
                    h_trial = np.zeros(n)
                    for i in trial:
                        h_trial[i] = 1.0
                    trial_score = compute_score(h_trial)
                    if trial_score > current_score + 1e-10:
                        current = trial
                        current_score = trial_score
                        improved = True
                        break
                if improved:
                    break
        
        return current, current_score
    
    def refine_with_lbfgs(heights, max_iter=200):
        """Refine heights using L-BFGS-B to maximize the ratio."""
        h0 = np.array(heights, dtype=float)
        # Normalize to avoid trivial solutions
        if np.sum(h0) < 1e-12:
            return h0
        
        result = minimize(
            neg_score,
            h0,
            method='L-BFGS-B',
            bounds=[(0, None)] * len(h0),
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return result.x
    
    def spectral_smooth(heights, alpha=0.1):
        """Apply spectral smoothing to distribute convolution energy."""
        h = np.array(heights, dtype=float)
        # Compute FFT
        H = np.fft.fft(h, n=2*N-1)
        # Apply low-pass filter to reduce high-frequency components
        freqs = np.fft.fftfreq(2*N-1)
        mask = np.exp(-alpha * (2*np.pi*freqs)**2)
        H_smoothed = H * mask
        h_smoothed = np.real(np.fft.ifft(H_smoothed, n=2*N-1))[:N]
        # Ensure nonnegative
        h_smoothed = np.maximum(h_smoothed, 0)
        return h_smoothed
    
    # Target Sidon set sizes to try
    best_heights = np.ones(N)
    best_score = compute_score(best_heights)
    
    # Try various Sidon set sizes with multiple random initializations
    for target_size in [8, 10, 12, 14, 16, 18, 20]:
        if time.time() > deadline - 25:
            break
        
        # Try multiple random starts
        for start_attempt in range(5):
            if time.time() > deadline - 20:
                break
            
            if start_attempt == 0:
                sidon = find_sidon_set_greedy(N, target_size, start=0)
            else:
                # Random subset initialization
                random_indices = random.sample(range(N), min(target_size * 2, N))
                sidon = []
                for i in random_indices:
                    trial = sidon + [i]
                    if is_sidon(trial) and len(sidon) < target_size:
                        sidon.append(i)
                if len(sidon) < 4:
                    continue
            
            if len(sidon) < 4:
                continue
            
            # Local search to improve the Sidon set
            improved_sidon, sidon_score = local_search_sidon(sidon, N)
            
            if sidon_score > best_score:
                best_score = sidon_score
                h = np.zeros(N)
                for i in improved_sidon:
                    h[i] = 1.0
                best_heights = h.copy()
        
        # Also try random sparse sets
        for trial in range(3):
            if time.time() > deadline - 15:
                break
            mask = np.zeros(N, dtype=bool)
            k = random.randint(6, min(target_size, N))
            mask[random.sample(range(N), k)] = True
            h = mask.astype(float)
            s = compute_score(h)
            if s > best_score:
                best_score = s
                best_heights = h.copy()
    
    # Refine with L-BFGS-B on the best support
    if time.time() < deadline - 10:
        support = np.where(best_heights > 0)[0]
        if len(support) > 0 and len(support) <= N:
            # Initialize with equal heights on support
            h_init = np.zeros(N)
            h_init[support] = 1.0 / np.sqrt(len(support))
            
            # Try L-BFGS-B refinement
            h_refined = refine_with_lbfgs(h_init, max_iter=300)
            h_refined = np.maximum(h_refined, 0)
            s_refined = compute_score(h_refined)
            
            if s_refined > best_score:
                best_score = s_refined
                best_heights = h_refined.copy()
            
            # Try spectral smoothing + refinement
            h_smoothed = spectral_smooth(h_init, alpha=0.05)
            if np.sum(h_smoothed) > 1e-12:
                h_smoothed_refined = refine_with_lbfgs(h_smoothed, max_iter=200)
                h_smoothed_refined = np.maximum(h_smoothed_refined, 0)
                s_smoothed = compute_score(h_smoothed_refined)
                if s_smoothed > best_score:
                    best_score = s_smoothed
                    best_heights = h_smoothed_refined.copy()
    
    # Final polish: try more L-BFGS-B iterations on the best solution
    if time.time() < deadline - 5:
        h_final = refine_with_lbfgs(best_heights, max_iter=500)
        h_final = np.maximum(h_final, 0)
        s_final = compute_score(h_final)
        if s_final > best_score:
            best_score = s_final
            best_heights = h_final.copy()
    
    # Ensure nonnegative and valid
    best_heights = np.maximum(best_heights, 0)
    if np.sum(best_heights) < 1e-12:
        return [1.0] * N
    
    return list(best_heights)
# EVOLVE-BLOCK-END
