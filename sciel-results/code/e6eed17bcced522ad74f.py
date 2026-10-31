import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    # Autoconvolution via FFT for efficiency
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    conv = np.fft.ifft(H * H).real
    # Autoconvolution of length N with itself has length 2N-1
    # The FFT gives circular conv, but for zero-padded it's linear
    conv_peak = np.max(conv[:2*N-1])
    if conv_peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_peak)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = np.ones(N) / N
    best_score = compute_ratio(best_h)
    
    def optimize_weights_for_support(support_mask):
        """
        Given a binary support mask, optimize the weights of the non-zero elements
        to maximize the ratio.
        Ratio = sum(h^2) / (sum(h) * max(conv))
        Since sum(h) is fixed (scale invariance), we can set sum(h)=1.
        """
        indices = np.where(support_mask > 0)[0]
        k = len(indices)
        if k == 0:
            return 0.0, np.zeros(N)
        
        def objective(w):
            h = np.zeros(N)
            h[indices] = w
            l1 = np.sum(h)
            if l1 < 1e-15:
                return 0.0
            l2_sq = np.sum(h*h)
            n_fft = 2 * N
            H = np.fft.fft(h, n=n_fft)
            conv = np.fft.ifft(H * H).real
            conv_peak = np.max(conv[:2*N-1])
            if conv_peak < 1e-15:
                return 0.0
            return - (l2_sq / (l1 * conv_peak)) # Minimize negative ratio
        
        # Initial guess: uniform weights on support
        w0 = np.ones(k) / k
        
        # Bounds: w >= 0
        bounds = [(0, None)] * k
        
        # Constraints: sum(w) = 1
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        try:
            res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 100, 'ftol': 1e-6})
            if res.success:
                h_opt = np.zeros(N)
                h_opt[indices] = res.x
                score = -res.fun
                return score, h_opt
        except Exception:
            pass
        return 0.0, np.zeros(N)

    def greedy_sidon_search(num_restarts=100):
        """
        Structural change: Use a greedy heuristic to find large Sidon-like sets.
        The 'addition graph' concept: we want to avoid sum collisions.
        A Sidon set in Z_N ensures all pair sums are distinct (modulo N).
        This minimizes the number of terms contributing to each bin of the autoconvolution.
        """
        best_local_score = 0.0
        best_local_h = np.zeros(N)
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Random permutation of indices
            indices = list(range(N))
            random.shuffle(indices)
            
            selected = []
            used_sums = set()
            
            for idx in indices:
                # Check if adding idx creates sum collisions
                # Sums to check: idx+idx, idx+j for j in selected
                new_sums = [(2*idx) % N]
                for j in selected:
                    new_sums.append((idx+j) % N)
                
                # If any of these sums are already in used_sums, it's a collision.
                # For a strict Sidon set, we require NO collisions.
                if any(s in used_sums for s in new_sums):
                    continue
                    
                selected.append(idx)
                for s in new_sums:
                    used_sums.add(s)
                    
                if len(selected) >= 15: # Cap size for optimization speed
                    break
                    
            if len(selected) < 3:
                continue
                
            mask = np.zeros(N)
            mask[selected] = 1.0
            
            score, h_opt = optimize_weights_for_support(mask)
            if score > best_local_score:
                best_local_score = score
                best_local_h = h_opt
                
        return best_local_score, best_local_h

    # 1. Try Greedy Sidon Search
    score_sidon, h_sidon = greedy_sidon_search(num_restarts=50)
    if score_sidon > best_score:
        best_score = score_sidon
        best_h = h_sidon

    # 2. Try Random Sparse Sets with Density Control
    for _ in range(50):
        if time.time() > deadline:
            break
        k = random.randint(5, 15)
        indices = np.random.choice(N, k, replace=False)
        mask = np.zeros(N)
        mask[indices] = 1.0
        
        score, h_opt = optimize_weights_for_support(mask)
        if score > best_score:
            best_score = score
            best_h = h_opt

    # 3. Final Polish: If we have time, try a few more random restarts of the optimization
    # on the best support found.
    if time.time() < deadline - 5:
        best_indices = np.where(best_h > 1e-6)[0]
        if len(best_indices) > 0:
            # Perturb the weights slightly and re-optimize
            mask = np.zeros(N)
            mask[best_indices] = 1.0
            score, h_opt = optimize_weights_for_support(mask)
            if score > best_score:
                best_score = score
                best_h = h_opt

    return best_h.tolist()

if __name__ == "__main__":
    # Simple test
    h = construct()
    print(f"Score: {compute_ratio(h)}")
