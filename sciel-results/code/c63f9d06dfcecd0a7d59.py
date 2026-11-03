import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def compute_ratio(h):
        """
        Computes the ratio: ||h||_2^2 / (||h||_1 * ||h||_inf * ||h*h||_inf)
        """
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        l_inf = np.max(h)
        if l1 < 1e-15 or l_inf < 1e-15:
            return 0.0
        
        # Autoconvolution
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        
        if max_conv < 1e-15:
            return 0.0
            
        return (l2_sq) / (l1 * l_inf * max_conv)

    def optimize_amplitudes(mask):
        """
        Given a boolean mask of active indices, optimize the amplitudes in [0, 1]
        to maximize the ratio.
        """
        indices = np.where(mask)[0]
        n_active = len(indices)
        if n_active == 0:
            return np.zeros(N), 0.0

        # Initial guess: 1.0 for all active
        x0 = np.ones(n_active)
        
        # Bounds for amplitudes
        bounds = [(0.0, 1.0) for _ in range(n_active)]
        
        def objective(x):
            h = np.zeros(N)
            h[indices] = x
            return -compute_ratio(h) # Minimize negative ratio

        # Use Nelder-Mead for non-smooth objective
        try:
            res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds,
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
            best_x = res.x
            best_score = -res.fun
        except Exception:
            best_x = x0
            best_score = -objective(x0)
            
        best_h = np.zeros(N)
        best_h[indices] = best_x
        return best_h, best_score

    # Initialize with a few candidate masks
    candidates = []
    
    # 1. Known Sidon-like set
    sidon_set = [0, 1, 4, 10, 18, 23, 30]
    mask1 = np.zeros(N, dtype=bool)
    mask1[sidon_set] = True
    candidates.append(mask1)
    
    # 2. Random sparse sets
    random.seed(42)
    for _ in range(10):
        size = random.randint(5, 12)
        s = random.sample(range(N), size)
        mask = np.zeros(N, dtype=bool)
        mask[s] = True
        candidates.append(mask)

    # Evaluate initial candidates
    best_h = None
    best_score = 0.0
    
    for mask in candidates:
        if time.time() > deadline:
            break
        h, score = optimize_amplitudes(mask)
        if score > best_score:
            best_score = score
            best_h = h.copy()

    if best_h is None:
        # Fallback
        best_h = np.zeros(N)
        best_h[0] = 1.0
        best_score = compute_ratio(best_h)

    # Simulated Annealing on the Mask
    current_mask = np.zeros(N, dtype=bool)
    current_mask[np.where(best_h > 1e-6)[0]] = True
    
    current_score = best_score
    T = 1.0
    T_min = 1e-4
    cooling = 0.999
    
    max_iterations = 2000
    
    for i in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Mutate the mask
        new_mask = current_mask.copy()
        move_type = random.random()
        
        if move_type < 0.33:
            # Add
            zeros = np.where(~current_mask)[0]
            if len(zeros) > 0:
                idx = random.choice(zeros)
                new_mask[idx] = True
        elif move_type < 0.66:
            # Remove
            ones = np.where(current_mask)[0]
            if len(ones) > 1: # Keep at least one
                idx = random.choice(ones)
                new_mask[idx] = False
        else:
            # Swap
            ones = np.where(current_mask)[0]
            zeros = np.where(~current_mask)[0]
            if len(ones) > 0 and len(zeros) > 0:
                idx_out = random.choice(ones)
                idx_in = random.choice(zeros)
                new_mask[idx_out] = False
                new_mask[idx_in] = True

        # Optimize amplitudes for the new mask
        new_h, new_score = optimize_amplitudes(new_mask)
        
        # Accept/Reject
        if new_score > current_score:
            current_mask = new_mask
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_h = new_h.copy()
        else:
            delta = current_score - new_score
            if delta < 1e-10:
                current_mask = new_mask
                current_score = new_score
            else:
                if random.random() < np.exp(-delta / T):
                    current_mask = new_mask
                    current_score = new_score
        
        T = max(T * cooling, T_min)
        
        # Restart if stuck
        if i % 500 == 0 and i > 0:
            if random.random() < 0.1:
                current_mask = np.zeros(N, dtype=bool)
                current_mask[np.where(best_h > 1e-6)[0]] = True
                current_score = best_score
                T = 0.5

    # Final polish: try to add small values to zero indices?
    # The optimizer already handles amplitudes in [0,1], so it might have set some to 0.
    # We can try a final coordinate ascent on the full vector.
    
    if best_h is not None:
        h_final = best_h.copy()
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                # Try small variations
                for delta in [-0.1, 0.1]:
                    h_test = h_final.copy()
                    h_test[i] = np.clip(h_test[i] + delta, 0, 1)
                    r_test = compute_ratio(h_test)
                    if r_test > compute_ratio(h_final):
                        h_final = h_test
                        improved = True
                        break
            if not improved:
                break
        
        final_score = compute_ratio(h_final)
        if final_score > best_score:
            best_h = h_final
            best_score = final_score

    return list(best_h)
