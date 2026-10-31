import numpy as np
import time
import random

N = 50
DEADLINE = 140.0  # 150s limit, 10s margin for safety

def compute_ratio(h):
    """
    Computes the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    h is a non-negative vector.
    """
    h = np.asarray(h, dtype=np.float64)
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

def construct():
    start_time = time.time()
    
    # Precompute DCT-II basis matrix for the first K modes.
    # We restrict the signal to be a linear combination of the first K DCT modes.
    # h = B_K @ c, where B_K is N x K.
    # This ensures h is smooth (low frequency), which generally keeps the auto-convolution
    # peak lower relative to the mass compared to spiky signals, while allowing 
    # some concentration via the DC and low-frequency components.
    
    # Let's try a few different subspace sizes K.
    # K=1 is uniform (ratio 1/N).
    # K=5, 10, 20 allow for smooth variations.
    
    n = np.arange(N)
    
    def get_dct_basis(K):
        B = np.zeros((N, K))
        for k in range(K):
            if k == 0:
                B[:, k] = np.sqrt(1.0 / N)
            else:
                B[:, k] = np.sqrt(2.0 / N) * np.cos(np.pi / N * (n + 0.5) * k)
        return B

    best_h = np.ones(N) / N
    best_val = compute_ratio(best_h)
    
    # Strategy: Simulated Annealing on the coefficients c.
    # Objective: Maximize Ratio(h(c))
    # Constraint: h(c) >= 0.
    # Since h = B @ c, we can clip h to be non-negative before evaluating, 
    # but this makes the objective non-smooth. 
    # Alternative: Penalize negative values heavily in the objective function.
    # Let's use a penalty approach for smoothness.
    
    def evaluate_c(c, B):
        h = B @ c
        # Penalty for negative values
        neg_penalty = np.sum(np.clip(-h, 0, None)**2)
        if neg_penalty > 1e-6:
            # If significantly negative, return a bad score
            # The ratio calculation will fail or be meaningless if we clip hard
            # But for SA, we want a continuous landscape.
            # Let's just clip and compute, but add a penalty to the score
            # to discourage exploring regions where h < 0.
            pass
        
        h_clipped = np.clip(h, 0, None)
        r = compute_ratio(h_clipped)
        
        if r < 0:
            return -1e10 + neg_penalty
        
        # Subtract penalty to steer away from negative h
        return r - 10.0 * neg_penalty

    # We will run SA for a few different K values.
    # Time budget allocation:
    # K=5: 20% time
    # K=10: 30% time
    # K=20: 50% time
    
    subspace_configs = [
        (5, 0.20),
        (10, 0.30),
        (20, 0.50)
    ]
    
    for K, time_frac in subspace_configs:
        if time.time() > DEADLINE:
            break
            
        B = get_dct_basis(K)
        
        # Initial guess: DC component dominant
        c0 = np.zeros(K)
        c0[0] = np.sqrt(N) # h ~ 1
        # Add small random perturbations
        c = c0 + np.random.randn(K) * 0.1
        
        current_h = np.clip(B @ c, 0, None)
        current_score = evaluate_c(c, B)
        
        best_c_in_run = c.copy()
        best_score_in_run = current_score
        best_h_in_run = current_h.copy()
        
        # Simulated Annealing Parameters
        T0 = 0.01  # Initial temperature
        T_end = 0.0001
        cooling_rate = 0.995
        max_iters = 50000 # Adjust based on time
        
        # Estimate time per iteration to fit budget
        # Roughly: (time_frac * (DEADLINE - start_time)) / max_iters
        # We will just run until time runs out for this K
        
        target_time = start_time + time_frac * (DEADLINE - start_time)
        
        T = T0
        it = 0
        
        while time.time() < target_time and it < max_iters:
            it += 1
            
            # Propose new c
            step_size = 0.1 * (T / T0 + 0.1)
            c_new = c + np.random.randn(K) * step_size
            
            # Ensure DC is positive enough to potentially allow non-negative h
            # If c_new[0] is too small, h might be mostly negative.
            # We don't strictly enforce it, the penalty handles it.
            
            score_new = evaluate_c(c_new, B)
            
            # Acceptance probability
            if score_new > current_score:
                accept = True
            else:
                delta = score_new - current_score
                if T > 0:
                    accept = random.random() < np.exp(delta / T)
                else:
                    accept = False
            
            if accept:
                c = c_new
                current_score = score_new
                
                # Update best
                if current_score > best_score_in_run:
                    best_score_in_run = current_score
                    best_c_in_run = c.copy()
                    best_h_in_run = np.clip(B @ c, 0, None)
                    
                    # Update global best
                    # We need the actual ratio of the clipped h
                    actual_ratio = compute_ratio(best_h_in_run)
                    if actual_ratio > best_val:
                        best_val = actual_ratio
                        best_h = best_h_in_run.copy()
            
            # Cool down
            T *= cooling_rate
            if T < T_end:
                T = T_end
        
    # Final Refinement: Local search on the best h found
    # Using Nelder-Mead on the full 50-dim space might be too slow/expensive.
    # Instead, refine the coefficients c of the best subspace if possible,
    # or just do a few random perturbations on h.
    
    if time.time() < DEADLINE - 5:
        # Simple local refinement on h using small random steps
        h_ref = best_h.copy()
        for _ in range(1000):
            if time.time() > DEADLINE - 2:
                break
            h_trial = h_ref + np.random.randn(N) * 0.01
            h_trial = np.clip(h_trial, 0, None)
            # Renormalize to keep L1 roughly constant? 
            # The ratio is scale invariant, so normalization doesn't change it.
            r = compute_ratio(h_trial)
            if r > best_val:
                best_val = r
                best_h = h_trial
                h_ref = h_trial.copy()
            else:
                # Revert
                pass

    # Ensure output is valid
    best_h = np.clip(best_h, 0, None)
    if np.sum(best_h) < 1e-12:
        best_h = np.ones(N) / N
        
    return list(best_h)
