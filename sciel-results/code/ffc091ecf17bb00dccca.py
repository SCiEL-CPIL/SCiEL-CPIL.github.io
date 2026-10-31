"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def compute_autocorr(f):
    """Compute autocorrelation C[k] = sum f[i]f[i+k]"""
    N = len(f)
    C = np.zeros(N)
    for k in range(N):
        C[k] = np.dot(f[:N-k], f[k:])
    return C

def greedy_flip_search(f_init, max_steps=2000, random_restarts=5):
    """
    Greedy Coordinate Descent (Sign Flipping) to minimize max(|C[k]|).
    """
    N = len(f_init)
    best_f = f_init.copy()
    best_score = np.inf
    
    for restart in range(random_restarts):
        f = f_init.copy() if restart == 0 else np.random.choice([-1.0, 1.0], size=N)
        C = compute_autocorr(f)
        current_score = np.max(np.abs(C))
        
        if current_score < best_score:
            best_score = current_score
            best_f = f.copy()
            
        for step in range(max_steps):
            # Find the lag k with maximum absolute autocorrelation
            k_max = np.argmax(np.abs(C))
            max_val = C[k_max]
            
            if abs(max_val) < 1e-12:
                break
                
            # Calculate the impact of flipping each f[j] on C[k_max]
            # C[k] = sum_i f[i] f[i+k]
            # Flip f[j] -> f[j] becomes -f[j]
            # Change in C[k] is -2 * f[j] * (f[j+k] + f[j-k])
            
            # Vectorized calculation of delta for all j
            # term1: f[j+k] if j+k < N else 0
            # term2: f[j-k] if j-k >= 0 else 0
            
            # Create indices
            j_idx = np.arange(N)
            
            # For term1 (f[j+k]): valid if j+k < N => j < N-k
            term1 = np.zeros(N)
            valid1 = j_idx < (N - k_max)
            term1[valid1] = f[j_idx[valid1] + k_max]
            
            # For term2 (f[j-k]): valid if j-k >= 0 => j >= k
            term2 = np.zeros(N)
            valid2 = j_idx >= k_max
            term2[valid2] = f[j_idx[valid2] - k_max]
            
            delta_C = -2.0 * f * (term1 + term2)
            
            new_C_val = max_val + delta_C
            
            # Find the flip that minimizes |new_C_val|
            j_best = np.argmin(np.abs(new_C_val))
            new_abs_val = np.abs(new_C_val[j_best])
            
            # If flipping doesn't reduce the peak, we are stuck locally
            if new_abs_val >= abs(max_val) - 1e-12:
                # Try a random flip to escape local minimum
                if np.random.random() < 0.1: # 10% chance to random flip
                    j_rand = np.random.choice(N)
                    # Apply random flip
                    f[j_rand] *= -1
                    # Update C incrementally
                    # Recalculate C is O(N^2), but we can do it less often or just accept the cost if N is small
                    # For N=400, O(N^2) is 160k ops. 2000 steps * 160k = 320M ops. Might be slow in Python loops.
                    # Let's use the incremental update for the chosen j.
                    
                    # Incremental update for flipping f[j]
                    # delta_C[k] = -2 * f[j] * (f[j+k] + f[j-k])
                    # Note: f[j] is the OLD value before flip
                    old_fj = f[j_rand]
                    f[j_rand] *= -1
                    
                    # Calculate delta for all k
                    delta_all = np.zeros(N)
                    for k in range(N):
                        t1 = f[j_rand + k] if j_rand + k < N else 0
                        t2 = f[j_rand - k] if j_rand - k >= 0 else 0
                        # The change is -2 * old_fj * (t1 + t2) ? 
                        # Wait, the formula derived earlier was for the NEW f[j].
                        # C_new = C_old - 2 * f_old[j] * (neighbors)
                        # Let's re-verify.
                        # C = sum f[i]f[i+k].
                        # If f[j] -> -f[j], the terms involving f[j] change sign.
                        # Term f[j]f[j+k] becomes -f[j]f[j+k]. Change: -2 f[j] f[j+k].
                        # Term f[j-k]f[j] becomes f[j-k](-f[j]). Change: -2 f[j] f[j-k].
                        # So delta = -2 * f_old[j] * (f[j+k] + f[j-k]).
                        # The neighbors f[j+k] and f[j-k] are the CURRENT values (unchanged).
                        delta_val = -2.0 * old_fj * (t1 + t2)
                        delta_all[k] = delta_val
                    C += delta_all
                    
                    current_score = np.max(np.abs(C))
                    if current_score < best_score:
                        best_score = current_score
                        best_f = f.copy()
                else:
                    break # Stuck, break to next restart
            
            else:
                # Apply the best flip
                old_fj = f[j_best]
                f[j_best] *= -1
                
                # Update C incrementally
                delta_all = np.zeros(N)
                for k in range(N):
                    t1 = f[j_best + k] if j_best + k < N else 0
                    t2 = f[j_best - k] if j_best - k >= 0 else 0
                    delta_val = -2.0 * old_fj * (t1 + t2)
                    delta_all[k] = delta_val
                C += delta_all
                
                current_score = np.max(np.abs(C))
                if current_score < best_score:
                    best_score = current_score
                    best_f = f.copy()
                    
    return best_f, best_score

def construct():
    np.random.seed(42)
    f_init = np.random.choice([-1.0, 1.0], size=N)
    
    best_f, _ = greedy_flip_search(f_init, max_steps=1000, random_restarts=10)
    
    # Normalize to sum=1
    s = np.sum(best_f)
    if s != 0:
        best_f = best_f / s
    else:
        # Fallback if sum is 0 (should not happen with odd N or random signs, but just in case)
        best_f = best_f + 1.0/N
        best_f = best_f / np.sum(best_f)
        
    return best_f
# EVOLVE-BLOCK-END
