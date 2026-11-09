"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    # Safety margin: finish before 150s (evaluator limit), aim for 140s max compute
    deadline = start_time + 140.0
    
    # 1. Generate Thue-Morse sequence as initial guess
    # Thue-Morse has low autocorrelation sidelobes
    def thue_morse(n):
        t = np.zeros(n, dtype=np.int8)
        for i in range(1, n):
            t[i] = t[i >> 1] ^ (i & 1)
        return 2 * t - 1  # Map 0,1 to 1,-1
        
    s = thue_morse(N).astype(np.float64)
    
    # 2. Compute Autocorrelation using FFT
    def calc_autocorr(s):
        # Zero pad to 2N to avoid circular convolution artifacts
        f = np.fft.fft(s, 2*N)
        r = np.fft.ifft(f * np.conj(f))
        return r[:N].real
        
    R = calc_autocorr(s)
    
    # Metric: Max absolute value of sidelobes
    # For seed (all ones), R[k] = N - k. Max is N-1 = 399.
    # We want to minimize this value.
    
    def get_psl(R):
        return np.max(np.abs(R[1:]))
        
    current_max = get_psl(R)
    best_s = s.copy()
    best_R = R.copy()
    best_max = current_max
    
    # 3. Optimization Strategy: Simulated Annealing with Greedy Local Search
    # We combine the robustness of Simulated Annealing (to escape local minima)
    # with the precision of Greedy Local Search (to exploit the neighborhood structure).
    
    # Pre-allocate delta array for O(N) update checks in greedy step
    delta = np.zeros(N)
    
    # Temperature parameters
    T = 0.5
    alpha = 0.999
    T_min = 1e-5
    
    stagnant_count = 0
    rng = np.random.default_rng(42)
    
    while time.time() < deadline and T > T_min:
        
        # Check for stagnation. If we haven't improved the global best for a while,
        # we might be stuck in a deep local minimum.
        # However, SA itself handles this via T. 
        # Let's integrate a "Greedy Pass" if T is low or if we've been stuck.
        
        # --- Step A: Greedy Local Search (Best Improvement) ---
        # Try to find the single best flip that reduces the PSL.
        # This is expensive O(N^2) but very effective for refinement.
        # We only run this if we are close to convergence or periodically.
        
        # To keep it fast, we check for the best flip only every few SA steps or when T drops.
        # Alternatively, we can do a "first improvement" greedy step which is faster.
        
        # Let's do a hybrid:
        # 1. Try a random flip (SA move).
        # 2. If it improves, accept.
        # 3. If T is low, attempt a full greedy pass to polish.
        
        # Actually, the previous "Best Improvement" loop was too slow if done every iteration.
        # Let's do a "First Improvement" greedy step which is O(N^2) worst case but often faster,
        # or just rely on SA.
        
        # Let's stick to SA but with a "Greedy Restart" mechanism.
        
        # Random Flip
        j = rng.integers(0, N)
        val = s[j]
        
        # Compute Delta R for flipping s[j]
        # R[k] = sum_i s[i] s[i+k]
        # Flipping s[j] changes:
        # 1. Term where i=j: s[j] s[j+k] -> -s[j] s[j+k]. Change: -2 s[j] s[j+k]
        # 2. Term where i+k=j => i=j-k: s[j-k] s[j] -> s[j-k] (-s[j]). Change: -2 s[j] s[j-k]
        
        delta[:] = 0
        
        # Term 1: k in 0..N-1-j
        if N - 1 - j >= 0:
            delta[0:N-j] -= 2 * val * s[j:N]
            
        # Term 2: k in 0..j
        if j >= 0:
            delta[0:j+1] -= 2 * val * s[j::-1]
            
        R_new = R + delta
        new_max = np.max(np.abs(R_new[1:]))
        
        delta_score = new_max - current_max
        
        # Acceptance Criterion
        if delta_score < 0 or rng.random() < np.exp(-delta_score / T):
            # Accept
            s[j] = -s[j]
            R = R_new
            current_max = new_max
            
            if current_max < best_max:
                best_max = current_max
                best_s = s.copy()
                best_R = R.copy()
                stagnant_count = 0
            else:
                stagnant_count += 1
                
            # If we've improved the global best, we might want to do a quick greedy polish
            # to squeeze out more gains.
            if stagnant_count == 0:
                # Perform a limited greedy pass
                # Find the best single flip that improves the current PSL
                improved_greedy = True
                greedy_limit = 10 # Limit number of greedy steps to keep time bound
                steps = 0
                while improved_greedy and steps < greedy_limit and time.time() < deadline:
                    improved_greedy = False
                    best_flip_j = -1
                    best_flip_max = current_max
                    best_flip_R = None
                    
                    # To speed up, we don't need to check ALL N if we are just looking for *an* improvement.
                    # But checking all N gives the *best* improvement.
                    # For N=400, checking all N is 400 * (cost of delta calc).
                    # Delta calc is O(N). Total O(N^2) = 160k ops. Very fast in NumPy.
                    
                    for j_g in range(N):
                        val_g = s[j_g]
                        delta[:] = 0
                        if N - 1 - j_g >= 0:
                            delta[0:N-j_g] -= 2 * val_g * s[j_g:N]
                        if j_g >= 0:
                            delta[0:j_g+1] -= 2 * val_g * s[j_g::-1]
                        
                        R_g = R + delta
                        max_g = np.max(np.abs(R_g[1:]))
                        
                        if max_g < best_flip_max:
                            best_flip_max = max_g
                            best_flip_j = j_g
                            best_flip_R = R_g.copy()
                            
                    if best_flip_j != -1:
                        s[best_flip_j] = -s[best_flip_j]
                        R = best_flip_R
                        current_max = best_flip_max
                        improved_greedy = True
                        steps += 1
                        
                        if current_max < best_max:
                            best_max = current_max
                            best_s = s.copy()
                            best_R = R.copy()
                    else:
                        break
        else:
            # Reject: do nothing (s and R remain unchanged)
            stagnant_count += 1
            
        # Cool down
        T *= alpha
        
        # If we are stuck for a long time and T is still high, we might force a random restart
        # or a larger perturbation.
        if stagnant_count > 100:
            # Random restart: flip a few random bits to escape local minimum
            for _ in range(5):
                j_r = rng.integers(0, N)
                s[j_r] = -s[j_r]
            R = calc_autocorr(s)
            current_max = np.max(np.abs(R[1:]))
            if current_max < best_max:
                best_max = current_max
                best_s = s.copy()
                best_R = R.copy()
            stagnant_count = 0
            # Reheat slightly
            T = max(T, 0.5)

    return best_s
# EVOLVE-BLOCK-END
