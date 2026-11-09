import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    N = 24
    k_arr = np.arange(N)
    
    # FFT size for fast sup-norm approximation
    # Larger FFT gives better approximation of the true max
    FFT_SIZE = 32768 
    theta_grid = 2 * np.pi * np.arange(FFT_SIZE) / FFT_SIZE
    roots = np.exp(1j * theta_grid)
    
    # Precompute powers for specific theta evaluations if needed for refinement
    # But for SA, we use FFT which is O(N log N)
    
    def eval_sup_fft(c):
        """
        Approximate sup norm using FFT.
        This is fast and good for ranking candidates during search.
        """
        c_pad = np.zeros(FFT_SIZE, dtype=np.complex128)
        c_pad[:N] = c
        F = np.fft.fft(c_pad)
        return float(np.max(np.abs(F)))

    def eval_sup_precise(c, n_peaks=5):
        """
        Refine the sup norm by finding local maxima near the top FFT peaks.
        This is slower but more accurate.
        """
        c_pad = np.zeros(FFT_SIZE, dtype=np.complex128)
        c_pad[:N] = c
        F = np.fft.fft(c_pad)
        abs_F = np.abs(F)
        
        # Get indices of the largest peaks
        # We need to be careful with wrapping around 0 and FFT_SIZE
        top_indices = np.argsort(abs_F)[::-1][:n_peaks]
        
        best_val = 0.0
        dtheta = 2 * np.pi / FFT_SIZE
        
        for idx in top_indices:
            # Initial theta estimate from FFT bin
            theta = 2 * np.pi * idx / FFT_SIZE
            
            # Refine using Newton's method on |P(theta)|^2
            # We want to maximize |P|^2, so we minimize -|P|^2
            # Or just find where derivative is 0.
            
            # Derivative of |P|^2 = 2 Re(conj(P) P')
            # Second derivative is more complex, so we use a small step or 
            # golden section search in a small window around theta.
            
            # Let's do a small grid search around theta first to ensure we are in the basin
            # Window of +/- 2 bins
            local_best_theta = theta
            local_best_val = 0.0
            for offset in range(-2, 3):
                t = (2 * np.pi * (idx + offset) / FFT_SIZE) % (2 * np.pi)
                val = np.abs(np.dot(c, np.exp(1j * t * k_arr)))
                if val > local_best_val:
                    local_best_val = val
                    local_best_theta = t
            
            # Now refine local_best_theta with Newton/Secant
            theta_cur = local_best_theta
            for _ in range(10):
                z = np.exp(1j * theta_cur)
                P_val = np.dot(c, z**k_arr)
                P_prime = np.dot(c * k_arr, 1j * z**k_arr)
                
                # Gradient of |P|^2
                grad = 2.0 * np.real(np.conj(P_val) * P_prime)
                
                if abs(grad) < 1e-12:
                    break
                
                # Approximate Hessian with finite difference
                h = 1e-5
                z_h = np.exp(1j * (theta_cur + h))
                P_h = np.dot(c, z_h**k_arr)
                Pp_h = np.dot(c * k_arr, 1j * z_h**k_arr)
                grad_h = 2.0 * np.real(np.conj(P_h) * Pp_h)
                
                z_mh = np.exp(1j * (theta_cur - h))
                P_mh = np.dot(c, z_mh**k_arr)
                Pp_mh = np.dot(c * k_arr, 1j * z_mh**k_arr)
                grad_mh = 2.0 * np.real(np.conj(P_mh) * Pp_mh)
                
                hess = (grad_h - grad_mh) / (2 * h)
                
                if abs(hess) < 1e-12:
                    break
                
                step = -grad / hess
                # Limit step size to avoid divergence
                if abs(step) > 0.1:
                    step = np.sign(step) * 0.1
                theta_cur = (theta_cur + step) % (2 * np.pi)
            
            val = float(np.abs(np.dot(c, np.exp(1j * theta_cur * k_arr))))
            if val > best_val:
                best_val = val
                
        return best_val

    # Simulated Annealing
    def simulated_annealing(initial_c, t_start=1.0, t_end=0.01, n_iters=20000):
        c = initial_c.copy()
        current_score = eval_sup_fft(c)
        best_c = c.copy()
        best_score = current_score
        
        T = t_start
        cooling_rate = (t_end / t_start) ** (1.0 / n_iters)
        
        for i in range(n_iters):
            # Propose move: flip 1 or 2 bits
            if np.random.random() < 0.8:
                idx = np.random.randint(N)
                c[idx] *= -1
                new_c = c
            else:
                idx1 = np.random.randint(N)
                idx2 = np.random.randint(N)
                while idx2 == idx1:
                    idx2 = np.random.randint(N)
                c[idx1] *= -1
                c[idx2] *= -1
                new_c = c
            
            new_score = eval_sup_fft(new_c)
            delta = new_score - current_score
            
            if delta < 0 or np.random.random() < math.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_c = c.copy()
            else:
                # Reject move
                if np.random.random() < 0.8:
                    c[idx] *= -1
                else:
                    c[idx1] *= -1
                    c[idx2] *= -1
            
            T *= cooling_rate
            
            # Check time
            if time.time() > deadline - 10.0:
                break
                
        return best_c, best_score

    # Generate diverse initial candidates
    np.random.seed(42)
    
    # 1. Random starts
    # 2. GRS (Golay-Riordan Sequence) and variants
    def grs(n):
        a = np.zeros(n)
        a[0] = 1.0
        for i in range(1, n):
            kk = i >> 1
            if i & 1:
                a[i] = -a[kk]
            else:
                a[i] = a[kk]
        return a
    
    g = grs(N)
    
    candidates = [g.copy()]
    # Add random starts
    for _ in range(20):
        candidates.append(np.random.choice([-1.0, 1.0], size=N))
    
    best_overall_c = None
    best_overall_score = float('inf')
    
    # Run SA for each candidate
    # We have ~140 seconds. 
    # Each SA run with 20k iters takes roughly 20k * (FFT time + overhead).
    # FFT size 32768 is fast, maybe ~10-20us. 20k * 20us = 0.4s.
    # So we can do many runs.
    
    time_per_sa_estimate = 0.5
    max_sas = int((deadline - time.time() - 10.0) / time_per_sa_estimate)
    
    # Ensure we at least run a few
    max_sas = max(10, min(max_sas, 100)) # Cap to avoid infinite loop if time estimate is off
    
    # Shuffle candidates to ensure we try different starts
    np.random.shuffle(candidates)
    
    # If we have more SA slots than candidates, we can re-run with different seeds or longer runs
    # Let's just run SA on the provided candidates, and if time remains, do more random restarts
    
    runs_done = 0
    while runs_done < max_sas and time.time() < deadline - 10.0:
        if runs_done < len(candidates):
            init_c = candidates[runs_done]
        else:
            init_c = np.random.choice([-1.0, 1.0], size=N)
            
        c_res, score_res = simulated_annealing(init_c, t_start=0.5, t_end=0.001, n_iters=30000)
        
        if score_res < best_overall_score:
            best_overall_score = score_res
            best_overall_c = c_res.copy()
            
        runs_done += 1
        
    # Final Polish
    # The best_c found by SA is based on FFT approximation.
    # We need to verify and potentially improve it with precise evaluation.
    # Also, do a local search (single bit flips) using precise evaluation on the best candidate.
    
    if best_overall_c is None:
        best_overall_c = g.copy()
    
    # Local Search with Precise Evaluation
    # This is slow, so we limit it.
    improved = True
    passes = 0
    while improved and passes < 5 and time.time() < deadline - 5.0:
        improved = False
        passes += 1
        cur_score = eval_sup_precise(best_overall_c)
        
        # Try single bit flips
        for j in range(N):
            if time.time() > deadline - 5.0:
                break
            trial = best_overall_c.copy()
            trial[j] *= -1
            new_score = eval_sup_precise(trial)
            if new_score < cur_score - 1e-12:
                best_overall_c = trial
                cur_score = new_score
                improved = True
        
        # If no improvement with single bits, try double bits (slower, maybe skip if time is tight)
        if not improved and passes < 3 and time.time() < deadline - 5.0:
             for j in range(N):
                for l in range(j+1, N):
                    if time.time() > deadline - 5.0:
                        break
                    trial = best_overall_c.copy()
                    trial[j] *= -1
                    trial[l] *= -1
                    new_score = eval_sup_precise(trial)
                    if new_score < cur_score - 1e-12:
                        best_overall_c = trial
                        cur_score = new_score
                        improved = True
                        break
                if improved:
                    break

    # Convert to integer list
    out = [1 if v >= 0 else -1 for v in best_overall_c]
    return out

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(res)
