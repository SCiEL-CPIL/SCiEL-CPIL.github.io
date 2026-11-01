import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We stop at 140s to ensure return.
    deadline = start_time + 140.0 
    
    # Pre-compute FFT size for speed
    n_fft = 1 << (2 * N - 1).bit_length()
    
    def compute_acf(h):
        """Compute linear autocorrelation using FFT."""
        F = np.fft.fft(h, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real[:N]
        return acf

    def metric(h):
        """
        The task mentions 'upper bound on C3'.
        In the context of autocorrelation inequalities (e.g., relating to C3 or similar constants),
        the metric is typically the maximum absolute value of the sidelobes (lag 1 to N-1).
        Sometimes it's just the max positive, but 'absolute value' is safer given 'signed step function'.
        Let's assume metric = max(|acf[1:N]|).
        """
        acf = compute_acf(h)
        return np.max(np.abs(acf[1:N]))

    def spectral_init(rng):
        """
        Initialize using a spectral mask that encourages low peaks.
        We start with a flat spectrum (white noise) but modulate it to reduce sidelobes.
        A common trick is to start with a Barker-like or Golay-like structure if known,
        but for N=400, we can use a 'smooth' phase spectrum.
        
        Alternative: Start with a known good pattern if available, or random.
        Since we don't have a hardcoded N=400 optimal sequence, we use a 
        'low-pass' spectral initialization to reduce high-frequency noise which 
        often contributes to high sidelobes in the time domain.
        """
        # Generate random phase
        phases = rng.uniform(0, 2*np.pi, n_fft)
        # Create a magnitude spectrum that is flat but with a slight roll-off
        # to reduce high-frequency components which cause sharp transitions and high sidelobes?
        # Actually, for binary sequences, the spectrum is fixed in magnitude if we just flip bits? 
        # No, flipping bits changes the spectrum.
        
        # Let's try a different initialization: 
        # Start with a random binary sequence, then project onto the "low peak" subspace?
        # Or just use random restarts with a strong local search.
        
        # Let's use a "Golay-like" construction if possible, or just random.
        # Given N=400 is not a power of 2, standard Golay doesn't fit directly.
        
        # Let's stick to random initialization but use Simulated Annealing which is better 
        # at global exploration than Greedy.
        return rng.choice([1, -1], N)

    def simulated_annealing(h0, rng, time_budget):
        """
        Simulated Annealing to minimize the peak sidelobe.
        """
        current_h = h0.copy()
        current_val = metric(current_h)
        best_h = current_h.copy()
        best_val = current_val
        
        # Initial temperature
        # The change in metric from a single flip is roughly O(sqrt(N)) or less.
        # Let's start T high enough to accept most moves.
        T = 5.0 
        T_min = 0.01
        cooling_rate = 0.9995 # Adjust based on time
        
        # Pre-compute indices for fast flip evaluation? 
        # A flip at index i changes the ACF.
        # We can update the ACF incrementally to save time.
        
        acf = compute_acf(current_h)
        
        it = 0
        no_improve_count = 0
        max_no_improve = 5000 # Restart or cool faster if stuck
        
        while time.time() < deadline - 5.0:
            # Choose a random index to flip
            i = rng.randint(N)
            
            # Calculate new ACF incrementally
            # r_k = sum h[j]h[j+k]
            # Flip h[i]:
            # delta_r_k = -2 * h[i] * (h[i-k] + h[i+k]) with bounds
            # This is O(N) per flip.
            
            new_h = current_h.copy()
            new_h[i] = -new_h[i]
            
            # Instead of full FFT, update ACF?
            # For N=400, FFT is very fast (~10us). 
            # 100,000 iterations * 10us = 1 second. 
            # So we can afford full FFT for simplicity and robustness.
            
            new_val = metric(new_h)
            
            delta = new_val - current_val
            
            # Accept if better, or with probability exp(-delta/T)
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_h = new_h
                current_val = new_val
                
                if current_val < best_val:
                    best_h = current_h.copy()
                    best_val = current_val
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                # Cool down
                T = max(T_min, T * cooling_rate)
            else:
                no_improve_count += 1
                
            it += 1
            
            # If stuck for a long time, reheat slightly or restart?
            if no_improve_count > max_no_improve:
                # Reheat
                T = max(T, 2.0)
                no_improve_count = 0
                cooling_rate = 0.9999 # Slow down cooling to explore more
        
        return best_h, best_val

    # Main loop: Multiple restarts
    rng = np.random.RandomState(42)
    best_overall_h = None
    best_overall_val = np.inf
    
    # Allocate time for multiple runs
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        h0 = spectral_init(rng)
        
        # Run SA for a portion of the time
        # Let's give each restart ~10 seconds
        h_sa, val_sa = simulated_annealing(h0, rng, time.time() + 10.0)
        
        if val_sa < best_overall_val:
            best_overall_val = val_sa
            best_overall_h = h_sa
    
    # Final polish with Greedy on the best result
    if best_overall_h is not None:
        # A quick greedy pass to remove any small imperfections
        h_final = best_overall_h.copy()
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            acf = compute_acf(h_final)
            current_max = np.max(np.abs(acf[1:N]))
            
            # Find best flip
            best_gain = 0
            best_idx = -1
            
            # Vectorized flip evaluation
            i_idx = np.arange(N)[:, None]
            k_idx = np.arange(N)[None, :]
            
            idx1 = i_idx - k_idx
            idx2 = i_idx + k_idx
            
            valid1 = (idx1 >= 0) & (idx1 < N)
            valid2 = (idx2 >= 0) & (idx2 < N)
            
            h_safe_idx1 = np.where(valid1, idx1, 0)
            h_safe_idx2 = np.where(valid2, idx2, 0)
            
            h1 = np.where(valid1, h_final[h_safe_idx1], 0.0)
            h2 = np.where(valid2, h_final[h_safe_idx2], 0.0)
            
            S = h1 + h2
            delta_matrix = -2.0 * h_final[:, None] * S
            new_acf_matrix = acf[None, :] + delta_matrix
            new_abs_acf_matrix = np.abs(new_acf_matrix[:, 1:N])
            new_maxs = np.max(new_abs_acf_matrix, axis=1)
            
            min_val = np.min(new_maxs)
            
            if min_val < current_max:
                best_idx = np.argmin(new_maxs)
                h_final[best_idx] = -h_final[best_idx]
                improved = True
        
        best_overall_h = h_final
    
    if best_overall_h is None:
        return rng.choice([1, -1], N)
        
    return best_overall_h
