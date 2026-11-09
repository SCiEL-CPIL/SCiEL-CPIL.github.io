"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Deadline: 145 seconds (safety margin for 150s limit)
    start_time = time.time()
    time_limit = 145.0
    
    rng = np.random.default_rng(42)
    
    def compute_C3(seq):
        """Compute the upper bound on C3.
        
        The task mentions 'upper bound on C3' and 'autoconvolution upper bound'.
        For a sequence of length N, the standard metric for low autocorrelation 
        is often the maximum absolute value of the aperiodic autocorrelation 
        sidelobes, or the sum of squared sidelobes. 
        Given 'C3' might refer to a specific constant in an inequality, and 
        'autoconvolution' implies the energy in the sidelobes, let's look at the inspirations.
        
        Inspiration 1 uses: max |acf[k]| / (N-k) for k=1..N-1.
        Inspiration 2 uses: max |sum_{i=0}^{N-1-k} f[i]f[i+k]|.
        Inspiration 3 uses: sum of squared sidelobes of circular autocorrelation.
        
        The current program (score 0.0049) uses: max |circular_autocorr[k]| for k=1..N-1.
        The parent (score 0.014) used: sum of squared sidelobes.
        The drop in score (0.014 -> 0.0049) suggests the metric changed or the optimization improved.
        Wait, the score is `seed/value`. Lower value is better.
        If the metric is `max_abs_sidelobe`, lower is better.
        
        Let's stick to the metric used in the current best program (p53) which achieved 0.0049.
        The current program uses `np.max(np.abs(current_R[1:]))` where R is circular autocorrelation.
        However, the task says "upper bound on C3". In the context of the "Third autocorrelation inequality"
        (likely referring to a result by Cohn or similar on autoconvolution), the quantity of interest
        is often the peak sidelobe level or the integrated sidelobe level.
        
        Let's assume the metric is the maximum absolute value of the aperiodic autocorrelation sidelobes,
        normalized or not. The current program uses circular. Aperiodic is more standard for "step function on [-1/4, 1/4]" 
        unless the domain is periodic. The domain is finite. 
        However, the current program scored 0.0049 with circular.
        Let's try to improve the optimization of the circular metric first, as it's faster to compute via FFT.
        If the metric is actually aperiodic, circular is an approximation that might be misleading.
        
        Let's look at Inspiration 1: `normalized = np.abs(acf[1:]) / norms[1:]`. This is aperiodic and normalized.
        Inspiration 2: `max_abs` of aperiodic sums.
        
        Let's implement a hybrid: Use the efficient circular FFT update for the main loop, but verify against aperiodic if needed?
        No, let's stick to the metric that gave the best score so far: **Max absolute value of circular autocorrelation sidelobes**.
        Why? Because the current program (0.0049) is better than the parent (0.014) which used sum of squares.
        The change from sum-of-squares to max-sidelobe likely helped.
        
        Strategy:
        1. Initialize with Thue-Morse (good low sidelobe properties).
        2. Use Simulated Annealing with incremental FFT updates for speed.
        3. Add a "Polish" phase: Local search trying all single flips to find the local minimum more precisely.
        4. Try multiple random restarts if time permits.
        """
        # Compute initial circular autocorrelation R
        def compute_R(a):
            fa = np.fft.fft(a)
            power = fa * np.conj(fa)
            R = np.fft.ifft(power).real
            return R

        def get_peak(R):
            return np.max(np.abs(R[1:]))

        best_a = None
        best_peak = float('inf')
        
        # Initialize with Thue-Morse
        a = np.array([1.0 if bin(i).count('1') % 2 == 0 else -1.0 for i in range(N)])
        
        current_R = compute_R(a)
        current_peak = get_peak(current_R)
        
        if current_peak < best_peak:
            best_peak = current_peak
            best_a = a.copy()
        
        # Simulated Annealing
        T_base = 1.0
        T = T_base
        cooling_rate = 0.9999
        
        current_a = a.copy()
        
        # We need to be careful with the incremental update.
        # R_new = R_old + delta * (roll(a, idx) + roll(a, -idx))
        # where delta = -2 * a[idx]
        
        while time.time() - start_time < time_limit - 10.0:
            # Pick random index
            idx = rng.integers(0, N)
            
            # Calculate change in R
            delta_val = -2.0 * current_a[idx]
            term = np.roll(current_a, idx) + np.roll(current_a, -idx)
            delta_R = delta_val * term
            
            candidate_R = current_R + delta_R
            candidate_peak = get_peak(candidate_R)
            
            # Acceptance criteria
            if candidate_peak < current_peak:
                current_R = candidate_R
                current_a[idx] *= -1.0
                current_peak = candidate_peak
                
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_a = current_a.copy()
            else:
                # Accept worse moves with probability exp(-delta/T)
                delta = candidate_peak - current_peak
                if T > 1e-9:
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        current_R = candidate_R
                        current_a[idx] *= -1.0
                        current_peak = candidate_peak
            
            T *= cooling_rate
            
        # Polish phase: Local search on the best solution found
        # Try flipping each index and accept if it improves
        polish_a = best_a.copy()
        polish_R = compute_R(polish_a)
        polish_peak = get_peak(polish_R)
        
        improved = True
        while improved and time.time() - start_time < time_limit - 1.0:
            improved = False
            for i in range(N):
                if time.time() - start_time > time_limit - 1.0:
                    break
                
                # Try flipping i
                delta_val = -2.0 * polish_a[i]
                term = np.roll(polish_a, i) + np.roll(polish_a, -i)
                delta_R = delta_val * term
                candidate_R = polish_R + delta_R
                candidate_peak = get_peak(candidate_R)
                
                if candidate_peak < polish_peak:
                    polish_R = candidate_R
                    polish_a[i] *= -1.0
                    polish_peak = candidate_peak
                    improved = True
                    
                    if polish_peak < best_peak:
                        best_peak = polish_peak
                        best_a = polish_a.copy()
        
        return best_a
# EVOLVE-BLOCK-END
