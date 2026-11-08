import numpy as np
import time

N = 400

def construct():
    """
    Returns a sequence of signed step heights (length N) that minimizes 
    the upper bound on C3 (interpreted as peak autocorrelation sidelobe).
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def compute_score(h):
        """
        Computes the peak autocorrelation sidelobe.
        """
        H = np.fft.fft(h)
        R = np.fft.ifft(np.abs(H) ** 2).real
        # Exclude the main lobe at lag 0
        # R is symmetric for real input, but we take max of abs of all non-zero lags
        # R[0] is the energy, which is N.
        # We want max(|R[1]|, ..., |R[N-1]|)
        return np.max(np.abs(R[1:]))

    def random_sequence():
        return np.random.choice([-1, 1], N)

    best_h = None
    best_score = float('inf')
    
    # Simulated Annealing Parameters
    T_max = 10.0  # Higher initial temperature to allow large jumps
    T_min = 0.001
    cooling_rate = 0.99995  # Slower cooling
    max_iterations_per_run = 50000
    
    # We will perform multiple SA restarts
    # Each restart starts with a fresh random sequence
    # This helps explore different basins of attraction
    
    for restart in range(50): # Max 50 restarts
        if time.time() > deadline - 5:
            break
            
        h = random_sequence()
        current_score = compute_score(h)
        T = T_max
        
        # Track local best for this run
        local_best_score = current_score
        local_best_h = h.copy()
        
        # Stagnation counter
        no_improve_count = 0
        
        for i in range(max_iterations_per_run):
            if time.time() > deadline - 5:
                break
                
            # Generate a move
            # Mix of single flips and block flips
            # Block flips are more effective at escaping local minima in spectral problems
            if np.random.random() < 0.7:
                # Single flip
                idx = np.random.randint(N)
                h[idx] *= -1
                is_block = False
                block_start = idx
                block_len = 1
            else:
                # Block flip
                # Length 2 to 30
                L = np.random.randint(2, 31)
                start_idx = np.random.randint(0, N - L + 1)
                h[start_idx:start_idx+L] *= -1
                is_block = True
                block_start = start_idx
                block_len = L
                
            new_score = compute_score(h)
            delta = new_score - current_score
            
            # Metropolis Acceptance
            if delta < 0:
                # Always accept improvement
                current_score = new_score
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best_h = h.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Accept worse solution with probability exp(-delta/T)
                if T > 0:
                    prob = np.exp(-delta / T)
                    if np.random.random() < prob:
                        current_score = new_score
                        no_improve_count += 1
                    else:
                        # Revert
                        if is_block:
                            h[block_start:block_start+block_len] *= -1
                        else:
                            h[block_start] *= -1
                        no_improve_count += 1
                else:
                    # Revert
                    if is_block:
                        h[block_start:block_start+block_len] *= -1
                    else:
                        h[block_start] *= -1
                    no_improve_count += 1
            
            # Cooling
            T *= cooling_rate
            
            # Reheating if stuck
            if no_improve_count > 2000:
                T = T_max * 0.5
                no_improve_count = 0
            
            if T < T_min:
                break
        
        # Update global best
        if local_best_score < best_score:
            best_score = local_best_score
            best_h = local_best_h.copy()
            
        # Early termination if score is very low
        if best_score < 1.0:
            break

    if best_h is None:
        best_h = random_sequence()
        
    return best_h
