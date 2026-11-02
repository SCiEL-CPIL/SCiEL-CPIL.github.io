"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Approach: Hill Climbing with Random Restarts on Peak Sidelobe Level (PSL)
    # This differs from the previous Simulated Annealing on Energy.
    # We explicitly select the metric as the maximum absolute value of the
    # non-zero autocorrelation lags (PSL), which is a common "upper bound"
    # metric in autocorrelation inequality contexts.
    
    np.random.seed(42)
    
    def get_psl(seq):
        ac = np.correlate(seq, seq, mode='full')
        # Peak is at index N-1
        # Sidelobes are indices 0 to N-2 and N to 2N-2
        # Due to symmetry, max(abs(ac[0:N-1])) is sufficient
        return np.max(np.abs(ac[:N-1]))

    best_seq = np.random.choice([-1, 1], size=N)
    best_psl = get_psl(best_seq)
    
    start_time = time.time()
    max_time = 110.0 # Leave buffer for return
    
    # Parameters for Hill Climbing
    stall_count = 0
    max_stall = 2000 # Restart after this many non-improving flips
    
    while (time.time() - start_time) < max_time:
        # If stalled, restart with a new random sequence
        if stall_count > max_stall:
            # Try a few restarts or a perturbed version
            # Perturb current best slightly
            seq = best_seq.copy()
            # Flip 5% of bits
            num_flips = N // 20
            indices = np.random.choice(N, num_flips, replace=False)
            seq[indices] *= -1
            current_psl = get_psl(seq)
            if current_psl < best_psl:
                best_seq = seq
                best_psl = current_psl
            else:
                # If perturbation didn't help, try a fresh random start
                # This helps escape deep local minima
                seq = np.random.choice([-1, 1], size=N)
                current_psl = get_psl(seq)
                if current_psl < best_psl:
                    best_seq = seq
                    best_psl = current_psl
            stall_count = 0
            continue
            
        # Pick a random index to flip
        i = np.random.randint(N)
        old_val = seq[i]
        seq[i] = -old_val
        
        new_psl = get_psl(seq)
        
        if new_psl < current_psl:
            current_psl = new_psl
            if current_psl < best_psl:
                best_psl = current_psl
                best_seq = seq.copy()
            stall_count = 0
        else:
            # Revert
            seq[i] = old_val
            stall_count += 1
            
    return best_seq
# EVOLVE-BLOCK-END
