import time
import numpy as np

def construct():
    """
    Constructs a subset A of [0, 128] such that every integer in [128, 256]
    is a sum of two elements of A. Minimizes the largest representation multiplicity.
    
    Strategy:
    1. Start with a dense, valid solution (e.g., A = [64, ..., 128]).
    2. Use Simulated Annealing to explore the space of subsets.
    3. Objective: Minimize (Peak Multiplicity * 100 + Number of Uncovered Sums).
       - Coverage is hard constraint (penalty 100 per missing sum).
       - Peak is soft constraint to be minimized.
    4. Moves: Add a random element, Remove a random element, or Swap.
    5. Fast evaluation using NumPy convolution.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    target_min = 128
    target_max = 256
    N = 129  # Range [0, 128]
    
    # Initial state: Dense set that is guaranteed to cover [128, 256]
    # If A = [64, ..., 128], min sum = 128, max sum = 256.
    # This is a valid starting point.
    A_mask = np.zeros(N, dtype=np.int8)
    A_mask[64:129] = 1
    
    def get_metrics(mask):
        """
        Returns (peak_count, uncovered_count).
        peak_count: max representation count in [128, 256].
        uncovered_count: number of sums in [128, 256] with count 0.
        """
        # Convolution of mask with itself
        # mask is length 129, conv is length 257
        conv = np.convolve(mask, mask, mode='full')
        
        # Extract target range [128, 256]
        target_conv = conv[128:257]
        
        # Calculate unordered pair counts
        # N_ord[s] = 2 * N_unord_distinct[s] + N_diag[s]
        # N_unord_total[s] = (N_ord[s] + N_diag[s]) / 2
        
        sums = np.arange(128, 257)
        even_mask = (sums % 2 == 0)
        mids = sums[even_mask] // 2
        
        diag_terms = np.zeros(129, dtype=np.int32)
        # mids are in [64, 128] roughly, but let's be safe.
        # sums are 128..256, mids are 64..128.
        # mask is 0..128. So mids are always valid indices for mask.
        diag_terms[even_mask] = mask[mids]
        
        # Note: conv values are int, diag is int. 
        # (conv + diag) is always even? 
        # conv[s] = sum_{i+j=s} mask[i]mask[j]. 
        # If s is even, i=j=s/2 term is 1 if mask[s/2]=1. 
        # The other terms come in pairs (i,j) and (j,i). 
        # So conv[s] = 2 * (sum_{i<j, i+j=s} mask[i]mask[j]) + mask[s/2]^2.
        # Since mask is binary, mask[s/2]^2 = mask[s/2].
        # So conv[s] - diag[s] is even.
        # (conv + diag) // 2 is correct integer division.
        
        counts = (target_conv + diag_terms) // 2
        
        peak = np.max(counts)
        uncovered = np.sum(counts == 0)
        
        return peak, uncovered

    # Initial metrics
    current_peak, current_uncovered = get_metrics(A_mask)
    current_obj = current_peak * 100 + current_uncovered
    
    best_A_mask = A_mask.copy()
    best_obj = current_obj
    best_peak = current_peak
    best_uncovered = current_uncovered
    
    # Simulated Annealing Parameters
    T = 10.0  # Initial temperature
    cooling_rate = 0.995
    
    # Precompute random indices for speed? Or just use rng.
    rng = np.random.default_rng(42)
    
    while time.time() < deadline and T > 0.01:
        # Generate a move
        move_type = rng.random()
        
        new_mask = A_mask.copy()
        
        if move_type < 0.3:
            # Add an element
            # Pick a random index not in A
            candidates = np.where(new_mask == 0)[0]
            if len(candidates) > 0:
                idx = rng.choice(candidates)
                new_mask[idx] = 1
            else:
                continue # No move possible
        elif move_type < 0.6:
            # Remove an element
            # Pick a random index in A
            candidates = np.where(new_mask == 1)[0]
            if len(candidates) > 1: # Don't remove last one
                idx = rng.choice(candidates)
                new_mask[idx] = 0
            else:
                continue
        else:
            # Swap: Remove one, Add one
            candidates_in = np.where(new_mask == 1)[0]
            candidates_out = np.where(new_mask == 0)[0]
            if len(candidates_in) > 1 and len(candidates_out) > 0:
                idx_in = rng.choice(candidates_in)
                idx_out = rng.choice(candidates_out)
                new_mask[idx_in] = 0
                new_mask[idx_out] = 1
            else:
                continue
        
        # Evaluate new state
        new_peak, new_uncovered = get_metrics(new_mask)
        new_obj = new_peak * 100 + new_uncovered
        
        # Acceptance criterion
        delta_obj = new_obj - current_obj
        
        if delta_obj < 0:
            # Improvement
            A_mask = new_mask
            current_peak, current_uncovered = new_peak, new_uncovered
            current_obj = new_obj
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_peak = current_peak
                best_uncovered = current_uncovered
                best_A_mask = A_mask.copy()
        else:
            # Worse, accept with probability exp(-delta/T)
            if rng.random() < np.exp(-delta_obj / T):
                A_mask = new_mask
                current_peak, current_uncovered = new_peak, new_uncovered
                current_obj = new_obj
        
        T *= cooling_rate
        
        # Optional: If we found a perfect solution (peak=1, uncovered=0), stop?
        # Peak=1 is the theoretical minimum.
        if best_uncovered == 0 and best_peak == 1:
            break

    # Return the best set found
    # Convert mask to list of indices
    best_A_list = np.where(best_A_mask == 1)[0].tolist()
    
    # Ensure non-empty
    if not best_A_list:
        return [128]
        
    return best_A_list

# EVOLVE-BLOCK-END
