import time
import numpy as np

def construct():
    """
    Constructs a subset A of [0, 128] such that every integer in [128, 256]
    is a sum of two elements of A. Minimizes the largest representation multiplicity.
    
    Strategy:
    1. Use NumPy convolution for fast evaluation of representation counts.
    2. Multi-restart Simulated Annealing to escape local minima.
    3. Objective: Minimize (Peak Multiplicity * 1000 + Uncovered Count).
       - High penalty for uncovered sums ensures validity.
       - Peak is minimized subject to validity.
    4. Moves: Add, Remove, Swap.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    target_min = 128
    target_max = 256
    N = 129  # Range [0, 128]
    
    # Precompute targets for evaluation
    # We only care about sums in [128, 256]
    
    def get_metrics(mask):
        """
        Returns (peak_count, uncovered_count).
        Uses convolution to count ordered pairs, then adjusts for unordered pairs.
        """
        # Convolution of mask with itself gives ordered pair counts
        # conv[s] = sum_{i+j=s} mask[i]*mask[j]
        conv = np.convolve(mask, mask, mode='full')
        
        # Extract target range [128, 256]
        # conv indices correspond to sum s
        target_conv = conv[128:257]
        
        # Convert ordered counts to unordered counts
        # For sum s:
        # If s is even, let m = s/2.
        # Ordered count C_ord = 2 * N_distinct + N_diag
        # Unordered count N_unord = N_distinct + N_diag
        # N_distinct = (C_ord - N_diag) / 2
        # N_unord = (C_ord - N_diag)/2 + N_diag = (C_ord + N_diag) / 2
        # Where N_diag = mask[s/2] if s is even, else 0.
        
        sums = np.arange(128, 257)
        even_mask = (sums % 2 == 0)
        
        # Initialize diag terms to 0
        diag_terms = np.zeros(129, dtype=np.int32)
        
        # Calculate diag terms for even sums
        # s = 128, 130, ..., 256
        # m = 64, 65, ..., 128
        even_sums = sums[even_mask]
        mids = even_sums // 2
        
        # mask is length 129 (indices 0..128)
        # mids are in [64, 128], which is valid
        diag_terms[even_mask] = mask[mids]
        
        # Calculate unordered counts
        # target_conv and diag_terms are integers
        # (target_conv + diag_terms) is always even?
        # C_ord = 2*N_dist + N_diag. 
        # If N_diag=1 (even s, mask[m]=1), C_ord is odd. C_ord + 1 is even.
        # If N_diag=0, C_ord is even. C_ord + 0 is even.
        # So integer division is safe.
        
        counts = (target_conv + diag_terms) // 2
        
        peak = np.max(counts)
        uncovered = np.sum(counts == 0)
        
        return peak, uncovered

    def run_sa(init_mask, seed, time_budget):
        """
        Runs a single Simulated Annealing chain.
        Returns best_mask, best_peak, best_uncovered.
        """
        rng = np.random.default_rng(seed)
        
        A_mask = init_mask.copy()
        current_peak, current_uncovered = get_metrics(A_mask)
        current_obj = current_peak * 1000 + current_uncovered
        
        best_A_mask = A_mask.copy()
        best_obj = current_obj
        best_peak = current_peak
        best_uncovered = current_uncovered
        
        T = 5.0  # Start lower to focus on fine-tuning, as dense start is good
        cooling_rate = 0.995
        min_T = 0.01
        
        while time.time() < time_budget and T > min_T:
            # Generate a move
            move_type = rng.random()
            
            new_mask = A_mask.copy()
            
            if move_type < 0.3:
                # Add an element
                candidates = np.where(new_mask == 0)[0]
                if len(candidates) > 0:
                    idx = rng.choice(candidates)
                    new_mask[idx] = 1
                else:
                    continue
            elif move_type < 0.6:
                # Remove an element
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
            new_obj = new_peak * 1000 + new_uncovered
            
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
                if T > 0:
                    # Avoid overflow
                    if delta_obj / T < 700:
                        prob = np.exp(-delta_obj / T)
                    else:
                        prob = 0.0
                    if rng.random() < prob:
                        A_mask = new_mask
                        current_peak, current_uncovered = new_peak, new_uncovered
                        current_obj = new_obj
            
            T *= cooling_rate
            
            # Early stop if perfect
            if best_uncovered == 0 and best_peak == 1:
                break
                
        return best_A_mask, best_peak, best_uncovered, best_obj

    # Initial states
    # 1. Dense set [64..128] - Guaranteed valid
    mask_dense = np.zeros(N, dtype=np.int8)
    mask_dense[64:129] = 1
    
    # 2. Random set with high probability
    rng_init = np.random.default_rng(123)
    mask_rand = np.zeros(N, dtype=np.int8)
    # Pick ~40 random elements
    indices = rng_init.choice(N, size=40, replace=False)
    mask_rand[indices] = 1
    
    # 3. Another random set
    mask_rand2 = np.zeros(N, dtype=np.int8)
    indices = rng_init.choice(N, size=35, replace=False)
    mask_rand2[indices] = 1

    initial_masks = [mask_dense, mask_rand, mask_rand2]
    
    best_mask_global = None
    best_obj_global = float('inf')
    best_peak_global = 0
    best_uncovered_global = 0
    
    # Distribute time budget among restarts
    # We have 85s total. Let's give each restart a chunk, but stop if we find a good solution.
    # 3 restarts.
    
    for i, init_mask in enumerate(initial_masks):
        if time.time() > deadline:
            break
            
        # Allocate time: first restart gets more time if it's the dense one (safer)
        # Actually, just run until deadline minus a small buffer for the next ones?
        # Better: Run each for a fixed fraction, or until deadline.
        # Let's run the first one for 30s, second for 25s, third for 20s.
        # Or just run them sequentially with a shrinking deadline.
        
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
            
        # Give the first restart 40% of remaining time, next 30%, last 30%?
        # Simpler: Just run with a hard deadline for the whole loop, but break if we find peak=1.
        # To ensure all get a chance, we can limit each run to (deadline - time.time()) / (len(initial_masks) - i)
        
        time_limit_for_this_run = time.time() + (deadline - time.time()) / (len(initial_masks) - i)
        
        mask_res, peak_res, unc_res, obj_res = run_sa(init_mask, seed=42+i, time_budget=time_limit_for_this_run)
        
        if obj_res < best_obj_global:
            best_obj_global = obj_res
            best_mask_global = mask_res
            best_peak_global = peak_res
            best_uncovered_global = unc_res
            
        # If we found a perfect solution, we can stop early
        if best_uncovered_global == 0 and best_peak_global == 1:
            break

    # If no valid solution was found (unlikely with dense start), fallback to dense
    if best_mask_global is None or best_uncovered_global > 0:
        # Try one more time with a very aggressive random search if time permits?
        # Or just return the dense set which is valid.
        mask_dense = np.zeros(N, dtype=np.int8)
        mask_dense[64:129] = 1
        best_mask_global = mask_dense
        # Recalculate metrics for return if needed, but we just return the list.
        
    result_list = np.where(best_mask_global == 1)[0].tolist()
    
    # Ensure non-empty
    if not result_list:
        return [128]
        
    return result_list
