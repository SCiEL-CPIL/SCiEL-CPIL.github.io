import numpy as np
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 128
    N = n + 1  # 129 elements in [0, 128]
    
    # We use a mask representation for speed
    # A_mask[i] = 1 if i in A
    
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

    def objective(peak, uncovered):
        # We want to minimize peak, subject to uncovered == 0.
        # If uncovered > 0, penalty is high.
        # Let's use: score = peak + 1000 * uncovered
        # This ensures we prioritize coverage.
        return peak + 1000 * uncovered

    # Strategy: 
    # 1. Start with a dense valid solution.
    #    A = [0, 1, ..., 128] is valid but peak is high.
    #    A = [64, ..., 128] is valid (min sum 128, max 256).
    #    Let's check peak for A = [64, ..., 128].
    #    k = 65 elements.
    #    Pairs = 65*66/2 = 2145.
    #    Target range size = 129.
    #    Avg count = 2145/129 ~ 16.6.
    #    This is likely high.
    
    #    Better starting point: Random set of size ~20-30?
    #    But it might be invalid.
    #    Let's try to construct a valid set with low peak.
    
    #    Idea: Use a greedy construction to find a valid set, then SA to optimize.
    #    Or start with a random set and repair?
    
    #    Let's try multiple restarts with random initializations of varying sizes.
    #    And use SA with fast evaluation (convolution is O(N^2) ~ 129^2 ~ 16k ops, very fast).
    
    rng = np.random.default_rng(42)
    
    best_mask = None
    best_score = float('inf')
    best_peak = float('inf')
    best_uncovered = float('inf')
    
    num_restarts = 100 # We have time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Random initialization
        # Size k between 15 and 35
        k = rng.integers(15, 36)
        
        # Sample k elements from [0, 128]
        indices = rng.choice(N, size=k, replace=False)
        mask = np.zeros(N, dtype=np.int8)
        mask[indices] = 1
        
        # Check if valid
        peak, uncovered = get_metrics(mask)
        score = objective(peak, uncovered)
        
        if score < best_score:
            best_score = score
            best_peak = peak
            best_uncovered = uncovered
            best_mask = mask.copy()
            
        # If we found a valid set with low peak, we can start SA from here
        # Otherwise, we might want to force validity first?
        # Let's just run SA from any state, the objective function handles it.
        
        # Simulated Annealing
        # We will run SA for a fixed number of steps or time budget per restart.
        # Since evaluation is fast, we can do many steps.
        
        current_mask = mask.copy()
        current_peak, current_uncovered = get_metrics(current_mask)
        current_score = objective(current_peak, current_uncovered)
        
        T = 5.0
        cooling = 0.999
        steps = 2000 # Fixed steps, fast eval
        
        for step in range(steps):
            if time.time() > deadline - 5.0:
                break
                
            # Move: Add, Remove, or Swap
            move_type = rng.random()
            
            if move_type < 0.3:
                # Add
                zeros = np.where(current_mask == 0)[0]
                if len(zeros) == 0: continue
                idx = rng.choice(zeros)
                new_mask = current_mask.copy()
                new_mask[idx] = 1
            elif move_type < 0.6:
                # Remove
                ones = np.where(current_mask == 1)[0]
                if len(ones) <= 1: continue
                idx = rng.choice(ones)
                new_mask = current_mask.copy()
                new_mask[idx] = 0
            else:
                # Swap
                ones = np.where(current_mask == 1)[0]
                zeros = np.where(current_mask == 0)[0]
                if len(ones) == 0 or len(zeros) == 0: continue
                idx_in = rng.choice(ones)
                idx_out = rng.choice(zeros)
                new_mask = current_mask.copy()
                new_mask[idx_in] = 0
                new_mask[idx_out] = 1
            
            new_peak, new_uncovered = get_metrics(new_mask)
            new_score = objective(new_peak, new_uncovered)
            
            delta = new_score - current_score
            
            if delta < 0 or (T > 0.01 and rng.random() < np.exp(-delta / T)):
                current_mask = new_mask
                current_peak, current_uncovered = new_peak, new_uncovered
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_peak = current_peak
                    best_uncovered = current_uncovered
                    best_mask = current_mask.copy()
                    
                    # If we hit peak 2 and uncovered 0, that's very good.
                    # Theoretical min peak?
                    # If k=20, pairs=210. 210/129 = 1.62. So peak 2 is possible.
                    # If we find peak 2, we can stop early?
                    if best_uncovered == 0 and best_peak == 2:
                        return np.where(best_mask == 1)[0].tolist()
            
            T *= cooling
            
    # If best_mask is None or invalid, try a fallback
    if best_mask is None:
        # Return a simple valid set
        # A = [0, 1, ..., 128]
        return list(range(N))
        
    # Check if best is valid
    if best_uncovered > 0:
        # We didn't find a valid set in the search.
        # This is unlikely if we started with valid sets or SA explored well.
        # But to be safe, let's try to construct a valid set quickly.
        # Greedy: Start with empty, add elements that cover most uncovered sums.
        # Or just return the dense set.
        # Let's try a quick greedy repair on best_mask if it's invalid.
        
        mask = best_mask.copy()
        peak, uncovered = get_metrics(mask)
        
        # Greedy add
        for _ in range(50): # Max 50 additions
            if time.time() > deadline - 1.0:
                break
            if uncovered == 0:
                break
                
            # Find which sums are uncovered
            # We need to pick an element x that covers the most uncovered sums.
            # This is expensive to compute exactly.
            # Approximation: Pick random x not in A, check improvement.
            
            improved = False
            for _ in range(20): # Try 20 random candidates
                zeros = np.where(mask == 0)[0]
                if len(zeros) == 0: break
                x = rng.choice(zeros)
                new_mask = mask.copy()
                new_mask[x] = 1
                new_peak, new_uncovered = get_metrics(new_mask)
                
                if new_uncovered < uncovered:
                    mask = new_mask
                    peak, uncovered = new_peak, new_uncovered
                    improved = True
                    break
            
            if not improved:
                break
        
        peak, uncovered = get_metrics(mask)
        if uncovered == 0:
            best_mask = mask
            best_peak = peak
            best_uncovered = 0
            best_score = objective(peak, uncovered)
        else:
            # Still invalid, return dense set
            return list(range(N))
            
    return np.where(best_mask == 1)[0].tolist()
