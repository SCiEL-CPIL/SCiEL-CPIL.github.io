import time
import random
import numpy as np

def construct():
    """
    Constructs a subset A of [0, 128] such that every integer in [128, 256]
    is a sum of two elements of A. Minimizes the largest representation multiplicity.
    
    Strategy:
    1. Greedy Construction: Start with a minimal valid seed and add elements
       that minimize the increase in peak multiplicity.
    2. Greedy Pruning: Remove elements that do not increase the peak multiplicity.
    3. Local Search Refinement: Use a fast local search (hill climbing with restarts)
       on the pruned set to further reduce the peak.
    """
    deadline = time.time() + 85
    rng = random.Random(42)
    
    # Helper to evaluate a set A
    # Returns (peak, is_valid, counts_array)
    def evaluate(A_set):
        if not A_set:
            return float('inf'), False, None
        
        # Create boolean vector
        v = np.zeros(257, dtype=np.int32)
        for a in A_set:
            if 0 <= a <= 128:
                v[a] = 1
        
        # Convolution
        # v has length 257 (indices 0..256).
        # We need sums 128..256.
        # conv = v * v
        # conv[k] is count of ordered pairs (a,b) in A such that a+b=k.
        # We only care about k in [128, 256].
        
        # Efficient convolution for specific range
        # Since v is sparse-ish, we can just compute the full conv if it's fast enough
        # 257*257 is ~66k ops, very fast.
        conv = np.convolve(v, v, mode='full')
        
        # Extract target range
        # Indices 128 to 256 inclusive
        target_counts = conv[128:257]
        
        # Adjust for unordered pairs with replacement
        # Ordered count N(k).
        # Unordered count U(k) = (N(k) + I(k/2 in A)) / 2 if k even?
        # Let's re-verify.
        # Pairs (a,b) with a<=b.
        # If a!=b, (a,b) and (b,a) are distinct in conv. So count is 2*U_offdiag.
        # If a==b, (a,a) is 1 in conv. So count is U_diag.
        # Total N(k) = 2 * U_offdiag + U_diag.
        # We want U(k) = U_offdiag + U_diag.
        # So 2*U(k) = N(k) + U_diag.
        # U_diag is 1 if k/2 is in A and k is even, else 0.
        
        target_sums = np.arange(128, 257)
        even_mask = (target_sums % 2 == 0)
        
        adjusted_counts = target_counts.astype(np.float64)
        
        if np.any(even_mask):
            mids = target_sums[even_mask] // 2
            # Check if mid is in A
            # mids are in [64, 128].
            # v has length 257, so v[mids] works.
            in_A = v[mids]
            adjusted_counts[even_mask] += in_A
            
        final_counts = adjusted_counts // 2
        
        # Check validity
        if np.any(final_counts == 0):
            return float('inf'), False, final_counts
            
        peak = int(np.max(final_counts))
        return peak, True, final_counts

    # --- Phase 1: Greedy Construction ---
    
    # Start with a set that is likely to be valid quickly
    # 0 and 128 are good anchors.
    # If we have 0 and 128, we can form 128 (0+128, 64+64 if 64 in A) and 256 (128+128).
    # Let's start with a random small set and expand.
    
    A = set()
    # Add anchors
    A.add(0)
    A.add(128)
    
    # Evaluate initial state
    peak, is_valid, counts = evaluate(A)
    
    # We want to expand A until it is valid and has a low peak.
    # Actually, we should expand until it's valid.
    # Then we can try to optimize.
    
    # Greedy expansion:
    # While not valid:
    #   Find x in [0,128]\A that minimizes the new peak.
    
    all_elements = list(range(129))
    
    while not is_valid and time.time() < deadline - 20:
        best_x = None
        best_new_peak = float('inf')
        best_new_counts = None
        
        # To speed up, we can sample candidates if the set is large, 
        # but initially it's small.
        candidates = [x for x in all_elements if x not in A]
        
        # Shuffle to break ties randomly and avoid bias
        rng.shuffle(candidates)
        
        for x in candidates:
            A.add(x)
            new_peak, new_valid, new_counts = evaluate(A)
            A.remove(x)
            
            if new_valid:
                # If it becomes valid, we prioritize it, but we still want low peak
                # If current is invalid, any valid is better.
                # But we might find a valid one with lower peak later?
                # No, we want to get valid first.
                # Let's pick the valid one with the lowest peak.
                if new_peak < best_new_peak:
                    best_new_peak = new_peak
                    best_x = x
                    best_new_counts = new_counts
            else:
                # If still invalid, we want to minimize the number of 0s?
                # Or minimize the peak of the non-zero parts?
                # Let's minimize the max count among non-zero, and maximize coverage.
                # Simple heuristic: minimize peak, tie-break by number of zeros (lower is better).
                zeros = np.sum(new_counts == 0)
                # We want fewer zeros.
                # Let's use a combined score: peak * 10 + zeros
                score = new_peak * 10 + zeros
                if score < best_new_peak:
                    best_new_peak = score
                    best_x = x
                    best_new_counts = new_counts
        
        if best_x is not None:
            A.add(best_x)
            peak, is_valid, counts = evaluate(A)
        else:
            # No candidate improved it? Should not happen if we can add any element.
            # Break to avoid infinite loop
            break
    
    if not is_valid:
        # Fallback: dense set
        A = set(range(64, 129))
        peak, is_valid, counts = evaluate(A)
        if not is_valid:
            return list(A)

    # --- Phase 2: Greedy Pruning ---
    # Try to remove elements to reduce peak or maintain peak with smaller set.
    # If removing x doesn't increase peak, remove it.
    # This often helps because removing "bad" elements (those contributing to high peaks)
    # can lower the peak if they are not critical for coverage.
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        current_peak, _, _ = evaluate(A)
        
        # List elements to try removing
        # Sort by some heuristic? Elements that contribute to high peaks?
        # Just shuffle for now.
        elements_to_try = list(A)
        rng.shuffle(elements_to_try)
        
        for x in elements_to_try:
            if len(A) <= 2:
                continue
                
            A.remove(x)
            new_peak, new_valid, _ = evaluate(A)
            
            if new_valid and new_peak <= current_peak:
                # Accept removal
                current_peak = new_peak
                improved = True
                # Continue checking other elements
            else:
                # Revert
                A.add(x)
                
    # --- Phase 3: Local Search Refinement ---
    # Now that we have a valid, pruned set, we can do a more targeted search.
    # Hill Climbing with Swaps.
    
    best_A = set(A)
    best_peak, _, _ = evaluate(best_A)
    
    T = 1.0
    cooling_rate = 0.95
    min_T = 0.01
    
    while time.time() < deadline - 2:
        if T < min_T:
            T = 1.0 # Restart with high T
            # Maybe randomize A slightly?
            # Let's just continue with current T reset.
            
        current_peak, _, _ = evaluate(A)
        
        # Generate neighbor: Swap one element
        if len(A) < 129:
            in_elem = rng.choice(list(A))
            out_candidates = [x for x in all_elements if x not in A]
            if out_candidates:
                out_elem = rng.choice(out_candidates)
                
                A.remove(in_elem)
                A.add(out_elem)
                
                new_peak, new_valid, _ = evaluate(A)
                
                if not new_valid:
                    # Revert
                    A.remove(out_elem)
                    A.add(in_elem)
                else:
                    if new_peak < current_peak:
                        current_peak = new_peak
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_A = set(A)
                    else:
                        # Accept worse with probability exp(-delta/T)
                        delta = new_peak - current_peak
                        if delta > 0:
                            prob = np.exp(-delta / T)
                            if rng.random() < prob:
                                current_peak = new_peak
                            else:
                                # Revert
                                A.remove(out_elem)
                                A.add(in_elem)
                        else:
                            # Equal or better (shouldn't be equal if we check <)
                            # If delta == 0, accept
                            pass
        
        T *= cooling_rate

    return list(best_A)
