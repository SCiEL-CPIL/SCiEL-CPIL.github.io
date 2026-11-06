"""Low-discrepancy on arithmetic progressions via histogram-accelerated simulated annealing."""
# EVOLVE-BLOCK-START
import time
import numpy as np

N = 256

def construct():
    t0 = time.time()
    deadline = t0 + 55.0  # safety margin before 60s limit
    
    # Precompute APs: list of lists of indices (0-based)
    aps = []
    for step in range(1, N + 1):
        for start in range(1, N + 1):
            idx = []
            pos = start - 1
            while pos < N:
                idx.append(pos)
                pos += step
            if idx:
                aps.append(idx)
    
    num_aps = len(aps)
    
    # Build position-to-AP mapping
    pos_to_aps = [[] for _ in range(N)]
    for ap_idx, idx in enumerate(aps):
        for i in idx:
            pos_to_aps[i].append(ap_idx)
    
    # Histogram size: sums can range from -N to N
    MAX_SUM = N
    HIST_SIZE = 2 * MAX_SUM + 1
    OFFSET = MAX_SUM
    
    def compute_initial_state(signs):
        sums = np.zeros(num_aps, dtype=np.int32)
        for i in range(N):
            s = signs[i]
            for ap_idx in pos_to_aps[i]:
                sums[ap_idx] += s
        
        hist = np.zeros(HIST_SIZE, dtype=np.int32)
        for s in sums:
            hist[s + OFFSET] += 1
            
        # Find initial max abs sum
        max_abs = 0
        for k in range(MAX_SUM, -1, -1):
            if hist[k + OFFSET] > 0 or hist[-k + OFFSET] > 0:
                max_abs = k
                break
                
        return sums, hist, max_abs

    def update_max_abs(hist, old_max_abs, old_sign, affected_aps, sums):
        """
        Update max_abs after flipping a sign.
        We need to check if the old max_abs is still valid.
        If hist[old_max_abs] and hist[-old_max_abs] are 0, we need to find the next lower max.
        Also, the new max could be higher if we increased a sum that was close to the max?
        No, flipping a sign changes sums by +/- 2.
        If we decrease a positive sum, max might drop.
        If we increase a negative sum (make it less negative), max might drop.
        If we increase a positive sum, max might rise?
        Wait, if we flip +1 to -1, sums decrease by 2.
        If we flip -1 to +1, sums increase by 2.
        
        Let's just recalculate max_abs from the histogram if it changes?
        Or simply: 
        1. Check if current max_abs is still present.
        2. If not, decrement until found.
        3. Also, could the max have increased?
           If we flipped -1 to +1, sums increased by 2.
           The new max could be old_max + 2?
           Actually, only the affected APs changed.
           If an affected AP had sum = old_max - 2 and we increased it by 2, it becomes old_max.
           If it had sum = old_max and we increased it by 2, it becomes old_max + 2.
           
        So we should check:
        - Is old_max_abs still in hist?
        - Is old_max_abs + 2 in hist? (if we increased sums)
        - Is old_max_abs - 2 in hist? (if we decreased sums, max might drop)
        
        Simplest robust way for small N:
        Since N=256, MAX_SUM=256.
        We can just find the new max by scanning from old_max_abs + 2 downwards?
        No, that's O(N) in worst case.
        
        However, the change is small.
        Let's track `current_max_abs`.
        After update:
        1. If we flipped -1 to +1 (sums increase):
           New max could be at most old_max_abs + 2.
           Check if hist[old_max_abs + 2 + OFFSET] > 0. If so, max is old_max_abs + 2.
           Else check if hist[old_max_abs + OFFSET] > 0. If so, max is old_max_abs.
           Else check if hist[old_max_abs - 2 + OFFSET] > 0... etc.
           
        2. If we flipped +1 to -1 (sums decrease):
           New max is at most old_max_abs.
           Check if hist[old_max_abs + OFFSET] > 0.
           If not, check old_max_abs - 2, etc.
           
        We can implement a helper that finds the new max starting from a bound.
        """
        # Determine direction of change
        # If old_sign was -1, new sign is +1, sums increased by 2.
        # If old_sign was +1, new sign is -1, sums decreased by 2.
        
        # We can just re-scan from a reasonable bound.
        # The max can only change by +/- 2 or stay same?
        # Actually, if the unique max AP was decreased, max drops.
        # If an AP was increased to become the new max, max rises.
        
        # Let's just find the max by scanning from MAX_SUM down?
        # 256 iterations is very fast.
        # num_aps is ~32k.
        # Scanning 512 bins is much faster than scanning 32k sums.
        
        new_max = 0
        # We can optimize by starting from old_max_abs + 2 (if increase) or old_max_abs (if decrease)
        # But simply scanning from MAX_SUM down is only 256 steps.
        # Let's do that to be safe and simple.
        
        # Optimization: 
        # If old_sign == -1 (increase):
        #   Check MAX_SUM, MAX_SUM-1... down to old_max_abs-2?
        #   Actually, just check from MAX_SUM down.
        
        # Let's use the histogram to find max.
        # Since we update hist, we can just find the index with max |k| where hist[k+OFFSET] > 0.
        
        # To be extremely fast, we can maintain a set of "active" max values?
        # Or just loop. 256 iterations is negligible compared to the flip update cost.
        
        # Let's just loop from MAX_SUM down to 0.
        for k in range(MAX_SUM, -1, -1):
            if hist[k + OFFSET] > 0 or hist[-k + OFFSET] > 0:
                new_max = k
                break
                
        return new_max

    # Simulated Annealing
    best_signs = None
    best_obj = 10**9
    
    rng = np.random.default_rng(171)
    
    # Try multiple restarts
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() >= deadline:
            break
        
        # Random initialization
        signs = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
        sums, hist, cur_obj = compute_initial_state(signs)
        
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_signs = signs.copy()
        
        # SA parameters
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        steps_per_restart = 50000  # More steps allowed now due to O(1) obj eval
        
        for step in range(steps_per_restart):
            if time.time() >= deadline:
                break
            
            # Pick random position to flip
            i = int(rng.integers(N))
            old_sign = signs[i]
            affected = pos_to_aps[i]
            
            delta = -2 * old_sign
            
            # Update sums and histogram
            for ap_idx in affected:
                old_sum = sums[ap_idx]
                new_sum = old_sum + delta
                sums[ap_idx] = new_sum
                
                hist[old_sum + OFFSET] -= 1
                hist[new_sum + OFFSET] += 1
            
            # Update sign
            signs[i] = -old_sign
            
            # Compute new objective
            new_obj = update_max_abs(hist, cur_obj, old_sign, affected, sums)
            
            # Acceptance criterion
            d = new_obj - cur_obj
            if d <= 0 or (T > T_min and rng.random() < np.exp(-d / T)):
                cur_obj = new_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_signs = signs.copy()
            else:
                # Revert
                signs[i] = old_sign
                for ap_idx in affected:
                    new_sum = sums[ap_idx]
                    old_sum = new_sum - delta
                    sums[ap_idx] = old_sum
                    
                    hist[new_sum + OFFSET] -= 1
                    hist[old_sum + OFFSET] += 1
                # cur_obj remains the same as before the flip attempt?
                # Yes, because we reverted the state completely.
                # So cur_obj is still valid.
            
            T *= cooling
            if T < T_min:
                break
    
    if best_signs is None:
        best_signs = np.array([1 if rng.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
    
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
