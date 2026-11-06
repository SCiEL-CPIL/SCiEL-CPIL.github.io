import numpy as np
import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0

    # 1. Generate APs
    # ap_terms: list of lists of indices
    # We only need the sums, so we can store the APs as numpy arrays
    ap_indices = []
    for d in range(1, N + 1):
        for s in range(N):
            # Generate terms
            terms = []
            curr = s
            while curr < N:
                terms.append(curr)
                curr += d
            # Add all prefixes
            # Prefix length 1: [s]
            # Prefix length 2: [s, s+d]
            # ...
            # We can just add the full list and then take prefixes?
            # Or just iterate L.
            for L in range(1, len(terms) + 1):
                ap_indices.append(np.array(terms[:L], dtype=np.int32))
    
    K = len(ap_indices)
    
    # 2. Precompute incidence
    # idx_to_aps[i] = list of AP indices containing i
    idx_to_aps = [[] for _ in range(N)]
    for k, terms in enumerate(ap_indices):
        for i in terms:
            idx_to_aps[i].append(k)
            
    # Convert to numpy arrays for faster access?
    # List of lists is fine, but numpy arrays for sums update are faster.
    # Let's keep as lists of ints for now, or convert to np.array for vectorized sum update?
    # sums[idxs] += delta is fast if idxs is np.array.
    idx_to_aps_np = [np.array(lst, dtype=np.int32) for lst in idx_to_aps]

    # 3. Initialize
    signs = np.array([1 if random.random() < 0.5 else -1 for _ in range(N)], dtype=np.float64)
    sums = np.zeros(K, dtype=np.float64)
    
    # Calculate initial sums
    # M is sparse, but K*N is ~ 180k * 256 ~ 46M floats. 368 MB. Too big for dense?
    # Better to compute sums by iterating APs.
    for k, terms in enumerate(ap_indices):
        sums[k] = np.sum(signs[terms])

    current_max = np.max(np.abs(sums))
    current_energy = np.sum(sums**2)
    
    best_signs = signs.copy()
    best_max = current_max
    
    # Helper: Flip sign i, update sums, return new max, new energy
    def flip(i):
        nonlocal signs, sums, current_max, current_energy
        delta = -2.0 * signs[i]
        idxs = idx_to_aps_np[i]
        old_sums_affected = sums[idxs].copy()
        
        sums[idxs] += delta
        signs[i] = -signs[i]
        
        # Update energy
        # Energy change = sum(new^2 - old^2) for affected
        # new = old + delta
        # new^2 - old^2 = 2*old*delta + delta^2
        delta_energy = np.sum(2.0 * old_sums_affected * delta + delta**2)
        current_energy += delta_energy
        
        # Update max
        # We need global max. 
        # If we only track affected, we might miss if the max was elsewhere and didn't change.
        # But if we are doing SA, we need the full max for acceptance?
        # Actually, for minimizing Max, we only care if the max DECREASES.
        # For minimizing Energy (SA), we care about Energy.
        
        # Let's calculate new global max.
        # To be fast, we can maintain a histogram or just compute max if K is small enough?
        # K ~ 180k. np.max is fast (~0.1ms).
        new_max = np.max(np.abs(sums))
        current_max = new_max
        
        return delta_energy, new_max

    def unflip(i, delta_energy, old_max):
        nonlocal signs, sums, current_max, current_energy
        delta = -2.0 * signs[i] # This is the delta to REVERT?
        # If we flipped s_i from v to -v, delta was -2v.
        # To revert, we add 2v? No, signs[i] is now -v.
        # We want to go back to v.
        # delta_revert = 2 * signs[i] (which is 2*(-v) = -2v)?
        # Let's just re-apply the inverse.
        # Original flip: signs[i] *= -1. sums[idxs] += -2*old_sign.
        # Revert: signs[i] *= -1. sums[idxs] -= -2*old_sign = +2*old_sign.
        # But we don't have old_sign stored easily.
        # Easier: store the delta used.
        
        # Let's refactor flip to return the delta used.
        pass

    # Let's implement a cleaner Local Search / SA loop.
    
    def evaluate():
        return np.max(np.abs(sums))

    # Phase 1: Simulated Annealing on Energy (Sum of Squares)
    # This helps find a "balanced" configuration.
    T = 10.0
    cooling = 0.99
    min_T = 0.1
    
    while time.time() < deadline - 20 and T > min_T:
        if random.random() < 0.1:
            # Occasional max-based move?
            pass
            
        i = random.randint(0, N - 1)
        
        # Perform flip
        delta = -2.0 * signs[i]
        idxs = idx_to_aps_np[i]
        old_sums_affected = sums[idxs].copy()
        
        sums[idxs] += delta
        signs[i] = -signs[i]
        
        delta_energy = np.sum(2.0 * old_sums_affected * delta + delta**2)
        current_energy += delta_energy
        
        new_max = np.max(np.abs(sums))
        
        # Acceptance criteria
        # We want to minimize Energy primarily, but not increase Max too much?
        # Or just minimize Energy.
        if delta_energy < 0:
            # Always accept if energy decreases
            pass
        else:
            if random.random() < np.exp(-delta_energy / T):
                # Accept
                pass
            else:
                # Reject: Revert
                sums[idxs] -= delta
                signs[i] = -signs[i]
                current_energy -= delta_energy
                continue # Skip max update logic if rejected? No, max didn't change.
                
        current_max = new_max
        
        if current_max < best_max:
            best_max = current_max
            best_signs = signs.copy()
            
        T *= cooling

    # Phase 2: Local Search on Max
    # Try to reduce the maximum sum.
    signs = best_signs.copy()
    sums = np.zeros(K, dtype=np.float64)
    for k, terms in enumerate(ap_indices):
        sums[k] = np.sum(signs[terms])
    current_max = np.max(np.abs(sums))
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Find APs with max sum
        # max_sum = current_max
        # hot_aps = np.where(np.abs(sums) >= current_max - 1e-9)[0]
        
        # Try flipping each position
        for i in range(N):
            if time.time() > deadline - 5:
                break
                
            delta = -2.0 * signs[i]
            idxs = idx_to_aps_np[i]
            
            # Simulate flip
            old_sums_affected = sums[idxs].copy()
            new_sums_affected = old_sums_affected + delta
            
            # Check if max decreases
            # The new global max is max( max(abs(new_sums_affected)), max(abs(sums[~idxs])) )
            # We don't have sums[~idxs] easily.
            # But we know old global max was current_max.
            # If all affected sums decrease in magnitude, and no other sum was close to max, it works.
            # Simpler: Just do the flip, check max, revert if not better.
            
            sums[idxs] += delta
            signs[i] = -signs[i]
            
            new_max = np.max(np.abs(sums))
            
            if new_max < current_max:
                current_max = new_max
                improved = True
                if current_max < best_max:
                    best_max = current_max
                    best_signs = signs.copy()
            else:
                # Revert
                sums[idxs] -= delta
                signs[i] = -signs[i]
                
    # Phase 3: Pair Flips (if stuck)
    if time.time() < deadline - 1:
        signs = best_signs.copy()
        sums = np.zeros(K, dtype=np.float64)
        for k, terms in enumerate(ap_indices):
            sums[k] = np.sum(signs[terms])
        current_max = np.max(np.abs(sums))
        
        for i in range(N):
            if time.time() > deadline - 1:
                break
            for j in range(i+1, N):
                if time.time() > deadline - 1:
                    break
                # Flip i and j
                di = -2.0 * signs[i]
                dj = -2.0 * signs[j]
                
                idxs_i = idx_to_aps_np[i]
                idxs_j = idx_to_aps_np[j]
                
                # Combined indices
                # This is expensive to compute unique every time.
                # Just update both.
                old_sums_i = sums[idxs_i].copy()
                old_sums_j = sums[idxs_j].copy()
                
                sums[idxs_i] += di
                sums[idxs_j] += dj
                signs[i] = -signs[i]
                signs[j] = -signs[j]
                
                new_max = np.max(np.abs(sums))
                
                if new_max < current_max:
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_signs = signs.copy()
                else:
                    # Revert
                    sums[idxs_i] -= di
                    sums[idxs_j] -= dj
                    signs[i] = -signs[i]
                    signs[j] = -signs[j]

    return [int(s) for s in best_signs]
