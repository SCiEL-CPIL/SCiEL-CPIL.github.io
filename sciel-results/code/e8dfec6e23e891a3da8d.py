import time
import random
import numpy as np

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 55.0  # safety margin before 60s kill

    # 1. Precompute APs
    ap_indices = []
    for d in range(1, n + 1):
        for a in range(n):
            # Indices: a, a+d, ... < n
            idx = np.arange(a, n, d)
            if len(idx) > 0:
                ap_indices.append(idx)
    
    M = len(ap_indices)
    
    # Precompute which APs affect which position
    # affected_aps[i] is a list of AP indices k such that i is in ap_indices[k]
    affected_aps = [[] for _ in range(n)]
    for k, idx in enumerate(ap_indices):
        for pos in idx:
            affected_aps[pos].append(k)
    
    # Convert affected_aps to numpy arrays for faster iteration? 
    # Lists are fine for small sizes, but numpy might be faster for indexing if we vectorize the update.
    # However, the update is `ap_sums[k] += delta`. 
    # If we have a list of k's, we can do `ap_sums[np.array(ks)] += delta`.
    affected_aps_np = [np.array(ks, dtype=np.int32) if ks else np.array([], dtype=np.int32) for ks in affected_aps]

    # 2. Initialize with Thue-Morse
    def thue_morse_signs(n):
        signs = np.zeros(n, dtype=np.int8)
        for i in range(n):
            cnt = bin(i).count('1')
            signs[i] = 1 if cnt % 2 == 0 else -1
        return signs

    s = thue_morse_signs(n)
    
    # 3. Initialize ap_sums
    ap_sums = np.zeros(M, dtype=np.int32)
    for k in range(M):
        ap_sums[k] = np.sum(s[ap_indices[k]])
        
    current_max = np.max(np.abs(ap_sums))
    
    # 4. Local Search
    rng = random.Random(42)
    improved = True
    passes = 0
    
    while improved and time.time() < deadline - 5.0:
        improved = False
        passes += 1
        # Shuffle positions to avoid bias
        positions = list(range(n))
        rng.shuffle(positions)
        
        for i in positions:
            if time.time() > deadline - 5.0:
                break
            
            old_sign = s[i]
            new_sign = -old_sign
            delta = 2 * old_sign # sum changes by +2*old_sign (if old was 1, new is -1, diff -2, wait. s_new = -s_old. sum_new = sum_old - s_old + s_new = sum_old - 2*s_old. So delta = -2*old_sign)
            # Let's re-verify:
            # s_old = 1, s_new = -1. Change = -2. delta = -2 * 1 = -2. Correct.
            # s_old = -1, s_new = 1. Change = 2. delta = -2 * (-1) = 2. Correct.
            
            s[i] = new_sign
            
            # Update affected APs
            ks = affected_aps_np[i]
            if len(ks) > 0:
                ap_sums[ks] += delta
            
            new_max = np.max(np.abs(ap_sums))
            
            if new_max < current_max:
                current_max = new_max
                improved = True
            else:
                # Revert
                s[i] = old_sign
                if len(ks) > 0:
                    ap_sums[ks] -= delta
    
    # 5. Try a few random restarts if time allows
    if time.time() < deadline - 10.0:
        best_s = s.copy()
        best_max = current_max
        
        for restart in range(20):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb best solution
            s = best_s.copy()
            ap_sums = np.zeros(M, dtype=np.int32)
            for k in range(M):
                ap_sums[k] = np.sum(s[ap_indices[k]])
            current_max = np.max(np.abs(ap_sums))
            
            # Random flips
            for _ in range(10):
                p = rng.randint(0, n - 1)
                old_sign = s[p]
                s[p] = -old_sign
                delta = -2 * old_sign
                ks = affected_aps_np[p]
                if len(ks) > 0:
                    ap_sums[ks] += delta
            
            # Local search on perturbed
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                positions = list(range(n))
                rng.shuffle(positions)
                for i in positions:
                    if time.time() > deadline - 5.0:
                        break
                    old_sign = s[i]
                    s[i] = -old_sign
                    delta = -2 * old_sign
                    ks = affected_aps_np[i]
                    if len(ks) > 0:
                        ap_sums[ks] += delta
                    
                    new_max = np.max(np.abs(ap_sums))
                    if new_max < current_max:
                        current_max = new_max
                        improved = True
                    else:
                        s[i] = old_sign
                        if len(ks) > 0:
                            ap_sums[ks] -= delta
            
            if current_max < best_max:
                best_s = s.copy()
                best_max = current_max

    return [int(x) for x in best_s]
