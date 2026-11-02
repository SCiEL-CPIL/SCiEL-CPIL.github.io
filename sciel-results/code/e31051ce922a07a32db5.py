import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    # Generate candidates
    candidates = []
    
    # Type 1: One 2, rest 0: ±2e_i (Norm sq 4)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0: ±e_i ± e_j ± e_k ± e_l (Norm sq 4)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    # Type 3: Three 1s, rest 0: ±e_i ± e_j ± e_k (Norm sq 3)
    for idxs in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix M
    M = candidates @ candidates.T
    
    # Compute squared norms
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Max norm squared is 4 (from Type 1 and 2)
    # We assume the set will contain at least one vector with norm sq 4.
    # If not, R^2 would be 3, but Type 1/2 are very packable, so this is unlikely to be optimal.
    # Even if R^2=3, the constraints for Type 3 vs Type 3 are the same (dot <= 1).
    # Constraints for Type 1/2 vs Type 1/2 would be looser (dot <= 2.5 -> 2), same as R^2=4.
    # Constraints for Type 1/2 vs Type 3 would be looser (dot <= 1.5 -> 1), same as R^2=4.
    # So using R^2=4 is a safe conservative bound that yields valid solutions even if R^2 turns out to be 3.
    R_sq = 4
    
    # Threshold matrix: Max allowed dot product for pair (i,j)
    # 2 * M[i,j] <= norms_sq[i] + norms_sq[j] - R_sq
    # M[i,j] <= (norms_sq[i] + norms_sq[j] - R_sq) / 2
    # Since M is integer, we can use floor division by 2 after ensuring numerator is even?
    # No, just check 2*M <= sum - R_sq.
    # Or Thresh = floor((norms_sq[:, None] + norms_sq[None, :] - R_sq) / 2)
    
    # Vectorized calculation of thresholds
    # norms_sq shape (N, 1), norms_sq shape (1, N)
    num = norms_sq[:, None] + norms_sq[None, :] - R_sq
    # Integer division by 2
    # If num is odd, (num-1)/2 is floor(num/2).
    # We want M[i,j] <= num/2. Since M is int, M[i,j] <= floor(num/2).
    Thresh = num // 2
    
    best_set = None
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Strategy A: Random orderings
    for trial in range(50):
        if time.time() > deadline:
            break
        
        if trial == 0:
            order = np.arange(N)
        else:
            order = rng.permutation(N)
        
        selected_mask = np.zeros(N, dtype=bool)
        count = 0
        
        for idx in order:
            if count == 0:
                selected_mask[idx] = True
                count += 1
                continue
            
            dots = M[idx, selected_mask]
            thresh = Thresh[idx, selected_mask]
            if np.all(dots <= thresh):
                selected_mask[idx] = True
                count += 1
        
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()
    
    # Strategy B: Greedy with "least conflicts" priority
    if time.time() < deadline - 20:
        # Conflict if M[i,j] > Thresh[i,j]
        conflict_matrix = M > Thresh
        conflict_counts = np.sum(conflict_matrix, axis=1) - 1  # exclude self (M[i,i]=norms_sq[i], Thresh[i,i]=(2*norms_sq[i]-4)//2 = norms_sq[i]-2. So M[i,i] > Thresh[i,i] is True. We want to exclude self from count)
        # Actually, for i=j, M[i,i] = norms_sq[i]. Thresh[i,i] = (2*norms_sq[i]-4)//2.
        # If norms_sq=4, Thresh=2. M=4. 4>2 is True.
        # If norms_sq=3, Thresh=1. M=3. 3>1 is True.
        # So diagonal is always conflict. Subtracting 1 is correct.
        
        order_b = np.argsort(conflict_counts, kind='stable')
        
        selected_mask = np.zeros(N, dtype=bool)
        count = 0
        
        for idx in order_b:
            if count == 0:
                selected_mask[idx] = True
                count += 1
                continue
            
            dots = M[idx, selected_mask]
            thresh = Thresh[idx, selected_mask]
            if np.all(dots <= thresh):
                selected_mask[idx] = True
                count += 1
        
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()
    
    # Strategy C: Greedy with "most central" (highest degree in compatibility graph)
    if time.time() < deadline - 10:
        compat_counts = N - 1 - conflict_counts
        order_c = np.argsort(-compat_counts, kind='stable')
        
        selected_mask = np.zeros(N, dtype=bool)
        count = 0
        
        for idx in order_c:
            if count == 0:
                selected_mask[idx] = True
                count += 1
                continue
            
            dots = M[idx, selected_mask]
            thresh = Thresh[idx, selected_mask]
            if np.all(dots <= thresh):
                selected_mask[idx] = True
                count += 1
        
        if count > best_count:
            best_count = count
            best_set = selected_mask.copy()
    
    # Strategy D: Try adding to best set by scanning unselected
    if best_set is not None and time.time() < deadline - 5:
        unselected = np.where(~best_set)[0]
        for idx in unselected:
            if time.time() > deadline - 2:
                break
            dots = M[idx, best_set]
            thresh = Thresh[idx, best_set]
            if np.all(dots <= thresh):
                best_set[idx] = True
                best_count += 1
    
    if best_set is None:
        vectors = []
        for i in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[i] = 2 * s
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return candidates[best_set].astype(np.int64)
