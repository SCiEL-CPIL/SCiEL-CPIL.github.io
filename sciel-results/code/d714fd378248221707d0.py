import numpy as np

def construct():
    n = 128
    targets = np.arange(128, 257)
    target_mask = np.zeros(257, dtype=bool)
    target_mask[128:257] = True
    
    A_mask = np.zeros(129, dtype=bool)
    counts = np.zeros(257, dtype=np.int32)
    
    # Precompute valid indices for A (empty initially)
    # We will update this as A grows
    
    while True:
        # Check coverage
        if np.all(counts[targets] > 0):
            break
            
        best_x = -1
        best_score = float('inf')
        best_delta = None
        
        # Get current elements of A
        valid_a = np.where(A_mask)[0]
        
        for x in range(129):
            if A_mask[x]:
                continue
                
            # Calculate delta counts for adding x
            # Sums are x + a for a in valid_a, and x + x
            if len(valid_a) > 0:
                idx = x + valid_a
                delta = np.bincount(idx, minlength=257).astype(np.int32)
            else:
                delta = np.zeros(257, dtype=np.int32)
            
            delta[2 * x] += 1
            
            new_counts = counts + delta
            
            # Evaluate score
            # Primary: minimize uncovered targets
            # Secondary: minimize max representation
            # Tertiary: minimize variance (sum of squares) to balance
            
            uncovered = np.sum(new_counts[targets] == 0)
            max_c = np.max(new_counts[targets])
            
            # If all covered, we want to minimize max_c
            # If not all covered, we want to maximize coverage first
            if uncovered > 0:
                score = uncovered * 10000 + max_c
            else:
                # Balance term: sum of squares
                ss = np.sum(new_counts[targets] ** 2)
                score = max_c * 100 + ss
            
            if score < best_score:
                best_score = score
                best_x = x
                best_delta = delta
        
        if best_x == -1:
            break
            
        A_mask[best_x] = True
        counts += best_delta
        
    return list(np.where(A_mask)[0])
