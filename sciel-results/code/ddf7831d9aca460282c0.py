import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin for 60s limit
    
    n = 30
    
    # Precompute LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)
    
    # A[i] = L // i, so sum c[i]/(i+1) = sum c[i] * A[i+1] / L
    # We work in integer weights W[i] = L // (i+1) for i in 0..29
    W = [L // (i + 1) for i in range(n)]
    
    # We want to find c in {-1,0,1}^30 such that K = sum c[i]*W[i] != 0 and |K| is minimized.
    # The value is |K|/L. Minimizing |K|/L is equivalent to minimizing |K| since L is constant.
    
    best_abs_K = float('inf')
    best_coeffs = None
    
    def try_update(coeffs):
        nonlocal best_abs_K, best_coeffs
        K = sum(coeffs[i] * W[i] for i in range(n))
        if K == 0:
            return
        abs_K = abs(K)
        if abs_K < best_abs_K:
            best_abs_K = abs_K
            best_coeffs = list(coeffs)
    
    # Phase 1: Meet-in-the-middle search for the global minimum |K|
    # Split into two halves of 15 each
    mid = n // 2  # 15
    
    def generate_half(start, end):
        """Generate all (sum, mask) pairs for indices start to end-1."""
        length = end - start
        sums = [0]
        masks = [0]
        
        for i in range(length):
            w = W[start + i]
            new_sums = []
            new_masks = []
            multiplier = 3 ** i
            
            for k in range(len(sums)):
                s = sums[k]
                m = masks[k]
                
                # c = -1 (digit 0)
                new_sums.append(s - w)
                new_masks.append(m)
                
                # c = 0 (digit 1)
                new_sums.append(s)
                new_masks.append(m + multiplier)
                
                # c = 1 (digit 2)
                new_sums.append(s + w)
                new_masks.append(m + 2 * multiplier)
                
            sums = new_sums
            masks = new_masks
        
        return sums, masks
    
    # Generate Half A (indices 0 to 14)
    sums_a, masks_a = generate_half(0, mid)
    
    # Generate Half B (indices 15 to 29)
    sums_b, masks_b = generate_half(mid, n)
    
    # Sort Half B by sum
    pairs_b = list(zip(sums_b, masks_b))
    pairs_b.sort(key=lambda x: x[0])
    
    arr_b_sums = [p[0] for p in pairs_b]
    arr_b_masks = [p[1] for p in pairs_b]
    
    # For each sum in A, find the closest sum in B to -sum_a
    # Use binary search
    import bisect
    
    for i in range(len(sums_a)):
        if time.time() > deadline:
            break
            
        a_sum = sums_a[i]
        target = -a_sum
        
        # Find insertion point in sorted arr_b_sums
        idx = bisect.bisect_left(arr_b_sums, target)
        
        candidates = []
        if idx < len(arr_b_sums):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
            
        for c_idx in candidates:
            b_sum = arr_b_sums[c_idx]
            total = a_sum + b_sum
            if total != 0:
                abs_total = abs(total)
                if abs_total < best_abs_K:
                    best_abs_K = abs_total
                    # Decode masks to get coefficients
                    coeffs = [0] * n
                    # Decode mask_a
                    for j in range(mid):
                        digit = (masks_a[i] // (3 ** j)) % 3
                        coeffs[j] = digit - 1
                    # Decode mask_b
                    for j in range(mid):
                        digit = (arr_b_masks[c_idx] // (3 ** j)) % 3
                        coeffs[mid + j] = digit - 1
                    best_coeffs = coeffs
    
    # If MITM found a solution, use it as starting point for local search
    if best_coeffs is None:
        # Fallback: simple nonzero solution
        best_coeffs = [0] * n
        best_coeffs[0] = 1
        best_coeffs[1] = -1  # 1/1 - 1/2 = 1/2
        K = sum(best_coeffs[i] * W[i] for i in range(n))
        best_abs_K = abs(K)
    
    # Phase 2: Local search to refine
    random.seed(42)
    current = list(best_coeffs)
    current_K = sum(current[i] * W[i] for i in range(n))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            old_c = current[i]
            
            # Try all three values for position i
            for new_c in [-1, 0, 1]:
                if new_c == old_c:
                    continue
                
                # Update K incrementally
                new_K = current_K + (new_c - old_c) * W[i]
                
                if new_K == 0:
                    continue
                
                new_abs_K = abs(new_K)
                if new_abs_K < best_abs_K:
                    # Accept the move
                    current[i] = new_c
                    current_K = new_K
                    best_abs_K = new_abs_K
                    best_coeffs = list(current)
                    improved = True
                    old_c = new_c  # Update old_c for subsequent iterations
                    break  # Move to next index
    
    # Phase 3: Additional random restarts if time permits
    if time.time() < deadline - 5:
        for restart in range(100):
            if time.time() > deadline:
                break
            
            # Random restart with small perturbation
            current = list(best_coeffs)
            current_K = sum(current[i] * W[i] for i in range(n))
            
            # Perturb 1-3 random positions
            num_perturb = random.randint(1, 3)
            for _ in range(num_perturb):
                idx = random.randint(0, n - 1)
                current[idx] = random.choice([-1, 0, 1])
            
            current_K = sum(current[i] * W[i] for i in range(n))
            if current_K == 0:
                continue
            
            # Local search from this perturbed state
            improved_local = True
            while improved_local and time.time() < deadline:
                improved_local = False
                indices = list(range(n))
                random.shuffle(indices)
                
                for i in indices:
                    if time.time() > deadline:
                        break
                        
                    old_c = current[i]
                    
                    for new_c in [-1, 0, 1]:
                        if new_c == old_c:
                            continue
                        
                        new_K = current_K + (new_c - old_c) * W[i]
                        
                        if new_K == 0:
                            continue
                        
                        new_abs_K = abs(new_K)
                        if new_abs_K < best_abs_K:
                            current[i] = new_c
                            current_K = new_K
                            best_abs_K = new_abs_K
                            best_coeffs = list(current)
                            improved_local = True
                            old_c = new_c
                            break
    
    return best_coeffs
