import time
import random
from math import gcd
import bisect

def construct():
    start_time = time.time()
    # Deadline with safety margin
    deadline = start_time + 55.0
    
    n = 30
    
    # Precompute LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)
    
    # W[i] = L // (i+1) for i in 0..29
    # Sum c[i]/(i+1) = (sum c[i] * W[i]) / L
    W = [L // (i + 1) for i in range(n)]
    
    best_abs_K = float('inf')
    best_coeffs = None
    
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
                
                # c = -1
                new_sums.append(s - w)
                new_masks.append(m)
                
                # c = 0
                new_sums.append(s)
                new_masks.append(m + multiplier)
                
                # c = 1
                new_sums.append(s + w)
                new_masks.append(m + 2 * multiplier)
                
            sums = new_sums
            masks = new_masks
        
        return sums, masks

    # Generate Half A (indices 0 to 14)
    sums_a, masks_a = generate_half(0, mid)
    
    # Generate Half B (indices 15 to 29)
    sums_b, masks_b = generate_half(mid, n)
    
    # Sort Half B by sum DESCENDING
    # This makes B_prime = -B_sums sorted ASCENDING
    pairs_b = list(zip(sums_b, masks_b))
    pairs_b.sort(key=lambda x: x[0], reverse=True)
    
    arr_b_sums = [p[0] for p in pairs_b]
    arr_b_masks = [p[1] for p in pairs_b]
    
    # B_prime sums = -arr_b_sums (Ascending)
    # We don't need to store B_prime sums explicitly, just use -arr_b_sums[j]
    
    # Sort Half A by sum ASCENDING
    pairs_a = list(zip(sums_a, masks_a))
    pairs_a.sort(key=lambda x: x[0])
    arr_a_sums = [p[0] for p in pairs_a]
    arr_a_masks = [p[1] for p in pairs_a]
    
    # Helper to decode mask to coeffs
    def decode_mask(mask, offset, length):
        coeffs = [0] * length
        for j in range(length):
            digit = (mask // (3 ** j)) % 3
            coeffs[j] = digit - 1
        return coeffs

    # Two-pointer scan
    i = 0
    j = 0
    len_a = len(arr_a_sums)
    len_b = len(arr_b_sums)
    
    while i < len_a and j < len_b:
        if time.time() > deadline:
            break
            
        a_sum = arr_a_sums[i]
        b_sum = arr_b_sums[j]
        # We want a_sum + b_sum to be close to 0.
        # diff = a_sum + b_sum
        diff = a_sum + b_sum
        
        if diff == 0:
            # Check neighbors for non-zero
            if i + 1 < len_a:
                d1 = abs(arr_a_sums[i+1] + b_sum)
                if d1 > 0 and d1 < best_abs_K:
                    best_abs_K = d1
                    coeffs_a = decode_mask(arr_a_masks[i+1], 0, mid)
                    coeffs_b = decode_mask(arr_b_masks[j], 0, mid)
                    best_coeffs = coeffs_a + coeffs_b
            if j + 1 < len_b:
                d2 = abs(a_sum + arr_b_sums[j+1])
                if d2 > 0 and d2 < best_abs_K:
                    best_abs_K = d2
                    coeffs_a = decode_mask(arr_a_masks[i], 0, mid)
                    coeffs_b = decode_mask(arr_b_masks[j+1], 0, mid)
                    best_coeffs = coeffs_a + coeffs_b
            i += 1
            j += 1
        elif diff < 0:
            # a + b < 0. To increase sum, increase a (since A is sorted asc).
            d = abs(diff)
            if d < best_abs_K:
                best_abs_K = d
                coeffs_a = decode_mask(arr_a_masks[i], 0, mid)
                coeffs_b = decode_mask(arr_b_masks[j], 0, mid)
                best_coeffs = coeffs_a + coeffs_b
            i += 1
        else:
            # a + b > 0. To decrease sum, increase b?
            # B is sorted DESC. So arr_b_sums[j] is large.
            # To decrease a + b, we need to decrease b.
            # Since B is sorted DESC, arr_b_sums[j+1] is smaller.
            # So we should increment j.
            d = abs(diff)
            if d < best_abs_K:
                best_abs_K = d
                coeffs_a = decode_mask(arr_a_masks[i], 0, mid)
                coeffs_b = decode_mask(arr_b_masks[j], 0, mid)
                best_coeffs = coeffs_a + coeffs_b
            j += 1

    if best_coeffs is None:
        # Fallback
        best_coeffs = [1] + [-1] + [0]*28
        best_abs_K = abs(W[0] - W[1])

    # Local Search / Polish
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
                
                new_K = current_K + (new_c - old_c) * W[i]
                
                if new_K == 0:
                    continue
                
                new_abs_K = abs(new_K)
                if new_abs_K < best_abs_K:
                    current[i] = new_c
                    current_K = new_K
                    best_abs_K = new_abs_K
                    best_coeffs = list(current)
                    improved = True
                    old_c = new_c
                    break  # Move to next index

    # Additional random restarts
    if time.time() < deadline - 5:
        for restart in range(200):
            if time.time() > deadline:
                break
            
            current = list(best_coeffs)
            current_K = sum(current[i] * W[i] for i in range(n))
            
            # Perturb
            num_perturb = random.randint(1, 5)
            for _ in range(num_perturb):
                idx = random.randint(0, n - 1)
                current[idx] = random.choice([-1, 0, 1])
            
            current_K = sum(current[i] * W[i] for i in range(n))
            if current_K == 0:
                continue
            
            # Local search
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
