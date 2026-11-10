import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    N = 10000000
    MAX_B = 2 * N // 3  # 6666666

    # Linear sieve for sigma
    sigma = np.zeros(N + 1, dtype=np.int64)
    sp = np.zeros(N + 1, dtype=np.int64)
    pe = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    primes = []

    for i in range(2, N + 1):
        if time.time() - start_time > 25.0:
            break
        if sp[i] == 0:
            sp[i] = i
            pe[i] = i
            sigma[i] = 1 + i
            primes.append(i)
        for p in primes:
            if p > sp[i] or i * p > N:
                break
            sp[i * p] = p
            if i % p == 0:
                pe[i * p] = pe[i] * p
                m = i // pe[i]
                sigma[i * p] = sigma[i] + sigma[m] * pe[i] * p
                break
            else:
                pe[i * p] = p
                sigma[i * p] = sigma[i] * (1 + p)

    # Initial best from known small pair
    # sigma(11)=12, sigma(6)=12. a=11, b=6.
    # 2*11 - 3*6 = 22 - 18 = 4.
    # Error = 4 / (2*6) = 4/12 = 1/3.
    # Wait, earlier I said 1/6.
    # a/b = 11/6. 3/2 = 9/6. Diff = 2/6 = 1/3.
    # Formula: |2a-3b| / 2b = |22-18|/12 = 4/12 = 1/3.
    best_error = 1.0 / 3.0
    best_a = 11
    best_b = 6

    # Vectorized search for k=1
    # b must be odd for a to be integer.
    # a = (3b +/- 1) / 2
    # We want to maximize b.
    
    # Generate odd b's from MAX_B down to 1
    # To save memory, we can process in chunks or just use the full array if memory allows.
    # MAX_B is ~6.6M. Array of ints is ~50MB. Acceptable.
    
    # Create array of odd numbers
    # np.arange(MAX_B, 0, -2) gives odd numbers if MAX_B is odd, else even.
    # We need to ensure we only consider b where a <= N.
    # a = (3b+1)/2 <= N => 3b+1 <= 2N => b <= (2N-1)/3
    # MAX_B is already 2N//3.
    
    b_arr = np.arange(MAX_B, 0, -2, dtype=np.int64)
    if b_arr[0] % 2 == 0:
        b_arr = b_arr[1:] # Ensure odd

    # Calculate a candidates
    # a1 = (3b + 1) // 2
    # a2 = (3b - 1) // 2
    # We need to check if a1, a2 are within [1, N]
    
    a1 = (3 * b_arr + 1) // 2
    a2 = (3 * b_arr - 1) // 2
    
    # Validity masks
    mask1 = (a1 >= 1) & (a1 <= N)
    mask2 = (a2 >= 1) & (a2 <= N)
    
    # Check sigma equality
    # We want the largest b, which is the first element in b_arr (since it's descending)
    # But we need to check both a1 and a2.
    
    found = False
    # Check a1
    if np.any(mask1):
        b_valid = b_arr[mask1]
        a_valid = a1[mask1]
        sig_b = sigma[b_valid]
        sig_a = sigma[a_valid]
        matches = np.where(sig_b == sig_a)[0]
        if len(matches) > 0:
            # First match is the largest b
            idx = matches[0]
            best_b = int(b_valid[idx])
            best_a = int(a_valid[idx])
            best_error = 1.0 / (2.0 * best_b)
            found = True
            
    # Check a2
    if np.any(mask2):
        b_valid = b_arr[mask2]
        a_valid = a2[mask2]
        sig_b = sigma[b_valid]
        sig_a = sigma[a_valid]
        matches = np.where(sig_b == sig_a)[0]
        if len(matches) > 0:
            idx = matches[0]
            b_candidate = int(b_valid[idx])
            a_candidate = int(a_valid[idx])
            err_candidate = 1.0 / (2.0 * b_candidate)
            if err_candidate < best_error:
                best_b = b_candidate
                best_a = a_candidate
                best_error = err_candidate
                found = True

    # If we found a k=1 solution, we can check k=2, 3... for larger b
    # But b must be > k * best_b
    # We iterate k
    max_k = 100 # Arbitrary limit, as b decreases, k increases
    # Actually, we can just loop k while 1/(2*1) < best_error is false? No.
    # We check if there exists b > k * best_b.
    # Max b is MAX_B.
    # So we need k * best_b < MAX_B => k < MAX_B / best_b
    
    if found:
        max_k_to_check = int(MAX_B / best_b)
        for k in range(2, max_k_to_check + 1):
            if time.time() - start_time > DEADLINE - 5:
                break
            # b must be > k * best_b
            b_min = k * best_b + 1
            if b_min > MAX_B:
                break
            
            # b parity:
            # 2a - 3b = k
            # 3b + k must be even => b and k must have same parity?
            # 3b + k even => b + k even => b, k same parity.
            # So if k is even, b must be even.
            # If k is odd, b must be odd.
            
            if k % 2 == 0:
                # b even
                # Start from MAX_B if even, else MAX_B-1
                start_b = MAX_B if MAX_B % 2 == 0 else MAX_B - 1
                # b > b_min
                if start_b <= b_min:
                    continue
                b_arr_k = np.arange(start_b, b_min, -2, dtype=np.int64)
            else:
                # b odd
                start_b = MAX_B if MAX_B % 2 == 1 else MAX_B - 1
                if start_b <= b_min:
                    continue
                b_arr_k = np.arange(start_b, b_min, -2, dtype=np.int64)
                
            if len(b_arr_k) == 0:
                continue
                
            a_k = (3 * b_arr_k + k) // 2
            mask_k = (a_k >= 1) & (a_k <= N)
            
            if np.any(mask_k):
                b_valid = b_arr_k[mask_k]
                a_valid = a_k[mask_k]
                sig_b = sigma[b_valid]
                sig_a = sigma[a_valid]
                matches = np.where(sig_b == sig_a)[0]
                if len(matches) > 0:
                    idx = matches[0]
                    b_candidate = int(b_valid[idx])
                    a_candidate = int(a_valid[idx])
                    err_candidate = k / (2.0 * b_candidate)
                    if err_candidate < best_error:
                        best_b = b_candidate
                        best_a = a_candidate
                        best_error = err_candidate
                        # Update max_k_to_check? 
                        # New best_b is larger? No, b_candidate < MAX_B.
                        # But best_error decreased.
                        # We can break if 1/(2*1) >= best_error? No.
                        # We can reduce max_k_to_check.
                        max_k_to_check = int(MAX_B / best_b)
                        # Note: best_b changed, so the loop condition `k < MAX_B/best_b` changes.
                        # But we are inside a for loop. We can't easily change the range.
                        # However, if we found a better error, the required b for subsequent k's is higher.
                        # If we found a very good error, we might not need to check large k's.
                        # But we are iterating k upwards.
                        # If we find a solution, we update best_error.
                        # The condition to break the outer k loop is if k >= MAX_B / best_b.
                        # Since best_b might have increased (if we found a larger b? No, we are checking b > old_best_b * k).
                        # Wait, if we find a solution at b_candidate, is it possible that b_candidate > old best_b?
                        # Yes, if k is small.
                        # But we are checking k=2, 3...
                        # The b's we check are > k * old_best_b.
                        # So b_candidate > 2 * old_best_b.
                        # So best_b increases.
                        # So MAX_B / best_b decreases.
                        # So we can break sooner.
                        if k >= max_k_to_check:
                            break

    return [int(best_a), int(best_b)]
