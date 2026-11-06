<<<<<<< SEARCH
    # Use segmented sieve approach: for each P, mark all P-smooth numbers
    # by dividing out small primes in a vectorized manner
    SEGMENT_SIZE = 50000  # Larger segments for fewer boundary checks
    best_n = None
    best_lpf = float('inf')
    
    # Try multiple smoothness thresholds
    for P in [30, 50, 100, 200, 500, 1000, 2000]:
        if time.time() > deadline:
            break
        
        # Get primes up to P
        primes_up_to_P = [p for p in small_primes if p <= P]
        
        # For each segment, compute which numbers are P-smooth
        for seg_start in range(N_MIN, N_MAX + 1, SEGMENT_SIZE):
            if time.time() > deadline:
                break
            
            seg_end = min(seg_start + SEGMENT_SIZE, N_MAX + 1)
            seg_len = seg_end - seg_start
            
            # Initialize array with the numbers in this segment
            nums = np.arange(seg_start, seg_end, dtype=np.int64)
            
            # For each prime <= P, divide it out completely
            for p in primes_up_to_P:
                # Create mask of positions divisible by p
                mask = (nums % p == 0)
                if not np.any(mask):
                    continue
                # Divide out all factors of p
                while np.any(mask):
                    nums[mask] //= p
                    mask = (nums[mask] % p == 0)
                    if not np.any(mask):
                        break
                # Reconstruct mask for next iteration
                mask = (nums % p == 0)
            
            # Numbers that are P-smooth have value 1 after removing all prime factors <= P
            smooth_mask = (nums == 1)
            
            # Find consecutive smooth pairs
            # smooth_mask[i] and smooth_mask[i+1] both True
            consecutive = smooth_mask[:-1] & smooth_mask[1:]
            
            if np.any(consecutive):
                indices = np.where(consecutive)[0]
                for idx in indices:
                    n_candidate = seg_start + idx
                    if N_MIN <= n_candidate <= N_MAX:
                        # Verify with exact trial division
                        lpf_n = largest_prime_factor(n_candidate)
                        lpf_np1 = largest_prime_factor(n_candidate + 1)
                        max_lpf = max(lpf_n, lpf_np1)
                        
                        if max_lpf < best_lpf:
                            best_lpf = max_lpf
                            best_n = n_candidate
=======
    # Use segmented sieve approach: for each P, mark all P-smooth numbers
    SEGMENT_SIZE = 50000  # Larger segments for fewer boundary checks
    best_n = None
    best_lpf = float('inf')
    
    # Try multiple smoothness thresholds
    for P in [30, 50, 100, 200, 500, 1000, 2000]:
        if time.time() > deadline:
            break
        
        # Get primes up to P
        primes_up_to_P = [p for p in small_primes if p <= P]
        
        # For each segment, compute which numbers are P-smooth
        for seg_start in range(N_MIN, N_MAX + 1, SEGMENT_SIZE):
            if time.time() > deadline:
                break
            
            seg_end = min(seg_start + SEGMENT_SIZE, N_MAX + 1)
            seg_len = seg_end - seg_start
            
            # Initialize array with the numbers in this segment
            nums = np.arange(seg_start, seg_end, dtype=np.int64)
            
            # For each prime <= P, divide it out completely
            for p in primes_up_to_P:
                # Keep dividing by p until no more factors
                while True:
                    mask = (nums % p == 0)
                    if not np.any(mask):
                        break
                    nums[mask] //= p
            
            # Numbers that are P-smooth have value 1 after removing all prime factors <= P
            smooth_mask = (nums == 1)
            
            # Find consecutive smooth pairs
            # smooth_mask[i] and smooth_mask[i+1] both True
            consecutive = smooth_mask[:-1] & smooth_mask[1:]
            
            if np.any(consecutive):
                indices = np.where(consecutive)[0]
                for idx in indices:
                    n_candidate = seg_start + idx
                    if N_MIN <= n_candidate <= N_MAX:
                        # Verify with exact trial division
                        lpf_n = largest_prime_factor(n_candidate)
                        lpf_np1 = largest_prime_factor(n_candidate + 1)
                        max_lpf = max(lpf_n, lpf_np1)
                        
                        if max_lpf < best_lpf:
                            best_lpf = max_lpf
                            best_n = n_candidate
>>>>>>> REPLACE
