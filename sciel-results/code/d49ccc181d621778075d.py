    def construct():
        import time
        import numpy as np
        
        start_time = time.time()
        deadline = start_time + 100  # 100s limit
        
        # Choose L
        L = 2_000_000
        
        # Generate powerful numbers
        max_b = int(L ** (1/3)) + 2
        # Sieve for square-free
        is_sq_free = [True] * (max_b + 1)
        for i in range(2, int(max_b**0.5) + 1):
            if is_sq_free[i]:
                for j in range(i*i, max_b+1, i*i):
                    is_sq_free[j] = False
        
        powerful = set()
        for b in range(1, max_b + 1):
            if not is_sq_free[b]:
                continue
            b3 = b ** 3
            if b3 > L:
                break
            max_a = int((L // b3) ** 0.5)
            for a in range(1, max_a + 1):
                val = a * a * b3
                if val <= L:
                    powerful.add(val)
        
        P = sorted(list(powerful))
        M = len(P)
        
        # Count sums
        # Max sum is 2*L
        counts = np.zeros(2*L + 1, dtype=np.int32)
        
        # Iterate pairs
        # Use numpy for inner loop if possible?
        # P is a list.
        # for i, a in enumerate(P):
        #    # b must be >= a
        #    # We can take slice P[i:]
        #    # Sums = a + P[i:]
        #    # Update counts
        #    # This creates an array of size M-i
        #    # Total operations ~ M^2/2
        #    # But creating arrays is fast in C.
        #    # Updating counts: np.add.at(counts, sums, 1)
        #    # np.add.at is slow.
        #    # Better: use a dictionary? No, numpy array is better for memory.
        #    # Alternative: bincount?
        #    # np.bincount(sums, minlength=2*L+1)
        #    # This is fast!
        
        # Let's try bincount approach.
        # It creates a new array each time.
        # Total time: sum of (M-i) log(M-i) or similar?
        # Actually bincount is O(N).
        # So total O(M^2).
        
        best_N = -1
        best_count = 0
        best_pairs = []
        
        # To avoid huge memory spikes, we can process in chunks?
        # Or just do it all at once if M is small enough.
        # M ~ 34,000.
        # Array size 34,000.
        # Loop 34,000 times.
        # Each step:
        #   slice = P[i:] (view)
        #   sums = a + slice (new array)
        #   bc = np.bincount(sums, minlength=2*L+1)
        #   counts += bc
        # This is efficient.
        
        for i, a in enumerate(P):
            if time.time() > deadline:
                break
            b_slice = P[i:]
            sums = a + b_slice
            # sums are within 2*L
            bc = np.bincount(sums, minlength=2*L+1)
            counts += bc
            
        # Find max
        max_val = np.max(counts)
        if max_val == 0:
             # Fallback
             return {'N': 2, 'pairs': [[1, 1]]}
             
        # Find index
        # There might be multiple N with same max count. Pick one.
        idxs = np.where(counts == max_val)[0]
        N = int(idxs[0])
        
        # Retrieve pairs
        pairs = []
        for a in P:
            b = N - a
            if b < a:
                continue
            if b <= L and b in powerful: # Check membership
                pairs.append([a, b])
                
        return {'N': N, 'pairs': pairs}
    