import time
from bisect import bisect_right

def construct():
        t0 = time.time()
        DEADLINE = 50.0 # Safety margin

        N = 10000
        is_prime = bytearray(b'\x01') * (N + 1)
        is_prime[0] = is_prime[1] = 0
        for x in range(2, int(N ** 0.5) + 1):
            if is_prime[x]:
                is_prime[x * x::x] = bytearray(len(is_prime[x * x::x]))
        primes = [i for i in range(2, N + 1) if is_prime[i]]
        n = len(primes)

        # dp[i] = list of (gap, length, prev_idx, prev_gap_idx_in_dp)
        # We store prev_gap_idx to retrieve the gap from the previous state's dp list?
        # Or just store the gap value directly in the tuple?
        # Storing gap value is easier.
        # dp[i] = list of (last_gap, length, prev_idx, prev_last_gap)
        
        dp = [[] for _ in range(n)]
        
        # To speed up binary search, we can store the gaps in a separate list?
        # Or just use the list of tuples.
        
        for i in range(n):
            if time.time() - t0 > DEADLINE:
                break
            pi = primes[i]
            cands = []
            
            for j in range(i):
                gap = pi - primes[j]
                
                # Start new
                cands.append((gap, 2, j, -1)) # prev_gap -1 means start
                
                if dp[j]:
                    # Find best in dp[j] with gap <= current gap
                    # dp[j] is sorted by gap
                    # We want max length among those with gap <= current gap
                    # Since length is strictly increasing with gap in the frontier,
                    # the best is the one with the largest gap <= current gap.
                    
                    # Binary search
                    lo, hi = 0, len(dp[j]) - 1
                    best_k = -1
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        if dp[j][mid][0] <= gap:
                            best_k = mid
                            lo = mid + 1
                        else:
                            hi = mid - 1
                    
                    if best_k != -1:
                        lg, ln, pj, plg = dp[j][best_k]
                        cands.append((gap, ln + 1, j, lg))
            
            # Reduce to frontier
            if cands:
                # Sort by gap
                cands.sort(key=lambda x: x[0])
                frontier = []
                max_len = 0
                for c in cands:
                    g, l, pi_, plg = c
                    if l > max_len:
                        frontier.append(c)
                        max_len = l
                dp[i] = frontier

        # Find best
        best_len = 0
        best_i = -1
        best_entry = None
        for i in range(n):
            if not dp[i]:
                continue
            for entry in dp[i]:
                if entry[1] > best_len:
                    best_len = entry[1]
                    best_i = i
                    best_entry = entry

        if best_len <= 1:
            return [primes[0]]

        # Reconstruct
        # We need to trace back.
        # best_entry = (last_gap, length, prev_idx, prev_last_gap)
        # If prev_idx is -1, stop.
        # Else, go to dp[prev_idx] and find the entry with gap == prev_last_gap.
        # Note: There might be multiple entries with same gap?
        # In our frontier reduction, we keep only the best length for each gap?
        # No, we keep entries where length is strictly increasing.
        # If two entries have the same gap, only the one with higher length survives?
        # In `reduce_to_frontier`, if we have (g, l1) and (g, l2) with l1 < l2,
        # sorted by gap, they are adjacent.
        # The first one (l1) is added. max_len = l1.
        # The second one (l2) has l2 > max_len, so it is added.
        # So we might have multiple entries with the same gap in the frontier?
        # No, if gaps are equal, they are sorted together.
        # But logically, for a fixed gap, we only care about the max length.
        # We should group by gap and keep max length BEFORE sorting and filtering.
        
        # Let's fix the reduction logic to be robust.
        
        result = []
        cur_i = best_i
        cur_gap = best_entry[0]
        # We need to find the entry in dp[best_i] with gap cur_gap.
        # Actually, best_entry IS the entry.
        
        while True:
            result.append(primes[cur_i])
            if best_entry[2] == -1:
                break
            
            prev_i = best_entry[2]
            prev_gap = best_entry[3]
            
            # Find entry in dp[prev_i] with gap prev_gap
            # Since dp[prev_i] is a frontier, and prev_gap was used to extend,
            # it must be in the frontier.
            # We can linear search or binary search.
            # Given small size, linear is fine.
            target_entry = None
            for e in dp[prev_i]:
                if e[0] == prev_gap:
                    target_entry = e
                    break
            
            if target_entry is None:
                # Should not happen
                break
                
            cur_i = prev_i
            best_entry = target_entry

        result.reverse()
        return result
