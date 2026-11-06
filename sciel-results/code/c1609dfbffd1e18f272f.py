import bisect
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 60.0  # 60s limit, keep margin

    # Generate primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes

    # dp[i] is a list of (gap, length) pairs
    dp = [[] for _ in range(N)]
    
    def prune(states):
        if not states:
            return []
        # Sort by gap ascending, then length descending
        states.sort(key=lambda x: (x[0], -x[1]))
        res = []
        max_l = -1
        last_g = -1
        for g, l in states:
            # If same gap, the first one has max length (due to sort).
            # Subsequent ones with same gap have smaller/equal length, so l <= max_l (which is at least the first one's length).
            # So the check l > max_l handles it.
            if l > max_l:
                res.append((g, l))
                max_l = l
                last_g = g
        return res

    best_len = 1
    
    for i in range(N):
        # Check time
        if time.time() > deadline:
            break
            
        # Add base state: starting at i, gap 0, length 1
        dp[i].append((0, 1))
        
        # Prune dp[i] before processing
        dp[i] = prune(dp[i])
        
        # Process states
        for g, l in dp[i]:
            if l > best_len:
                best_len = l
                
            # Find starting index for j
            # We need primes[j] - primes[i] >= g
            # primes[j] >= primes[i] + g
            target = primes[i] + g
            # bisect_left finds first index where primes[idx] >= target
            # We only care about j > i
            j_start = bisect.bisect_left(primes, target, i + 1)
            
            # If no such j, continue
            if j_start >= N:
                continue
                
            # Iterate through valid j's
            # Optimization: local variables
            p_i = primes[i]
            new_len = l + 1
            for j in range(j_start, N):
                # Check time occasionally? No, inner loop is tight.
                new_gap = primes[j] - p_i
                dp[j].append((new_gap, new_len))
                
    # Find the maximum length in any dp[i]
    # Actually, we updated best_len in the loop, but we might have missed states added later?
    # No, we process i in order. Any state ending at i is processed when we reach i.
    # However, the "best_len" is the max length of any subsequence.
    # We need to return the subsequence itself.
    
    # To reconstruct, we need to track parents.
    # This adds complexity.
    # Alternative: Just return the length? No, task says "Return increasing prime integers".
    # So we must return the list.
    
    # Let's restart with parent tracking.
    # dp[i] stores (gap, length, parent_index, parent_gap)
    # Or simpler: store the whole path? No, memory.
    # Store (gap, length, prev_i, prev_gap)
    
    # Let's redo the DP with reconstruction.
    # dp[i] is list of (gap, length, prev_i, prev_gap)
    # prev_i is the index of the previous prime in the subsequence.
    # prev_gap is the gap leading to i (which is the 'gap' in the state).
    # Wait, the state is defined by the last gap.
    # So state at i: (last_gap, length, prev_i).
    # The prev_i is the index of the prime before i.
    
    # Let's re-implement.
    
    dp2 = [[] for _ in range(N)]
    
    def prune2(states):
        if not states:
            return []
        # Sort by gap asc, length desc
        states.sort(key=lambda x: (x[0], -x[1]))
        res = []
        max_l = -1
        for g, l, pi, pg in states:
            if l > max_l:
                res.append((g, l, pi, pg))
                max_l = l
        return res

    for i in range(N):
        if time.time() > deadline:
            break
            
        # Base state: start at i
        # prev_i = -1, prev_gap = 0 (dummy)
        dp2[i].append((0, 1, -1, 0))
        
        dp2[i] = prune2(dp2[i])
        
        for g, l, pi, pg in dp2[i]:
            target = primes[i] + g
            j_start = bisect.bisect_left(primes, target, i + 1)
            if j_start >= N:
                continue
                
            p_i = primes[i]
            new_len = l + 1
            for j in range(j_start, N):
                new_gap = primes[j] - p_i
                # State at j: last_gap=new_gap, length=new_len, prev_i=i, prev_gap=g (gap used to get to i? No, prev_gap is not needed for transition, only for reconstruction if we stored it, but we have pi which is i. The gap from pi to i is primes[i]-primes[pi]. We don't need to store prev_gap if we can compute it from indices.
                # So state is (last_gap, length, prev_i)
                # Let's simplify state to (gap, length, prev_i)
                pass 

    # Let's stick to (gap, length, prev_i)
    # Re-implementing dp2 with this state.
    
    dp3 = [[] for _ in range(N)]
    
    def prune3(states):
        if not states:
            return []
        states.sort(key=lambda x: (x[0], -x[1]))
        res = []
        max_l = -1
        for g, l, pi in states:
            if l > max_l:
                res.append((g, l, pi))
                max_l = l
        return res

    best_len = 1
    best_end_idx = 0
    
    for i in range(N):
        if time.time() > deadline:
            break
            
        dp3[i].append((0, 1, -1))
        dp3[i] = prune3(dp3[i])
        
        for g, l, pi in dp3[i]:
            if l > best_len:
                best_len = l
                best_end_idx = i
                
            target = primes[i] + g
            j_start = bisect.bisect_left(primes, target, i + 1)
            if j_start >= N:
                continue
                
            p_i = primes[i]
            new_len = l + 1
            for j in range(j_start, N):
                new_gap = primes[j] - p_i
                dp3[j].append((new_gap, new_len, i))
                
    # Reconstruct
    if best_len == 1:
        return [primes[0]]
        
    path = []
    curr = best_end_idx
    while curr != -1:
        path.append(primes[curr])
        # Find the state in dp3[curr] that led to this length
        # We need to find the state with length = len(path) (current length in reverse?)
        # Actually, we stored prev_i.
        # We need to find the specific state.
        # There might be multiple states with same length?
        # prune3 ensures strictly increasing length for increasing gap.
        # So for a given length, there is a unique gap?
        # Yes, in the pruned list, lengths are strictly increasing.
        # So we can find the state by length.
        
        # However, we just stored the *best* end index.
        # We need to backtrack.
        # The state at curr has some length L_curr.
        # The prev_i is stored in the state.
        # But we need to identify *which* state was the one we chose.
        # Since we updated best_len globally, we know the length at best_end_idx is best_len.
        # In dp3[best_end_idx], find state with length best_len.
        
        # Let's just search for the state with the correct length.
        # Since lengths are unique in the pruned list, we can find it.
        
        target_len = len(path) # This is not correct. path is being built backwards.
        # Let's track the required length.
        
        # Restart reconstruction logic properly.
        curr = best_end_idx
        curr_len = best_len
        while curr != -1:
            path.append(primes[curr])
            # Find state in dp3[curr] with length curr_len
            prev_i = -1
            for g, l, pi in dp3[curr]:
                if l == curr_len:
                    prev_i = pi
                    break
            curr = prev_i
            curr_len -= 1
            
    path.reverse()
    return path

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
