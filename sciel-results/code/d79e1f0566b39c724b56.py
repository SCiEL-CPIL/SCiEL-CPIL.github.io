"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Generate all primes <= 10000 using Sieve of Eratosthenes
    limit = 10000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)
    
    # Precompute prime-to-index mapping for O(1) lookup
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    import bisect
    
    # For each prime index i, we store:
    #   dp_gaps[i] = list of (gap, length) sorted by gap
    #   dp_prefix[i] = list of prefix max lengths corresponding to dp_gaps[i]
    #   dp_gaps_flat[i] = flat list of just the gap values for bisect
    dp_gaps = [None] * n
    dp_prefix = [None] * n
    dp_gaps_flat = [None] * n
    
    best_overall = 1
    best_j = 0
    
    for j in range(n):
        if time.time() > deadline:
            break
        candidates = {0: 1}  # gap 0 means single element
        
        for i in range(j):
            gap = primes[j] - primes[i]
            # Query dp_prefix[i] for max length with gap' <= gap
            flat = dp_gaps_flat[i]
            if flat is None:
                continue
            pos = bisect.bisect_right(flat, gap)
            if pos > 0:
                best_prev = dp_prefix[i][pos - 1]
                if best_prev > 0:
                    new_len = best_prev + 1
                    if gap not in candidates or candidates[gap] < new_len:
                        candidates[gap] = new_len
        
        # Build structures for j
        items = sorted(candidates.items())  # (gap, length)
        sg_list = []
        prefix_list = []
        flat_list = []
        cur_max = 0
        for gap, length in items:
            cur_max = max(cur_max, length)
            sg_list.append((gap, cur_max))
            prefix_list.append(cur_max)
            flat_list.append(gap)
        
        dp_gaps[j] = sg_list
        dp_prefix[j] = prefix_list
        dp_gaps_flat[j] = flat_list
        
        # Track best
        max_len_j = cur_max
        if max_len_j > best_overall:
            best_overall = max_len_j
            best_j = j
    
    # Backtrack to reconstruct the sequence
    seq = [primes[best_j]]
    cur_j = best_j
    cur_len = best_overall
    cur_gap = None
    
    # Find the gap at cur_j that achieves cur_len
    if dp_gaps[cur_j]:
        for gap, plen in dp_gaps[cur_j]:
            if plen == cur_len:
                cur_gap = gap
                break
        if cur_gap is None:
            cur_gap = dp_gaps[cur_j][-1][0]
    
    while cur_len > 1:
        if time.time() > deadline:
            break
        prev_prime = primes[cur_j] - cur_gap
        found = prime_to_idx.get(prev_prime, -1)
        
        if found < 0:
            break
        
        # Find the gap' in dp_gaps[found] with gap' <= cur_gap and prefix length == cur_len - 1
        prev_gap = None
        flat = dp_gaps_flat[found]
        pos = bisect.bisect_right(flat, cur_gap)
        if pos > 0:
            # Check if prefix at pos-1 gives cur_len - 1
            if dp_prefix[found][pos - 1] >= cur_len - 1:
                # Find the actual gap that achieves at least cur_len - 1
                # Walk backwards from pos-1 to find the gap where prefix first reaches cur_len-1
                for k in range(pos - 1, -1, -1):
                    if dp_prefix[found][k] >= cur_len - 1:
                        prev_gap = flat[k]
                        break
        
        if prev_gap is None:
            break
        
        seq.append(primes[found])
        cur_j = found
        cur_len -= 1
        cur_gap = prev_gap
    
    seq.reverse()
    
    # Verify
    if len(seq) >= 2:
        gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        if all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1)):
            return seq
    
    # Fallback to greedy
    values = []
    for p in primes:
        if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
            values.append(p)
    return values
# EVOLVE-BLOCK-END
