import time
from bisect import bisect_right

def construct():
    t0 = time.time()
    DEADLINE = 85.0

    N = 10000
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for x in range(2, int(N ** 0.5) + 1):
        if is_prime[x]:
            is_prime[x * x::x] = bytearray(len(is_prime[x * x::x]))
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    n = len(primes)

    # dp[i] is the Pareto frontier of (last_gap, length, prev_idx, prev_entry_idx)
    # sorted by last_gap ascending. Lengths are strictly increasing.
    dp = [[] for _ in range(n)]

    for i in range(n):
        if time.time() - t0 >= DEADLINE:
            break
        pi = primes[i]
        frontier = []  # list of (gap, length, prev_idx, prev_entry_idx), sorted by gap, lengths strictly increasing
        
        for j in range(i):
            if time.time() - t0 >= DEADLINE:
                break
            gap = pi - primes[j]
            
            # Option 1: Start new sequence
            new_len = 2
            new_pi = j
            new_pei = -1
            
            # Check if dominated by existing frontier
            dominated = False
            # Find if any frontier entry has gap <= this gap and length >= new_len
            # Since frontier is sorted by gap with increasing lengths, we can binary search
            # Find rightmost entry with gap <= current gap
            lo, hi = 0, len(frontier) - 1
            best_k = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if frontier[mid][0] <= gap:
                    best_k = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            
            if best_k != -1:
                if frontier[best_k][1] >= new_len:
                    dominated = True
            
            if not dominated:
                # Add to frontier and remove dominated entries
                # Find insertion point
                ins_pos = 0
                for k in range(len(frontier)):
                    if frontier[k][0] <= gap:
                        ins_pos = k + 1
                    else:
                        break
                # Remove entries at ins_pos and beyond that have length <= new_len
                while ins_pos < len(frontier) and frontier[ins_pos][1] <= new_len:
                    ins_pos += 1
                frontier.insert(ins_pos, (gap, new_len, new_pi, new_pei))
            
            # Option 2: Extend existing sequence from j
            if dp[j]:
                # Find largest last_gap <= gap in dp[j]
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
                    lg, ln, pj, pei = dp[j][best_k]
                    new_len = ln + 1
                    new_pi = j
                    new_pei = best_k
                    
                    # Check if dominated by existing frontier
                    dominated = False
                    lo, hi = 0, len(frontier) - 1
                    best_k2 = -1
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        if frontier[mid][0] <= gap:
                            best_k2 = mid
                            lo = mid + 1
                        else:
                            hi = mid - 1
                    
                    if best_k2 != -1:
                        if frontier[best_k2][1] >= new_len:
                            dominated = True
                    
                    if not dominated:
                        # Add to frontier and remove dominated entries
                        ins_pos = 0
                        for k in range(len(frontier)):
                            if frontier[k][0] <= gap:
                                ins_pos = k + 1
                            else:
                                break
                        while ins_pos < len(frontier) and frontier[ins_pos][1] <= new_len:
                            ins_pos += 1
                        frontier.insert(ins_pos, (gap, new_len, new_pi, new_pei))
        
        dp[i] = frontier

    # Find best
    best_len = 1
    best_i = 0
    best_k = -1
    for i in range(n):
        for k, (g, l, pi, pei) in enumerate(dp[i]):
            if l > best_len:
                best_len = l
                best_i = i
                best_k = k

    # Reconstruct
    if best_len <= 1:
        return [primes[0]]

    result = []
    cur_i = best_i
    cur_k = best_k
    while True:
        result.append(primes[cur_i])
        if cur_k == -1:
            break
        g, l, pi, pei = dp[cur_i][cur_k]
        cur_i = pi
        cur_k = pei
    
    result.reverse()
    return result
