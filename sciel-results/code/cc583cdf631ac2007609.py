import time
import math

def construct():
    start_time = time.time()
    timeout = 140  # Leave 10s safety margin before 150s limit
    
    best_M = 333312
    best_k_list = [336, 372, 434]
    best_count = 3
    
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    def search_smooth(primes, max_k, deadline):
        n = len(primes)
        # Precompute powers and sigma values
        powers = []
        sig_powers = []
        for p in primes:
            ps = [1]
            v = 1
            while v <= max_k:
                v *= p
                if v <= max_k:
                    ps.append(v)
            powers.append(ps)
            sps = [1]
            for j in range(1, len(ps)):
                sps.append((pow(p, j + 1) - 1) // (p - 1))
            sig_powers.append(sps)
        
        map_M_to_ks = {}
        
        def dfs(i, k, sig):
            if time.time() > deadline:
                return
            if i == n:
                M = k * sig
                if M <= 10**18:
                    if M in map_M_to_ks:
                        map_M_to_ks[M].append(k)
                    else:
                        map_M_to_ks[M] = [k]
                return
            for j in range(len(powers[i])):
                new_k = k * powers[i][j]
                if new_k > max_k:
                    break
                dfs(i + 1, new_k, sig * sig_powers[i][j])
        
        dfs(0, 1, 1)
        
        if not map_M_to_ks:
            return 0, [], 0
        bm, bk, bc = 0, [], 0
        for M, ks in map_M_to_ks.items():
            if len(ks) > bc:
                bc, bm, bk = len(ks), M, ks
        return bm, bk, bc
    
    # Try configurations with increasing prime sets and appropriate limits
    configs = [
        (3, 10**10),
        (4, 10**10),
        (5, 10**10),
        (6, 10**9),
        (7, 10**8),
    ]
    
    for np, lim in configs:
        if time.time() > timeout - 5:
            break
        M_f, ks_f, c_f = search_smooth(base_primes[:np], lim, timeout)
        if c_f > best_count:
            best_count, best_M, best_k_list = c_f, M_f, ks_f
    
    if best_count < 1:
        return {'M': 1, 'k': [1]}
    return {'M': int(best_M), 'k': list(best_k_list)}
