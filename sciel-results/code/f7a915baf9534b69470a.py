import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def time_left():
        return deadline - time.time()

    # Primes 1 mod 4 for M
    p_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 599, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 713, 733, 757, 761, 769, 773, 797, 809, 811, 821, 829, 853, 857, 877, 881, 929, 937, 941, 953, 977, 991, 997]

    # Primes for Q (primorial base)
    primes_all = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]

    best_N = 0
    best_pairs = []

    # Prioritize k values that result in M_max in the range [10^10, 10^14]
    # k=4: Q ~ 4.4e4, M_max ~ 2e13
    # k=5: Q ~ 5.3e6, M_max ~ 2e11
    # k=6: Q ~ 9e8, M_max ~ 1e9
    # k=3: Q ~ 3600, M_max ~ 2e14
    # k=7: Q ~ 1.2e11, M_max ~ 8e6
    # k=2: Q ~ 36, M_max ~ 2e16 (Slow enumeration)
    # k=1: Q ~ 4, M_max ~ 2e17 (Very slow enumeration)
    
    k_order = [4, 5, 6, 3, 7, 8, 2, 1, 9, 10, 11, 12]

    for k in k_order:
        if time_left() < 5:
            break
            
        S = 1
        for i in range(k):
            S *= primes_all[i]
        Q = S * S
        
        M_max = 10**18 // Q
        if M_max < 2:
            continue
        
        # Find best M <= M_max using DFS
        # M = prod p_i^e_i, p_i in p_1mod4
        # Maximize prod (e_i + 1)
        
        best_M = 1
        best_score = 0
        
        # Use more primes in search
        search_primes = p_1mod4[:40]
        
        def dfs(idx, current_M, current_score, prev_exp):
            nonlocal best_M, best_score
            
            if time_left() < 2:
                return
                
            if current_score > best_score:
                best_score = current_score
                best_M = current_M
                
            if idx == len(search_primes):
                return
            
            p = search_primes[idx]
            
            val = 1
            e = 0
            while current_M * val <= M_max and e <= prev_exp:
                next_M = current_M * val
                next_score = current_score * (e + 1)
                dfs(idx + 1, next_M, next_score, e)
                val *= p
                e += 1
                if val > M_max // max(current_M, 1):
                    break

        dfs(0, 1, 1, 200)
        
        M = best_M
        N = Q * M
        
        if N < 2 or N > 10**18:
            continue
        
        # Enumerate pairs: x^2 + y^2 = M
        limit_x = int(math.isqrt(M // 2)) + 1
        
        # Estimate time: ~0.1us per iteration in Python, so 10^7 ~ 1s
        # Allow up to 2*10^7 iterations (~2s) to be safe
        max_iters = min(limit_x, 2 * 10**7)
        
        if limit_x > max_iters:
            # If too large, we might still try if we have lots of time
            if time_left() < 10:
                continue
        
        pairs = []
        M_val = M
        Q_val = Q
        isqrt = math.isqrt
        
        for x in range(1, max_iters):
            x2 = x * x
            y2 = M_val - x2
            y = isqrt(y2)
            if y * y == y2 and x <= y:
                a = Q_val * x2
                b = Q_val * y2
                pairs.append([a, b])
            
            if x % 1000000 == 0 and time_left() < 5:
                break

        if len(pairs) > len(best_pairs):
            best_pairs = pairs
            best_N = N

    # Fallback
    if not best_pairs:
        best_N = 5
        best_pairs = [[1, 4]]
        
    return {'N': best_N, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
