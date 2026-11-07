import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 110  # 10s margin
    
    # 1. Determine N (max k to check)
    # We want to maximize N such that O(N) sieve + O(N) sigma calc + O(N) hash check fits in time.
    # Python is slow. Let's start with N = 5,000,000 and see if we can go higher.
    # Actually, let's try to be aggressive: N = 10,000,000.
    # If it's too slow, we can reduce N. But we can't easily adapt N mid-run without wasting time.
    # Let's estimate: Sieve is ~0.5s for 10M. Sigma DP is ~0.5s. Hash check is ~0.5s. Total ~1.5s.
    # So we can go much higher. Let's try N = 50,000,000.
    # 50M iterations in Python might take 5-10 seconds.
    # Let's set N = 20,000,000 as a safe bet for 120s limit, allowing for overhead.
    
    N = 20_000_000
    
    # 2. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 3. Compute sigma(k) for all k using DP
    # sigma[k] stores sigma(k)
    # p_pow[k] stores p^a where p is the smallest prime factor of k and a is its exponent
    # Actually, we just need sigma[k].
    # Recurrence:
    # Let p = spf[k], m = k // p.
    # If m % p == 0:
    #   Let m' = m // p.
    #   sigma[k] = sigma[m] + p^(a+1) ... this is tricky to track without storing exponents.
    # Alternative recurrence:
    # sigma[k] = sigma[m] * (1 + p) if p not in m.
    # sigma[k] = sigma[m] + p^a * sigma[m'] ... no.
    
    # Let's use the formula: sigma(k) = sigma(m) * (p^(a+1)-1)/(p-1) / ((p^a-1)/(p-1)) ? No.
    # sigma(k) = sigma(p^a * r) = (1 + p + ... + p^a) * sigma(r).
    # If m = p^a * r, then sigma(m) = (1 + ... + p^a) sigma(r).
    # k = p^(a+1) * r.
    # sigma(k) = (1 + ... + p^(a+1)) sigma(r) = sigma(m) + p^(a+1) sigma(r).
    # We need sigma(r) and p^(a+1).
    # This requires storing more state.
    
    # Simpler: Precompute sigma for all k using a different sieve?
    # The O(N log N) sieve is:
    # sigma = [1] * (N + 1)
    # for i in range(1, N + 1):
    #   for j in range(i, N + 1, i):
    #     sigma[j] += i
    # This is too slow in pure Python for N=20M (20M * log(20M) ~ 300M ops).
    
    # Let's stick to the SPF DP but optimize the state.
    # We store:
    # sigma[k]
    # p_pow[k]: the highest power of spf[k] dividing k.
    # p_exp[k]: the exponent of spf[k] in k.
    
    sigma = [0] * (N + 1)
    p_pow = [0] * (N + 1)
    p_exp = [0] * (N + 1)
    
    sigma[1] = 1
    p_pow[1] = 1
    p_exp[1] = 0
    
    for k in range(2, N + 1):
        p = spf[k]
        m = k // p
        if m % p == 0:
            # p divides m
            # p_pow[k] = p_pow[m] * p
            # p_exp[k] = p_exp[m] + 1
            # sigma[k] = sigma[m] + p_pow[k] * (sigma[m] // p_pow[m]) ? No.
            # sigma(k) = sigma(p^(a+1) * r) = sigma(p^a * r) + p^(a+1) * sigma(r)
            # sigma[m] = sigma(p^a * r)
            # sigma(r) = sigma[m] // ((1 + ... + p^a)) ... division is risky.
            # Let's store sigma_r[k] = sigma(r) where k = p^a * r, p not in r.
            # Then sigma[k] = sigma_r[k] * (1 + p + ... + p^a).
            # If m % p == 0:
            #   r is the same as r for m.
            #   sigma_r[k] = sigma_r[m]
            #   p_pow[k] = p_pow[m] * p
            #   p_exp[k] = p_exp[m] + 1
            #   sigma[k] = sigma_r[k] * (p_pow[k] * (p - 1) + 1) // (p - 1)
            #   Wait, 1 + ... + p^a = (p^(a+1)-1)/(p-1).
            #   Let S_a = 1 + ... + p^a.
            #   sigma[m] = sigma_r * S_a.
            #   sigma[k] = sigma_r * S_{a+1} = sigma_r * (S_a + p^(a+1)).
            #   sigma[k] = sigma[m] + sigma_r * p^(a+1).
            #   We need sigma_r.
            
            sigma_r_k = sigma_r[m] # We need to store sigma_r
            p_pow_k = p_pow[m] * p
            p_exp_k = p_exp[m] + 1
            sigma_k = sigma[m] + sigma_r_k * p_pow_k
            # Update storage
            sigma[k] = sigma_k
            p_pow[k] = p_pow_k
            p_exp[k] = p_exp_k
            sigma_r[k] = sigma_r_k
        else:
            # p does not divide m
            # r = m
            # p_pow[k] = p
            # p_exp[k] = 1
            # sigma_r[k] = sigma[m]
            # sigma[k] = sigma[m] * (1 + p)
            sigma_r_k = sigma[m]
            p_pow_k = p
            p_exp_k = 1
            sigma_k = sigma[m] * (1 + p)
            sigma[k] = sigma_k
            p_pow[k] = p_pow_k
            p_exp[k] = p_exp_k
            sigma_r[k] = sigma_r_k

    # We forgot to initialize sigma_r array.
    # Let's re-implement this block carefully.
    
    # Re-do Sigma DP
    sigma = [0] * (N + 1)
    sigma_r = [0] * (N + 1)
    p_pow = [0] * (N + 1)
    
    sigma[1] = 1
    sigma_r[1] = 1
    p_pow[1] = 1
    
    for k in range(2, N + 1):
        p = spf[k]
        m = k // p
        if m % p == 0:
            # k = p^(a+1) * r, m = p^a * r
            # sigma_r[k] = sigma_r[m]
            # p_pow[k] = p_pow[m] * p
            # sigma[k] = sigma[m] + sigma_r[m] * p_pow[k]
            sigma_r[k] = sigma_r[m]
            p_pow[k] = p_pow[m] * p
            sigma[k] = sigma[m] + sigma_r[m] * p_pow[k]
        else:
            # k = p * m, m = r (p not in r)
            # sigma_r[k] = sigma[m]
            # p_pow[k] = p
            # sigma[k] = sigma[m] * (1 + p)
            sigma_r[k] = sigma[m]
            p_pow[k] = p
            sigma[k] = sigma[m] * (1 + p)
            
    # 4. Find M with max collisions
    freq = defaultdict(int)
    k_lists = defaultdict(list)
    best_M = None
    best_count = 0
    
    for k in range(1, N + 1):
        if time.time() > deadline:
            break
        M = k * sigma[k]
        if M > 10**18:
            continue
        freq[M] += 1
        if freq[M] >= 2:
            k_lists[M].append(k)
        if freq[M] > best_count:
            best_count = freq[M]
            best_M = M
            best_k_list = k_lists[M]
            
    if best_count < 3:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    k_result = [int(k) for k in best_k_list]
    M_result = int(best_M)
    
    return {'M': M_result, 'k': k_result}
