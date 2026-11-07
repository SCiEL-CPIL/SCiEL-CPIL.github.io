import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140  # 10s margin from 150s limit

    # Strategy:
    # 1. We know k must divide M.
    # 2. If we fix a set of primes P, and generate M as a product of powers of primes in P,
    #    then any divisor k of M also only has prime factors from P.
    # 3. We can enumerate divisors k of M and check if k * sigma(k) == M.
    # 4. To find M with many solutions, we should look at M with many divisors.
    #    But more importantly, the structure of M and k matters.
    #    
    #    Let's use the "targeted collision" idea from inspiration 1.
    #    Instead of scanning k, we can construct M such that it has many k.
    #    
    #    Observation: If M = prod p_i^{a_i}, then k = prod p_i^{b_i} is a candidate
    #    if sum_{i} p_i^{b_i+1}/(p_i-1) * (M/k) ... wait, sigma(k) = prod (p_i^{b_i+1}-1)/(p_i-1).
    #    We need k * sigma(k) = M.
    #    
    #    Let's try a different approach:
    #    Generate candidate M's that are highly composite with small primes.
    #    For each M, enumerate divisors k and check the condition.
    #    
    #    To be efficient, we need a fast way to generate M's with many divisors.
    #    We can use a priority queue to generate the next smallest highly composite number?
    #    Or just iterate over exponent vectors for the first few primes.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    best_M = 333312
    best_k_list = [336, 372, 434]
    best_count = 3

    # Helper to compute sigma from factorization
    def sigma_from_factors(factors):
        # factors: list of (p, a)
        s = 1
        for p, a in factors:
            s *= (p**(a+1) - 1) // (p - 1)
        return s

    # Helper to get all divisors from factorization
    def get_divisors(factors):
        divs = [1]
        for p, a in factors:
            new_divs = []
            p_pow = 1
            for i in range(a + 1):
                for d in divs:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divs = new_divs
        return divs

    # We will search over M's formed by the first N primes.
    # Let's try N=6 to N=10.
    # For each N, we generate all exponent combinations such that M <= 10^18.
    # This is a multi-dimensional grid search.
    
    # To avoid generating too many M's, we can prune.
    # But let's try a random sampling approach first to see if we find better M's.
    # Or a systematic search for small N.
    
    # Let's try systematic search for N=4, 5, 6.
    # N=4: 2, 3, 5, 7. Max exponents: 2^60, 3^37, 5^25, 7^21.
    # This is too many combinations (60*37*25*21 ~ 1.1M).
    # But we can prune based on M <= 10^18.
    
    # Let's write a recursive generator for M's.
    
    def generate_ms(n_primes, max_M=10**18):
        # n_primes: number of primes to use
        # Returns list of (M, factors)
        results = []
        
        def recurse(idx, current_M, factors):
            if time.time() > deadline:
                return
            if idx == n_primes:
                if current_M > 1 and current_M <= max_M:
                    results.append((current_M, factors))
                return
            
            p = primes[idx]
            # Max exponent for this prime
            # current_M * p^a <= max_M
            # a <= log_p(max_M / current_M)
            max_exp = 0
            while current_M * (p ** (max_exp + 1)) <= max_M:
                max_exp += 1
                if max_exp > 100: break # safety
            
            for a in range(max_exp + 1):
                new_M = current_M * (p ** a)
                if new_M > max_M:
                    break
                new_factors = factors + [(p, a)]
                recurse(idx + 1, new_M, new_factors)
        
        recurse(0, 1, [])
        return results

    # This might generate too many M's for N=6.
    # Let's try N=4 first.
    
    for n_p in [4, 5, 6]:
        if time.time() > deadline - 20:
            break
            
        # Generate M's
        ms = generate_ms(n_p)
        
        # For each M, check divisors
        # To speed up, we can skip M's with too few divisors?
        # Or just check all.
        # Number of M's for N=4 is ~1M. Checking divisors for 1M M's might be slow.
        # Let's filter: only check M's that are "interesting".
        # What makes an M interesting?
        # Maybe M's that are divisible by many small primes?
        
        # Let's just iterate and check.
        # But we need to be fast.
        # Let's limit the number of M's we check per n_p.
        # Or check only M's that are multiples of a certain base?
        
        # Actually, the previous best M=333312 = 2^9 * 3 * 7 * 31.
        # It used prime 31, which is the 11th prime.
        # So N=4 (primes 2,3,5,7) won't find it.
        # We need to include larger primes.
        
        # Let's change strategy:
        # Instead of fixing the set of primes, let's pick random sets of primes.
        # Or, let's pick M's that are products of random small primes.
        
        pass

    # Let's try a different approach:
    # Pick a target number of solutions T (e.g., 4, 5, ...).
    # Try to find M with T solutions.
    # This is hard.
    
    # Let's go back to the k-scan but optimize it.
    # The bottleneck was the sieve and sigma calculation.
    # Let's use a faster sigma calculation.
    # We can precompute sigma for all k up to N using a sieve-like method.
    
    # sigma(n) = sum of divisors.
    # We can compute it in O(N log N) time.
    # for i in 1..N:
    #   for j in i..N step i:
    #     sigma[j] += i
    
    # For N=10^7, this is 10^7 * (1 + 1/2 + 1/3 + ...) ~ 10^7 * log(10^7) ~ 1.6 * 10^8 operations.
    # In Python, this might take ~10-20s.
    # Let's try N=5,000,000.
    
    N = 5_000_000
    sigma = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
            if time.time() > deadline - 10:
                # If we run out of time, we can break and use partial results?
                # But sigma needs to be complete for correctness.
                # Let's just break and return best so far.
                pass
    
    # If we broke early, sigma is incomplete.
    # Let's check if we finished.
    # Actually, the inner loop is fast. 5M * H(5M) ~ 5M * 15 = 75M iterations.
    # Python can do ~10M-50M simple iterations per second.
    # So 75M might take 1.5s - 7.5s. This is feasible.
    
    # Now, compute M = k * sigma(k) for k in 1..N.
    m_to_k_list = defaultdict(list)
    
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M <= 10**18:
            m_to_k_list[M].append(k)
    
    # Find best M
    for M, k_list in m_to_k_list.items():
        if len(k_list) > best_count:
            best_count = len(k_list)
            best_M = M
            best_k_list = k_list

    # If we have time, try larger N?
    # But N=5M is already a big jump from 2M.
    # Let's see if we can go to 10M.
    # 10M * H(10M) ~ 10M * 16 = 160M iterations. Might take 3-15s.
    # Total time budget is 140s.
    # We spent ~10s on sieve.
    # We have ~130s left.
    # Let's try N=10M if time permits.
    
    if time.time() < deadline - 30:
        N2 = 10_000_000
        # Extend sigma?
        # We can't easily extend the sigma array computed by the nested loop method
        # without recomputing or using a different method.
        # Let's just re-run with N2? No, too slow.
        # Let's stick with N=5M result.
        
    # Ensure types
    return {'M': int(best_M), 'k': best_k_list}

if __name__ == "__main__":
    construct()
