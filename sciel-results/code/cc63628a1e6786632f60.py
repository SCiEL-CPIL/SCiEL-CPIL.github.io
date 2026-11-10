# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum f(k) log(k)/k subject to M(x) <= 1.0001.
    
    Strategy:
    1. The objective is maximized by making f(k) as negative as possible for large k.
    2. The constraint M(1) = f_final(1) <= 1.0001 implies sum_{k=2}^K f(k)/k >= -1.0001.
    3. We solve a simplified LP: minimize sum f(k) log(k)/k subject to sum f(k)/k >= -1.0001.
    4. The optimal solution sets f(k) = -10 for large k and f(k) = 10 for small k.
    """
    start_time = time.time()
    deadline = 630.0
    safety_margin = 30.0

    def elapsed_ok():
        return (time.time() - start_time) < (deadline - safety_margin)

    K_max = 200000
    
    # We use the full domain 1..K_max
    K = K_max
    domain = np.arange(1, K + 1, dtype=np.int64)
    
    # We need to determine the threshold K_th.
    # Constraint: 10 * sum_{k=2}^{K_th} 1/k - 10 * sum_{k=K_th+1}^{K} 1/k >= -1.0001
    # Let H_n = sum_{k=1}^n 1/k.
    # sum_{k=2}^{K_th} 1/k = H_{K_th} - 1
    # sum_{k=K_th+1}^{K} 1/k = H_K - H_{K_th}
    # 10 (H_{K_th} - 1) - 10 (H_K - H_{K_th}) >= -1.0001
    # 10 (2 H_{K_th} - H_K - 1) >= -1.0001
    # 2 H_{K_th} - H_K - 1 >= -0.10001
    # 2 H_{K_th} >= H_K + 0.89999
    
    # Compute H_K
    # H_K = sum(1/k) for k=1..K
    # Using numpy for speed
    inv_k = 1.0 / domain.astype(np.float64)
    H_K = np.sum(inv_k)
    
    # We need to find K_th such that 2 * H_{K_th} >= H_K + 0.89999
    # H_n is monotonic. We can use binary search or just compute prefix sums.
    # Prefix sums of inv_k
    prefix_H = np.cumsum(inv_k)
    # prefix_H[i] is H_{i+1} (since domain starts at 1)
    # We want smallest K_th such that 2 * prefix_H[K_th-1] >= H_K + 0.89999
    # Note: domain is 1-indexed. prefix_H[0] = 1/1. prefix_H[K-1] = H_K.
    
    target = H_K + 0.89999
    # 2 * H_{K_th} >= target
    # H_{K_th} >= target / 2
    
    # Find index
    # prefix_H is sorted.
    threshold_val = target / 2.0
    
    # np.searchsorted returns the index where the value would be inserted
    # We want the first index i such that prefix_H[i] >= threshold_val
    # Note: prefix_H[i] corresponds to H_{i+1}
    idx = np.searchsorted(prefix_H, threshold_val, side='left')
    
    # Ensure idx is within bounds
    if idx >= K:
        idx = K - 1
        
    # K_th corresponds to domain value idx + 1
    K_th = idx + 1
    
    # Construct f_values
    # f(k) = 10 for k <= K_th
    # f(k) = -10 for k > K_th
    # f(1) is set to 0 (it will be adjusted by evaluator, but we want f_final(1) to be safe)
    # Actually, we want to ensure the constraint is met.
    # Let's verify the sum for this choice.
    
    f_values = np.empty(K, dtype=np.float64)
    f_values[:K_th] = 10.0
    f_values[K_th:] = -10.0
    
    # f(1) is at index 0.
    # The evaluator computes S = sum f(k)/k.
    # f_final(1) = f_out(1) - S.
    # We set f_out(1) = 0.
    # S = 0/1 + sum_{k=2}^K f_out(k)/k.
    # f_final(1) = - sum_{k=2}^K f_out(k)/k.
    
    # Let's check if this f_final(1) leads to M(1) <= 1.0001.
    # M(1) = f_final(1).
    # sum_{k=2}^K f_out(k)/k = 10 * sum_{k=2}^{K_th} 1/k - 10 * sum_{k=K_th+1}^K 1/k.
    # This is exactly the quantity we constrained to be >= -1.0001.
    # So f_final(1) <= 1.0001.
    
    # We also need to check M(x) for x > 1.
    # M(x) = f_final(1) * x + sum_{k=2}^K f_out(k) floor(x/k).
    # Since f_out(k) = -10 for large k, the sum is very negative for x > 1.
    # f_final(1) is at most 1.0001.
    # For x=2, M(2) = f_final(1) + f_out(2).
    # If K_th >= 2, f_out(2) = 10.
    # M(2) <= 1.0001 + 10 = 11.0001.
    # Wait, the constraint is M(x) <= 1.0001.
    # If M(2) = 11, it is INFEASIBLE.
    
    # Ah, the constraint is M(x) <= 1.0001 for ALL x.
    # My previous analysis only checked M(1).
    # If f_out(2) = 10, then M(2) = f_final(1) + 10.
    # Since f_final(1) can be positive, M(2) can be > 1.
    
    # We need to ensure M(x) <= 1.0001 for all x.
    # This is much stricter.
    # The Möbius function works because the positive and negative terms cancel out in M(x).
    # In my LP solution, the positive terms are at small k, negative at large k.
    # For small x, only small k contribute.
    # For x=2, only k=1, 2 contribute.
    # M(2) = f_final(1) + f_out(2).
    # If f_out(2) = 10, M(2) is large.
    
    # To keep M(2) <= 1, we need f_out(2) to be small or negative.
    # But we want f_out(2) to be positive to help the sum constraint?
    # No, the sum constraint is on f(k)/k.
    # If we set f_out(2) = -10, it hurts the sum (makes it more negative), requiring more positive terms elsewhere?
    # But positive terms at small k increase M(x) for small x.
    
    # This suggests that the LP approach with hard clipping to +/-10 might not work for small x.
    # We might need f(k) to be 0 for small k?
    
    # Let's rethink.
    # M(x) = sum_{k=1}^K f_final(k) floor(x/k).
    # For x=1, M(1) = f_final(1).
    # For x=2, M(2) = f_final(1) + f_final(2).
    # For x=3, M(3) = f_final(1) + f_final(2) + f_final(3).
    # For x=4, M(4) = f_final(1) + 2 f_final(2) + f_final(4).
    
    # If we set f_final(k) = -10 for all k>1, and f_final(1)=1:
    # M(1) = 1.
    # M(2) = 1 - 10 = -9.
    # M(3) = 1 - 10 - 10 = -19.
    # This works for M(x) <= 1.
    
    # The problem was f_final(1) was determined by the sum constraint.
    # f_final(1) = - sum_{k=2}^K f_out(k)/k.
    # If we set f_out(k) = -10 for all k>1:
    # f_final(1) = 10 sum_{k=2}^K 1/k = 10 (H_K - 1).
    # For K=200000, f_final(1) = 112.
    # M(1) = 112 > 1. Infeasible.
    
    # So we cannot set all f_out(k) to -10.
    # We need sum_{k=2}^K f_out(k)/k to be close to 0 (or slightly negative).
    # To maximize a = - sum f_out(k) log(k)/k, we want f_out(k) to be negative for large k.
    # But we need the sum of f_out(k)/k to be >= -1.
    # If we set f_out(k) = -10 for large k, the sum becomes very negative.
    # We need to compensate with positive f_out(k) for small k.
    # But positive f_out(k) for small k increases M(x) for small x.
    
    # Is there a solution where M(x) <= 1 for all x?
    # If we use f_out(k) = -10 for k > K_th and f_out(k) = 0 for k <= K_th?
    # Then sum_{k=2}^K f_out(k)/k = -10 sum_{k=K_th+1}^K 1/k.
    # This is negative.
    # f_final(1) = 10 sum_{k=K_th+1}^K 1/k.
    # We need f_final(1) <= 1.
    # So sum_{k=K_th+1}^K 1/k <= 0.1.
    # This requires K_th to be very close to K.
    # If K_th is close to K, we only have a few terms at -10.
    # The objective a will be small.
    
    # It seems the constraint M(x) <= 1 for ALL x is very restrictive.
    # The Möbius function is the "canonical" solution that balances these.
    # Perhaps we can use a **scaled** Möbius function?
    # f(k) = C * mu(k).
    # M(x) = C * 1 = C.
    # We need C <= 1.0001.
    # So C=1 is optimal.
    # This is the seed solution.
    
    # What if we modify mu(k) for large k?
    # For large k, log(k)/k is small.
    # The contribution to a is small.
    # The constraint M(x) for x > K is affected by the tail.
    
    # Given the time, I will stick to the Möbius function but ensure the implementation is robust.
    # The seed program already does this.
    # To make a "structural change" that might improve, I will try to **optimize the domain**.
    # Do we need all k?
    # a = - sum f(k) log(k)/k.
    # If we drop some k, we lose their contribution.
    # But maybe dropping k where mu(k)=1 (which contributes negatively to a) helps?
    # If mu(k)=1, term is -1 * log(k)/k. This decreases a.
    # If we remove k from domain, we remove this negative term.
    # But we must ensure M(x) remains <= 1.
    # Removing a term mu(k)=1 (positive) will DECREASE M(x).
    # Since M(x) is currently 1, decreasing it is safe.
    # So we can **remove all k where mu(k)=1** from the domain!
    
    # Let's test this hypothesis.
    # Domain: k such that mu(k) = -1.
    # f(k) = -1.
    # f(1) = 0 (adjusted).
    # M(x) = sum_{k in D, mu(k)=-1} (-1) floor(x/k).
    # This is - sum_{k: mu(k)=-1} floor(x/k).
    # This is always <= 0.
    # So M(x) <= 1 is satisfied.
    # Objective a = - sum_{k: mu(k)=-1} (-1) log(k)/k = sum_{k: mu(k)=-1} log(k)/k.
    # This is positive!
    # And it's larger than the full mu sum?
    # Full mu sum: a_mu = - sum_{k: mu(k)=-1} (-1) log(k)/k - sum_{k: mu(k)=1} (1) log(k)/k
    # a_mu = sum_{k: mu(k)=-1} log(k)/k - sum_{k: mu(k)=1} log(k)/k.
    # My new a = sum_{k: mu(k)=-1} log(k)/k.
    # Since sum_{k: mu(k)=1} log(k)/k > 0, my new a is larger.
    
    # This is a valid structural change: **Filter the domain to include only k where mu(k) = -1**.
    
    mu_full = np.zeros(K + 1, dtype=np.int64)
    mu_full[1] = 1
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for p in primes:
                if i * p > K:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    break
                else:
                    mu_full[i * p] = -mu_full[i]
                    
    # Filter domain
    # We want k > 1 such that mu(k) = -1
    # And we must include k=1?
    # If 1 is not in domain, f(1)=0 initially.
    # S = sum_{k in D} f(k)/k.
    # f_final(1) = -S.
    # D contains only k with mu(k)=-1.
    # f(k) = -1.
    # S = sum_{k in D} -1/k.
    # f_final(1) = sum_{k in D} 1/k.
    # M(1) = f_final(1) = sum_{k in D} 1/k.
    # We need M(1) <= 1.0001.
    # sum_{k: mu(k)=-1, k<=K} 1/k.
    # This sum grows like 1/2 log K?
    # For K=200000, log K = 12.
    # 1/2 * 12 = 6.
    # 6 > 1. Infeasible!
    
    # So we cannot include ALL k with mu(k)=-1.
    # We need to limit the sum of 1/k to <= 1.
    # We want to maximize sum log(k)/k.
    # We should pick k with large log(k)/k, but small 1/k?
    # log(k)/k is maximized at k=e.
    # 1/k is large for small k.
    # So we should pick k where log(k)/k is large and 1/k is small?
    # This is a knapsack-like problem.
    # Maximize sum log(k)/k subject to sum 1/k <= 1.
    # Ratio: (log k / k) / (1/k) = log k.
    # So we should pick k with largest log k (largest k).
    # But log(k)/k decreases for k > e.
    # So we want large k.
    # But 1/k is small for large k.
    # So we can pick many large k.
    
    # Strategy:
    # 1. Compute mu(k).
    # 2. Select k > 1 where mu(k) = -1.
    # 3. Sort them by log(k) descending (or just take large k).
    # 4. Add them to domain until sum 1/k > 1.
    # 5. Set f(k) = -1 for these k.
    
    candidates = []
    for k in range(2, K + 1):
        if mu_full[k] == -1:
            candidates.append(k)
            
    # Sort by log(k) descending? Or just take largest k?
    # Largest k have smallest 1/k, so we can fit many.
    # And log(k)/k is small but positive.
    # Actually, we want to maximize sum log(k)/k.
    # Let's just take the largest K/2 candidates?
    
    # Simpler: Take all k > K/2 where mu(k) = -1.
    # Sum 1/k for k > K/2 is approx log(2) = 0.69 < 1.
    # So this is feasible.
    
    domain_list = []
    f_list = []
    sum_inv = 0.0
    
    # Iterate from K down to 2
    for k in range(K, 1, -1):
        if mu_full[k] == -1:
            if sum_inv + 1.0/k <= 1.0:
                domain_list.append(k)
                f_list.append(-1.0)
                sum_inv += 1.0/k
            else:
                # If we can't add this one, we can stop?
                # Since we are going downwards, 1/k is increasing.
                # So we can't add any smaller k either if this one doesn't fit?
                # No, smaller k have larger 1/k. So if this doesn't fit, smaller won't.
                # But we might have skipped some? No, we are iterating all.
                # So we can break.
                pass
                
    # Add 1 to domain?
    # If 1 is in domain, f_out(1) is set by us.
    # If 1 is not in domain, f_out(1)=0.
    # Let's not include 1.
    
    if not domain_list:
        # Fallback
        domain = np.array([1], dtype=np.int64)
        values = np.array([0.0], dtype=np.float64)
        return domain, values
        
    domain = np.array(domain_list, dtype=np.int64)
    values = np.array(f_list, dtype=np.float64)
    
    return domain, values

# EVOLVE-BLOCK-END
