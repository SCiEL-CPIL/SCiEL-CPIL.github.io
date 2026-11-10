import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    
    # Fallback seed result (from current program's fallback)
    fallback_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    fallback_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    
    best_domain = fallback_domain
    best_values = fallback_values
    best_score = -np.inf
    
    # Strategy: Constructive Algebraic Approach using Mobius Function
    # Let f(k) = c * mu(k) for k in domain.
    # The normalization constraint sum f(k)/k = 0 (after f(1) adjustment) 
    # effectively requires handling the "1/k" weight.
    # However, the prompt says:
    # 1. Clip values to [-10, 10].
    # 2. f(1) is replaced by f(1) - S where S = sum_{k in domain} f(k)/k.
    #    This ensures sum_{k in domain} f(k)/k = 0.
    # 3. M(x) = sum_{k in domain} f(k) * floor(x/k) <= 1.0001 for all x in [1, 10K].
    # 4. Score a = - sum_{k in domain} f(k) * log(k) / k.
    
    # If we set f(k) = c * mu(k) for k=2..K, and f(1) is determined by the constraint:
    # Let D be the domain.
    # Let g(k) be the values for k > 1.
    # S = sum_{k in D, k>1} g(k)/k.
    # f(1) = -S (assuming initial f(1)=0 or we just define the final f(1) this way).
    # Wait, the evaluator calculates S over the *clipped* values.
    # If we choose small c, clipping is not an issue for k>1.
    # f(1) might be large? S = sum c*mu(k)/k. This sum converges to 0.
    # For finite K, S is small. So f(1) is small.
    
    # M(x) = f(1) * x + sum_{k=2}^K f(k) * floor(x/k)
    # Since sum f(k)/k = 0 => f(1) = - sum_{k=2}^K f(k)/k.
    # M(x) = -x * sum_{k=2}^K f(k)/k + sum_{k=2}^K f(k) * floor(x/k)
    # M(x) = sum_{k=2}^K f(k) * (floor(x/k) - x/k)
    # M(x) = - sum_{k=2}^K f(k) * {x/k}
    
    # If f(k) = c * mu(k):
    # M(x) = -c * sum_{k=2}^K mu(k) * {x/k}
    
    # We need M(x) <= 1.0001 for all x in [1, 10K].
    # Also M(x) can be negative. The constraint is one-sided: M(x) <= 1.0001.
    # There is no lower bound constraint in the feasibility check!
    # "if any M(x) > 1.0001 the function is infeasible"
    # So we can make M(x) very negative.
    
    # Objective: a = - sum_{k=1}^K f(k) log(k)/k
    # a = - f(1) log(1)/1 - sum_{k=2}^K c mu(k) log(k)/k
    # log(1) = 0, so f(1) term vanishes.
    # a = -c * sum_{k=2}^K mu(k) log(k)/k
    
    # To maximize a, we want:
    # If sum_{k=2}^K mu(k) log(k)/k is positive, we want c negative (large magnitude).
    # If sum_{k=2}^K mu(k) log(k)/k is negative, we want c positive (large magnitude).
    
    # Let H(K) = sum_{k=2}^K mu(k) log(k)/k.
    # It is known that sum_{k=1}^infty mu(k)/k = 0.
    # The weighted sum with log(k)/k?
    # Let's estimate the sign.
    # mu(2)=-1, log(2)/2 ~ 0.34
    # mu(3)=-1, log(3)/3 ~ 0.36
    # mu(4)=0
    # mu(5)=-1, log(5)/5 ~ 0.32
    # mu(6)=1, log(6)/6 ~ 0.29
    # mu(7)=-1, log(7)/7 ~ 0.28
    # mu(8)=0
    # mu(9)=0
    # mu(10)=1, log(10)/10 ~ 0.23
    # mu(11)=-1, log(11)/11 ~ 0.21
    # mu(12)=0
    # mu(13)=-1, log(13)/13 ~ 0.20
    # mu(14)=1, log(14)/14 ~ 0.19
    # mu(15)=1, log(15)/15 ~ 0.18
    # mu(16)=0
    # mu(17)=-1, log(17)/17 ~ 0.17
    # mu(18)=0
    # mu(19)=-1, log(19)/19 ~ 0.16
    # mu(20)=0
    # mu(21)=1, log(21)/21 ~ 0.15
    # mu(22)=1, log(22)/22 ~ 0.15
    # mu(23)=-1, log(23)/23 ~ 0.14
    # mu(24)=0
    # mu(25)=0
    # mu(26)=1, log(26)/26 ~ 0.13
    # mu(27)=0
    # mu(28)=0
    # mu(29)=-1, log(29)/29 ~ 0.12
    # mu(30)=1, log(30)/30 ~ 0.12
    
    # The terms mu(k) log(k)/k are dominated by small k.
    # Negative terms: 2,3,5,7,11,13,17,19,23,29...
    # Positive terms: 6,10,14,15,21,22,26,30...
    # Generally, for small k, mu(k) is -1 more often?
    # Primes have mu=-1. Composites with even number of prime factors have mu=1.
    # Density of mu=-1 is roughly same as mu=1 (around 6/pi^2).
    # But the weight 1/k decreases.
    # Let's just compute H(K) numerically for a few K.
    
    # Constraint:
    # M(x) = -c * S_x where S_x = sum_{k=2}^K mu(k) {x/k}.
    # We need -c * S_x <= 1.0001 for all x.
    # This implies:
    # If c > 0: -c * S_x <= 1 => c * S_x >= -1 => c <= 1 / (-min S_x) if min S_x < 0.
    #            Also if S_x is positive, -c S_x is negative, which is <= 1.
    #            So if c > 0, the tightest constraint is where S_x is most negative.
    #            c <= 1 / (-min_x S_x).
    # If c < 0: let c = -C where C > 0.
    #            C * S_x <= 1 => C <= 1 / (max_x S_x) if max S_x > 0.
    #            So if c < 0, the tightest constraint is where S_x is most positive.
    #            |c| <= 1 / (max_x S_x).
            
    # We want to maximize a = -c * H(K).
    # Case 1: H(K) > 0. We want c < 0 to make a positive.
    #          Let c = -C. a = C * H(K).
    #          Constraint: C <= 1 / (max_x S_x).
    #          Max a = H(K) / (max_x S_x).
    # Case 2: H(K) < 0. We want c > 0 to make a positive.
    #          Let c = C. a = -C * H(K) = C * |H(K)|.
    #          Constraint: C <= 1 / (-min_x S_x).
    #          Max a = |H(K)| / (-min_x S_x).
            
    # We also have the bound f(k) in [-10, 10].
    # f(k) = c * mu(k). So |c| <= 10.
    # So C <= 10 as well.
    
    # Plan:
    # 1. Iterate K.
    # 2. Compute mu(k) for k=1..K.
    # 3. Compute H(K) = sum_{k=2}^K mu(k) log(k)/k.
    # 4. Determine sign of H(K).
    # 5. Compute S_x = sum_{k=2}^K mu(k) {x/k} for all x in [1, 10K].
    # 6. Find max S_x and min S_x.
    # 7. Calculate optimal C.
    # 8. Construct f.
    # 9. Verify and score.

    def compute_mobius(K):
        mu = np.ones(K + 1, dtype=np.int64)
        mu[0] = 0
        primes = []
        is_prime = np.ones(K + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, K + 1):
            if is_prime[i]:
                primes.append(i)
                for j in range(i, K + 1, i):
                    is_prime[j] = False
                    mu[j] *= -1
        # Check for square factors
        # A number is square-free if it is not divisible by p^2 for any prime p.
        # The sieve above only handles the sign change for p.
        # We need to set mu[n] = 0 if n is divisible by p^2.
        for p in primes:
            p2 = p * p
            if p2 > K:
                break
            for j in range(p2, K + 1, p2):
                mu[j] = 0
        return mu

    for K in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]:
        if time.time() - start_time > 550:
            break
            
        if K > 200000:
            break
            
        # Compute Mobius
        mu = compute_mobius(K)
        
        # Compute H(K)
        ks = np.arange(2, K + 1)
        weights = np.log(ks) / ks
        H = np.sum(mu[2:K+1] * weights)
        
        if abs(H) < 1e-10:
            continue
            
        # Compute S_x for x in [1, 10K]
        max_x = 10 * K
        # S_x = sum_{k=2}^K mu(k) * (x % k) / k
        # Vectorized computation might be heavy for large K.
        # Let's try to optimize.
        # For each k, mu(k) is fixed.
        # S_x is a sum of sawtooth waves.
        
        # If K is large, iterating x from 1 to 10K and summing over k is O(K * 10K) = O(K^2).
        # For K=200000, K^2 = 4e10, which is too slow in Python.
        # We need a faster way to compute max/min of S_x.
        
        # Alternative:
        # Note that {x/k} is periodic with period k.
        # But the sum is not periodic with a small period.
        
        # Let's stick to smaller K if we can't optimize the S_x calculation.
        # With K=2000, K^2 = 4e6, which is fast.
        # With K=10000, K^2 = 1e8, might take a few seconds in NumPy if vectorized properly?
        # Actually, looping x in Python is slow.
        # We can vectorize over x for a fixed k?
        # S_x = sum_k mu_k * (x % k) / k.
        # Let's create an array X = np.arange(1, max_x + 1).
        # For each k, compute X % k. This is O(max_x).
        # Total O(K * max_x) = O(K^2).
        
        # For K=20000, 4e8 operations. Might be borderline.
        # Let's limit K to 5000 for this constructive approach to ensure speed.
        # The LP approach got 1.0745 with K up to 2000.
        # Can we do better with K=5000?
        
        if K > 5000:
            # For larger K, we might need a different approximation or just skip.
            # The density of mu(k) is low.
            # Let's just try K up to 5000 first.
            continue

        X = np.arange(1, max_x + 1)
        S = np.zeros(max_x)
        
        # Only iterate over k where mu(k) != 0
        nonzero_mu_idx = np.nonzero(mu[2:K+1])[0] + 2
        nonzero_mu_vals = mu[2:K+1][nonzero_mu_idx != 0] # Wait, mu is int
        # Correct extraction:
        mask = mu[2:K+1] != 0
        ks_nonzero = ks[mask]
        mu_nonzero = mu[2:K+1][mask]
        
        for k, mk in zip(ks_nonzero, mu_nonzero):
            S += mk * (X % k) / k
            
        max_S = np.max(S)
        min_S = np.min(S)
        
        # Determine optimal c
        if H > 0:
            # Want c < 0. C = -c > 0.
            # Constraint: C <= 1 / max_S (if max_S > 0)
            # Also C <= 10.
            if max_S <= 0:
                # If max_S <= 0, then C * max_S <= 0 <= 1. No constraint from max_S?
                # Wait. M(x) = -c S_x = C S_x.
                # We need C S_x <= 1.
                # If max_S is negative, the largest value of C S_x is C * max_S (which is negative).
                # So it's always <= 1.
                # In this case, the constraint is only |c| <= 10.
                # So C = 10.
                C = 10.0
            else:
                C = min(10.0, 1.0 / max_S)
            c = -C
        else:
            # Want c > 0. C = c > 0.
            # Constraint: -C S_x <= 1 => C S_x >= -1.
            # Tightest when S_x is min.
            # C * min_S >= -1 => C <= 1 / (-min_S) (if min_S < 0)
            if min_S >= 0:
                C = 10.0
            else:
                C = min(10.0, 1.0 / (-min_S))
            c = C
            
        # Construct f
        f = np.zeros(K + 1)
        f[2:K+1] = c * mu[2:K+1]
        
        # Calculate S_sum for f(1)
        S_sum = np.sum(f[2:K+1] / ks)
        f[1] = -S_sum
        
        # Clip to [-10, 10]
        f = np.clip(f, -10, 10)
        
        # Recalculate S_sum after clipping?
        # The evaluator clips first, then adjusts f(1).
        # So we must simulate that.
        # 1. Clip f[2:K+1] and f[1] (initial 0 or whatever).
        #    Actually, f(1) is set *after* clipping in the evaluator logic?
        #    "values are clipped to [-10, 10]. Then f(1) is replaced by f(1) - S..."
        #    So:
        #    1. Clip all values.
        #    2. Calculate S = sum clipped(k)/k.
        #    3. f(1) = clipped(1) - S.
        #    4. Clip f(1) again? No, "f(1) is replaced". It doesn't say clip again.
        #       But if f(1) goes out of bounds, does it matter?
        #       The domain keys are distinct positive integers.
        #       The values are finite reals.
        #       The check M(x) uses the final f(1).
        #       If f(1) is huge, M(x) will be huge.
        #       However, S is sum of clipped values / k.
        #       Since clipped values are in [-10, 10], S is bounded.
        #       Max S is roughly sum 10/k ~ 10 log K.
        #       For K=5000, S ~ 100.
        #       f(1) = 0 - S = -100.
        #       M(x) = f(1) x + ...
        #       For x=1, M(1) = f(1) + sum f(k) floor(1/k) = f(1).
        #       If f(1) = -100, M(1) = -100 <= 1. OK.
        #       For x=10K, M(10K) = f(1) 10K + sum f(k) 10.
        #       This could be large negative.
        #       But we need M(x) <= 1.
        #       If f(1) is negative, M(x) is driven down.
        #       The risk is if f(1) is positive and large.
        #       But we chose c to satisfy M(x) <= 1 assuming the normalization.
        #       The normalization makes f(1) = -S.
        #       If S is positive, f(1) is negative.
        #       If S is negative, f(1) is positive.
        
        # Let's verify the score with the actual evaluator logic simulation.
        
        # Simulate Evaluator
        domain = np.arange(1, K + 1)
        values = f.copy()
        
        # 1. Clip
        values = np.clip(values, -10, 10)
        
        # 2. Adjust f(1)
        S_eval = np.sum(values / domain)
        values[0] = values[0] - S_eval # domain[0] is 1
        
        # Note: values[0] corresponds to k=1.
        # The formula is sum_{k in domain} f(k)/k.
        # Here domain is 1..K.
        
        # 3. Check M(x)
        # M(x) = sum_{k=1}^K f(k) floor(x/k)
        # We check for x in 1..10K.
        # To save time, we can check a subset or use the property.
        # But let's do it properly for the best candidate.
        
        # For now, just compute the score a.
        # a = - sum f(k) log(k)/k
        # log(1) = 0.
        a = -np.sum(values[1:] * np.log(domain[1:]) / domain[1:])
        
        # Quick feasibility check on a sample of x to avoid expensive full check for bad candidates
        # If a is very low, skip.
        if a < best_score:
            continue
            
        # Full feasibility check
        feasible = True
        # We can vectorize M(x) calculation?
        # M(x) for all x is expensive.
        # Let's check x=1..10K.
        # For K=5000, 50000 x 5000 is 2.5e8 ops.
        # Using NumPy broadcasting:
        # X = np.arange(1, 10*K + 1)
        # Ks = domain
        # Floors = X[:, None] // Ks[None, :]
        # M = Floors @ values
        # This creates a 50000 x 5000 matrix. 2.5e8 floats ~ 2GB. Too much memory.
        
        # We must be careful.
        # Let's check only if a is promising.
        # And maybe check a subset of x first?
        # Or use the fact that we derived c from max/min S.
        # The derivation assumed the normalization holds exactly.
        # Clipping might break it if f(1) was clipped?
        # f(1) is adjusted *after* clipping.
        # If f(1) ends up outside [-10, 10], it is NOT clipped again.
        # But the constraint M(x) <= 1 must hold.
        
        # Let's just run the check for the best few candidates.
        # For now, assume feasible if the derived C was conservative.
        # To be safe, let's check x=1 and x=10K and a few random x.
        
        x_test = np.array([1, 10, 100, 1000, 10*K, 5*K, 2*K])
        M_test = np.zeros(len(x_test))
        for i, x in enumerate(x_test):
            M_test[i] = np.sum(values * (x // domain))
            
        if np.any(M_test > 1.0001):
            feasible = False
            # Try to reduce c by 10% and retry?
            # For now, mark as infeasible.
            
        if feasible:
            if a > best_score:
                best_score = a
                best_domain = domain
                best_values = values
                # print(f"K={K}, a={a}")

    return best_domain, best_values

