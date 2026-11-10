# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy:
    1. Define the target function g(k) = log(k)/k for k in [1, K].
    2. We want to find f(k) such that the step function M(x) = sum_k f(k) floor(x/k) <= 1.
    3. In the dual LP, we minimize sum_x y(x) subject to sum_{x=k}^{10K} y(x) >= log(k)/k.
    4. The optimal solution for the dual is y(x) = max(0, g(x) - g(x+1)) roughly, 
       but since g(k) is not monotonic, we need to be careful.
       
       Actually, let's look at the primal:
       Maximize a = sum_k f(k) log(k)/k
       Subject to:
         sum_k f(k) floor(x/k) <= 1 for all x in [1, 10K]
         f(k) can be any real.
         
       Wait, the constraint is M(x) <= 1.
       M(x) = sum_k f(k) floor(x/k).
       
       If we choose f(k) such that M(x) is exactly 1 for all x, then:
       M(x+1) - M(x) = sum_k f(k) (floor((x+1)/k) - floor(x/k)).
       floor((x+1)/k) - floor(x/k) is 1 if k divides x+1, else 0.
       So M(x+1) - M(x) = f(x+1).
       If M(x) = 1 for all x, then f(x+1) = 0 for all x. This implies f(k)=0 for k>=2.
       Then M(1) = f(1) = 1.
       Objective a = -f(1)log(1)/1 = 0. This is not good.
       
       The objective in the prompt is a = -sum_k f(k) * log(k) / k.
       Wait, the prompt says: "metric is a = -sum_k f(k) * log(k) / k".
       And the seed uses mu(k) where a -> 1.
       For mu(k), sum mu(k) floor(x/k) = 1 if x>=1? No, sum_{k|x} mu(k) = 1 if x=1 else 0.
       M(x) = sum_{k<=x} mu(k) floor(x/k) = sum_{d|x} sum_{k|d} mu(k) ... no.
       M(x) = sum_{k=1}^x mu(k) floor(x/k).
       It is a known identity that sum_{k=1}^x mu(k) floor(x/k) = 1 for x=1, and 0 for x>1?
       Let's check x=2: mu(1)floor(2/1) + mu(2)floor(2/2) = 1*2 + (-1)*1 = 1.
       x=3: 1*3 - 1*1 - 1*1 = 1.
       x=4: 1*4 - 1*2 - 1*1 + 0*1 = 1.
       x=5: 1*5 - 1*1 - 1*1 - 1*1 + 0 = 1.
       x=6: 1*6 - 1*3 - 1*2 - 1*1 + 0*1 + 1*1 = 6-3-2-1+1 = 1.
       Yes, M(x) = 1 for all x >= 1.
       
       So the constraint M(x) <= 1 is satisfied with equality.
       The objective a = -sum_{k=1}^K mu(k) log(k)/k.
       As K -> infinity, this sum approaches 1.
       
       The problem is to maximize a.
       The dual approach suggests that if we can find f(k) such that M(x) <= 1,
       the maximum possible value of a is bounded by the dual objective.
       
       Let's try to construct f(k) directly to approximate mu(k) but with a larger K.
       The seed has K=30.
       We can compute mu(k) for k up to K_max.
       The cost of evaluation is O(m * 10K).
       If K=200000, m=200000, cost is 200000 * 2000000 = 4e11. Too slow.
       
       We need a sparse representation or a faster evaluation method?
       The evaluator computes M(x) for every integer x in [1, 10K].
       It says "cost about m * 10K".
       If we use m=200000, 10K=2,000,000. 4*10^11 operations.
       In Python, this will take forever.
       
       However, we can reduce K.
       The sum -sum mu(k) log(k)/k converges to 1.
       How fast?
       The error term is related to the Mertens function M(x).
       |sum_{k<=K} mu(k) log(k)/k - 1| ~ O(1/sqrt(K)) or something similar?
       
       Let's test K=1000.
       m=1000, 10K=10000. Cost 10^7. Fast.
       K=10000. m=10000, 10K=100000. Cost 10^9. Might be ~1s in C, but in Python?
       The evaluator is likely implemented in C/NumPy? "Python standard library, NumPy and SciPy are available."
       If the evaluator uses NumPy, 10^9 operations might be ~1-10 seconds.
       
       Let's assume the evaluator is efficient enough for K=10000.
       We can compute mu(k) for k=1..10000.
       
       Let's verify the constraint M(x) <= 1.
       For mu(k), M(x) = 1 exactly for all x.
       So it is feasible.
       
       We just need to return domain = [1..K] and values = mu(1..K).
       
       Let's try K=5000 to be safe on time.
       Cost 5000 * 50000 = 2.5e8. Very fast.
       
       We can increase K to 20000 if time permits.
       20000 * 200000 = 4e9. Might be too slow for Python evaluator.
       
       Let's start with K=10000.
       
       Algorithm:
       1. Sieve of Eratosthenes to compute mu(k) for k=1..K.
       2. domain = np.arange(1, K+1)
       3. values = mu_array
       4. Return domain, values.
       
       Deadline guard:
       Compute mu for K=10000.
       Check time. If we have more time, try K=20000?
       Actually, the score is proportional to a.
       a(K) = -sum_{k=1}^K mu(k) log(k)/k.
       a(inf) = 1.
       a(10000) is very close to 1.
       a(100) is ~0.99.
       a(10000) is ~0.9999.
       The gain from 10000 to 20000 is tiny.
       The risk of timeout is real.
       
       Let's stick to K=10000.
    """
    start_time = time.time()
    
    # Choose K based on expected evaluator speed.
    # If evaluator is O(m * 10K) in Python, 10000 * 100000 = 10^9 is risky.
    # Let's use K=5000. 5000 * 50000 = 2.5 * 10^8. Safe.
    K = 5000
    
    # Compute Mobius function mu(k) for k=1..K
    mu = np.ones(K + 1, dtype=np.int8)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
        else:
            # Not prime, but we need to handle squares of primes correctly.
            # The standard linear sieve or simple sieve works.
            # The above loop marks multiples of i.
            # If i is not prime, it's a multiple of a smaller prime.
            # We only need to process primes for the mu calculation.
            pass
            
    # Wait, the sieve above is slightly wrong for mu.
    # Standard method:
    mu = np.ones(K + 1, dtype=np.int8)
    mu[0] = 0
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
        # If i is not prime, it is handled by its prime factors.
        # But we need to ensure we only multiply by -1 for prime factors.
        # The loop `for j in range(i, K+1, i)` only runs if `is_prime[i]` was True initially?
        # No, `is_prime[i]` is set to False when i is a multiple of a smaller prime.
        # So the block only executes for primes.
        # This is correct.
        
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 600:
        # Should not happen, but just in case
        pass
        
    return domain, values

# EVOLVE-BLOCK-END
