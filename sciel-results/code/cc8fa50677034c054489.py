import time
from math import gcd, isqrt
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin

    N = 256
    
    # Generate primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Map number to set of prime factors
    # We only care about primes up to N
    def get_prime_factors(x):
        factors = set()
        temp = x
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors.add(temp)
        return factors

    # Precompute prime factors for all numbers 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        prime_factors[i] = get_prime_factors(i)
    prime_factors[1] = set()

    # Initialize S with all primes and 1
    S = [1] + primes
    
    # Helper to check if a set of numbers can be merged into x
    # A: list of numbers from S
    # Returns x if merge is beneficial, else None
    def try_merge(A):
        # Union of prime factors
        P_A = set()
        sum_A = 0
        for a in A:
            P_A.update(prime_factors[a])
            sum_A += a
            
        if not P_A:
            return None
            
        # Calculate M = product of primes in P_A
        M = 1
        for p in P_A:
            M *= p
            if M > N:
                break
        
        if M > N:
            return None
            
        # Find largest x <= N such that PF(x) == P_A
        best_x = 0
        # x must be a multiple of M
        # x = M * k
        # PF(x) must be exactly P_A
        for k in range(1, N // M + 1):
            x = M * k
            if x > N:
                break
            if prime_factors[x] == P_A:
                if x > best_x:
                    best_x = x
        
        if best_x > sum_A:
            return best_x
        return None

    # Iterative improvement
    improved = True
    while improved:
        improved = False
        current_size = len(S)
        
        # Check subsets of size 2
        for i in range(current_size):
            for j in range(i + 1, current_size):
                A = [S[i], S[j]]
                x = try_merge(A)
                if x is not None:
                    # Replace A with x
                    # Remove S[i] and S[j]
                    # Be careful with indices
                    S.pop(j)
                    S.pop(i)
                    S.append(x)
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # Check subsets of size 3
        for i in range(current_size):
            for j in range(i + 1, current_size):
                for k in range(j + 1, current_size):
                    A = [S[i], S[j], S[k]]
                    x = try_merge(A)
                    if x is not None:
                        S.pop(k)
                        S.pop(j)
                        S.pop(i)
                        S.append(x)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # Check subsets of size 4
        # This might be slow if done naively, but current_size is decreasing
        # Also, M grows fast, so many will be skipped
        for i in range(current_size):
            for j in range(i + 1, current_size):
                for k in range(j + 1, current_size):
                    for l in range(k + 1, current_size):
                        A = [S[i], S[j], S[k], S[l]]
                        x = try_merge(A)
                        if x is not None:
                            S.pop(l)
                            S.pop(k)
                            S.pop(j)
                            S.pop(i)
                            S.append(x)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if time.time() > deadline:
            break

    # Verify result
    if len(set(S)) != len(S):
        return [1] + primes
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            if gcd(S[i], S[j]) != 1:
                return [1] + primes
    for x in S:
        if x < 1 or x > N:
            return [1] + primes

    return S
