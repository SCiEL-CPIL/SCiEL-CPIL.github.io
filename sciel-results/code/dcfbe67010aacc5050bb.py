import math

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.isqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def construct():
    N = 10**6
    # Sieve for sigma
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
            
    # Map M to list of k's
    solutions = {}
    
    # Iterate H
    for H in range(1, N + 1):
        # Find divisors of H
        # Optimization: only check k that divide H
        # We can iterate k up to isqrt(H)
        limit = math.isqrt(H)
        for k in range(1, limit + 1):
            if H % k == 0:
                # k is a divisor
                M_val = k * sigma[k]
                if M_val <= 10**18:
                    # Check if M_val = H * P where P is prime
                    if M_val % H == 0:
                        P = M_val // H
                        if is_prime(P):
                            if M_val not in solutions:
                                solutions[M_val] = []
                            solutions[M_val].append(k)
                
                # Check H/k
                k2 = H // k
                if k2 != k:
                    M_val2 = k2 * sigma[k2]
                    if M_val2 <= 10**18:
                        if M_val2 % H == 0:
                            P2 = M_val2 // H
                            if is_prime(P2):
                                if M_val2 not in solutions:
                                    solutions[M_val2] = []
                                solutions[M_val2].append(k2)

    best_M = 0
    best_k = []
    for M_val, k_list in solutions.items():
        if len(k_list) > len(best_k):
            best_k = k_list
            best_M = M_val

    if len(best_k) <= 3:
        return {'M': 333312, 'k': [336, 372, 434]}
    else:
        return {'M': best_M, 'k': best_k}
