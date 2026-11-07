import math
import scipy.optimize as opt

def construct():
    n = 256
    # Generate primes
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
                
    primes = [p for p in range(2, n + 1) if is_prime[p]]
    
    # Split into small and large
    # Small: 2, 3, 5, 7, 11, 13
    # Large: rest
    small_primes = [2, 3, 5, 7, 11, 13]
    large_primes = [p for p in primes if p > 13]
    
    # Max power for each small prime
    P_p = {}
    for p in small_primes:
        max_power = p
        while max_power * p <= n:
            max_power *= p
        P_p[p] = max_power
        
    # Base result
    result = [1]
    # Add P_2 always
    result.append(P_p[2])
    
    # Candidates for matching: 3, 5, 7, 11, 13
    match_candidates = [3, 5, 7, 11, 13]
    
    best_gain = 0
    best_A = []
    best_matching = {}
    
    # Iterate over all subsets of match_candidates
    # There are 2^5 = 32 subsets
    import itertools
    for r in range(len(match_candidates) + 1):
        for A in itertools.combinations(match_candidates, r):
            if not A:
                continue
            
            # Build cost matrix
            # Rows: A, Cols: large_primes
            # We want to maximize sum of q*(p-1)
            # So minimize sum of -q*(p-1)
            
            # Filter large_primes that are valid for ANY p in A?
            # No, linear_sum_assignment handles it if we use INF.
            
            m = len(A)
            k = len(large_primes)
            
            if m == 0:
                continue
                
            cost_matrix = [[0] * k for _ in range(m)]
            
            for i, p in enumerate(A):
                for j, q in enumerate(large_primes):
                    if p * q <= n:
                        cost_matrix[i][j] = -q * (p - 1)
                    else:
                        cost_matrix[i][j] = 10**9 # Large cost for invalid edge
                        
            # Solve
            row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
            
            # Check if all rows are assigned to valid edges
            valid = True
            gain = 0
            matching = {}
            for i, j in zip(row_ind, col_ind):
                p = A[i]
                q = large_primes[j]
                if p * q > n:
                    valid = False
                    break
                matching[p] = q
                gain += q * (p - 1)
            
            if not valid:
                continue
                
            # Penalty
            penalty = sum(P_p[p] for p in A)
            net_gain = gain - penalty
            
            if net_gain > best_gain:
                best_gain = net_gain
                best_A = list(A)
                best_matching = matching
                
    # Construct final result
    # 1 is already in result
    # P_2 is already in result
    
    # Add P_p for small primes NOT in best_A
    for p in match_candidates:
        if p not in best_A:
            result.append(P_p[p])
            
    # Add large primes
    used_large = set(best_matching.values())
    for q in large_primes:
        if q in used_large:
            p = [p for p, qq in best_matching.items() if qq == q][0]
            result.append(p * q)
        else:
            result.append(q)
            
    return result
