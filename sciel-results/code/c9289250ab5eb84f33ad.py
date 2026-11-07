import math

def _square_free_part(n):
    """Return the square-free part of n."""
    sf = 1
    d = 2
    while d * d <= n:
        while n % (d * d) == 0:
            n //= (d * d)
        d += 1
    # If n is prime, it's square-free. If n is 1, sf is 1.
    # Actually, the loop removes all square factors.
    # The remaining n is square-free.
    return sf * n

def construct():
    # Target: minimize max(selected) - 210
    # Constraints: 1..100 distinct ints > 210, <= 10210
    # Product * 210 is a perfect square.
    
    # We search for a pair {a, b} with max(a,b) = M
    # Iterate M from 211 to 10210
    for M in range(211, 10211):
        s_M = _square_free_part(M)
        
        # Determine L such that 210 | s_M * L^2
        # L must contain primes in 210 not in s_M
        L = 1
        # Primes in 210: 2, 3, 5, 7
        for p in [2, 3, 5, 7]:
            if s_M % p != 0:
                L *= p
        
        # x = (s_M * L^2 * j^2) / 210
        # We need 210 < x < M
        # 210 < (s_M * L^2 * j^2) / 210 < M
        # 44100 < s_M * L^2 * j^2 < 210 * M
        
        A = s_M * L * L
        # j^2 > 44100 / A
        # j^2 < 210 * M / A
        
        j_min_sq = 44100 / A
        j_max_sq = (210 * M) / A
        
        # We need an integer j such that j_min_sq < j^2 < j_max_sq
        # j > sqrt(j_min_sq)
        # j < sqrt(j_max_sq)
        
        j_low = int(math.isqrt(j_min_sq))
        if j_low * j_low <= j_min_sq:
            j_low += 1
            
        j_high = int(math.isqrt(j_max_sq))
        
        if j_low <= j_high:
            # Valid j exists
            j = j_low
            x = (s_M * L * L * j * j) // 210
            
            # Verify x is integer and in range
            if 210 < x < M and x != M:
                # Check if product is square
                # This is guaranteed by construction, but verify just in case
                # prod = 210 * x * M
                # k = int(math.isqrt(prod))
                # if k*k == prod:
                return [x, M]
                
    # Fallback to trivial solution
    return [840]
