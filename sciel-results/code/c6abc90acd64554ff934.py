import math

def _largest_prime_factor(n):
    if n <= 1:
        return 1
    max_factor = 1
    # Handle 2
    while n % 2 == 0:
        max_factor = 2
        n //= 2
    # Handle odd factors
    f = 3
    while f * f <= n:
        while n % f == 0:
            max_factor = f
            n //= f
        f += 2
    if n > 1:
        max_factor = n
    return max_factor

def construct():
    total = 123457
    best_val = float('inf')
    best_a = 1
    best_b = total - 1
    
    # We only need to check up to total // 2 because a+b is symmetric
    for a in range(1, total // 2 + 1):
        b = total - a
        pa = _largest_prime_factor(a)
        pb = _largest_prime_factor(b)
        
        current_max = pa if pa > pb else pb
        
        if current_max < best_val:
            best_val = current_max
            best_a = a
            best_b = b
            # Theoretical minimum for this problem is 3 (one even, one odd, 
            # odd cannot be 1 as 123456 is not smooth enough, even min is 2, 
            # odd min is 3 if smooth). 
            # If we find 3, we cannot do better.
            if best_val <= 3:
                break
                
    return [best_a, best_b]
