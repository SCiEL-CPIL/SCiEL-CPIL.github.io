"""Find a, b with sigma(a)=sigma(b) minimizing |a/b - 3/2|."""
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    MAX = 10_000_000
    TARGET_NUM = 3
    TARGET_DEN = 2
    
    # Sieve for sigma(n)
    sigma = [0] * (MAX + 1)
    for i in range(1, MAX + 1):
        for j in range(i, MAX + 1, i):
            sigma[j] += i
    
    # For each b, compute the ideal a = 3b/2, check nearby integers
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # So we want to minimize |2a - 3b| / (2b) subject to sigma(a) = sigma(b)
    
    best_error = None
    best_pair = None
    
    # We need a, b positive and <= MAX
    # a must be in [1, MAX], b must be in [1, MAX]
    # a ≈ 3b/2, so b ≈ 2a/3
    # For a given b, a candidates are near 3b/2
    
    # Iterate over b values
    # a = floor(3b/2) or ceil(3b/2), check a few nearby
    for b in range(1, MAX + 1):
        if time.time() > deadline:
            break
        
        sb = sigma[b]
        
        # Ideal a = 3b/2
        a_ideal = 3 * b // 2
        remainder = 3 * b % 2
        
        # Check candidates around 3b/2
        # The error |a/b - 3/2| = |2a - 3b| / (2b)
        # For a = floor(3b/2): |2a - 3b| = |2*(3b//2) - 3b| = |3b - 2*(3b//2)|
        # If 3b is even: a = 3b/2, error = 0 (but that's invalid)
        # If 3b is odd: a = (3b-1)/2, error = 1/(2b)
        
        # Candidates to check: a_ideal - 2, a_ideal - 1, a_ideal, a_ideal + 1, a_ideal + 2
        # But we need sigma(a) == sigma(b)
        
        for da in range(-3, 4):
            a = a_ideal + da
            if a < 1 or a > MAX:
                continue
            if a == b:
                # Then a/b = 1, |1 - 3/2| = 1/2, not great but valid if sigma matches
                pass
            
            if sigma[a] == sb:
                # Compute error |a/b - 3/2| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                if num == 0:
                    # This would mean a/b = 3/2 exactly, which is invalid
                    continue
                err_num = num
                err_den = 2 * b
                # Compare errors: err_num/err_den vs best_error
                if best_error is None:
                    best_error = (err_num, err_den)
                    best_pair = (a, b)
                else:
                    # Compare err_num/err_den < best_num/best_den
                    # err_num * best_den < best_num * err_den
                    if err_num * best_error[1] < best_error[0] * err_den:
                        best_error = (err_num, err_den)
                        best_pair = (a, b)
    
    if best_pair is None:
        # Fallback: find any valid pair
        # sigma(1)=1, sigma(2)=3, sigma(3)=4, sigma(4)=7, sigma(5)=6, sigma(6)=12
        # Let's just return a known valid pair
        # sigma(1)=1, need another n with sigma(n)=1: only n=1
        # sigma(2)=3, need another: only n=2 (prime)
        # Actually sigma(p)=p+1 for prime p. sigma(2)=3, sigma(3)=4...
        # Let me think: sigma(1)=1. No other n has sigma=1.
        # sigma(2)=3. No other.
        # Let's find pairs with same sigma value.
        # sigma(4)=1+2+4=7, sigma(6)=1+2+3+6=12
        # Actually let me just search a small range
        for b2 in range(1, 10000):
            sb2 = sigma[b2]
            for a2 in range(1, 10000):
                if a2 != b2 and sigma[a2] == sb2:
                    num = abs(2 * a2 - 3 * b2)
                    if num > 0:
                        best_pair = (a2, b2)
                        break
            if best_pair:
                break
    
    if best_pair is None:
        return [11, 6]  # fallback
    
    a, b = best_pair
    # Ensure they're Python ints
    return [int(a), int(b)]
