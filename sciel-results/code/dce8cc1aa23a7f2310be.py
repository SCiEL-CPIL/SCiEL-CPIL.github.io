import time
import math

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(is_prime[i*i:limit+1:i]))
    return [i for i, v in enumerate(is_prime) if v]

def construct():
    start_time = time.time()
    deadline = start_time + 118
    
    # Precompute primes up to 10^6
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    max_k = 200
    max_n = 10**12
    
    def check_window(s, k):
        """
        Check if phi(s), ..., phi(s+k-1) are distinct.
        Uses segmented sieve approach to factorize numbers in [s, s+k-1].
        """
        # remaining[i] holds the remaining part of s+i after dividing by small primes
        remaining = [s + i for i in range(k)]
        # phi_val[i] will hold phi(s+i)
        phi_val = [1] * k
        
        # For each prime p, process multiples in the window
        for p in primes:
            if p * p > s + k - 1:
                break
            
            # Find the first multiple of p in [s, s+k-1]
            # start = ceil(s / p) * p
            start = ((s + p - 1) // p) * p
            
            # Iterate through multiples
            for j in range(start - s, k, p):
                n = s + j
                if remaining[j] % p == 0:
                    phi_val[j] -= phi_val[j] // p
                    while remaining[j] % p == 0:
                        remaining[j] //= p
        
        # Handle remaining prime factors > sqrt(max_n)
        for i in range(k):
            if remaining[i] > 1:
                phi_val[i] -= phi_val[i] // remaining[i]
                
        # Check for distinctness
        return len(set(phi_val)) == k

    # Search strategy
    # Start from high end and go down? Or random?
    # The problem statement says "Long runs below 10^12 are finite evidence".
    # Let's try a few high values and some random ones.
    
    # Try the largest possible start
    s_max = max_n - max_k + 1
    if check_window(s_max, max_k):
        return {'start': s_max, 'length': max_k}
        
    # Try a few offsets from the top
    # And maybe some random starts
    # Since we don't know the distribution, let's just scan downwards a bit.
    # But scanning 1000 windows might take too long if factorization is slow.
    
    # Let's estimate time for one window.
    # 78,000 primes.
    # Inner loop runs k/p times.
    # Total inner loop iterations ~ k * sum(1/p) ~ 200 * 2.5 = 500.
    # But the outer loop runs 78,000 times.
    # The cost is dominated by the outer loop overhead in Python.
    # 78,000 iterations * (check + division) might be ~0.05s?
    # 118s / 0.05s = 2360 windows.
    # So we can check ~2000 windows.
    
    # Let's check a range of starts.
    # Maybe the best run is not at the very top.
    # Let's try starts at 10^12 - 200, 10^12 - 400, etc.
    # Or just linear search downwards.
    
    best_k = 1
    best_s = 1
    
    # Try to find length 200
    # If not found, try 199, etc?
    # The task asks to maximize k.
    # It's likely that if 200 is not possible, 199 might be?
    # But the constraint is s+k-1 <= 10^12.
    # So for k=199, s can go up to 10^12 - 198.
    
    # Let's just search for k=200 first.
    # If we find one, return it.
    # If time runs out, return the best found (which might be 1 if none found).
    # But wait, if we don't find 200, should we try 199?
    # The score is the value k.
    # So finding k=199 is better than k=1.
    
    # Strategy:
    # 1. Search for k=200.
    # 2. If not found, search for k=199, etc.
    # But searching for k=199 is similar cost.
    # We can reuse the factorization? No, the window is different.
    
    # Let's implement a loop that tries decreasing k.
    # But we need to be careful with time.
    
    # Actually, let's just try to find ANY valid run of length 200.
    # If we fail, we try 199, etc.
    
    # To speed up, we can sample.
    # Let's try 1000 random starts for k=200.
    # Then 500 for k=199, etc.
    
    import random
    random.seed(42)
    
    # Pre-allocate list for random starts
    # We need s >= 1 and s+k-1 <= 10^12.
    
    # Let's try a mix of high-end and random.
    
    # First, try the top few
    for offset in range(0, 50):
        if time.time() > deadline:
            break
        s = max_n - max_k + 1 - offset
        if s < 1: break
        if check_window(s, max_k):
            return {'start': s, 'length': max_k}
            
    # Random search for k=200
    # 10^12 is large.
    # Let's pick random s in [1, 10^12 - 199]
    # But Python int is fine.
    
    for i in range(1000):
        if time.time() > deadline:
            break
        s = random.randint(1, max_n - max_k + 1)
        if check_window(s, max_k):
            return {'start': s, 'length': max_k}
            
    # If k=200 not found, try k=199
    # ...
    
    # This might take too long if we do full search for each k.
    # Let's just return the best found so far.
    # But we haven't found any yet in this logic.
    
    # Let's restructure:
    # Keep a list of (k, s) candidates.
    # Try to find max k.
    
    # Given the low probability, it's better to just try k=200 extensively.
    # If we find one, we are done.
    # If not, we return k=1?
    # The seed program returns k=1 if it fails.
    # But maybe k=10 is possible?
    # Collisions are common, but 10 distinct?
    # Let's check if small k is easy.
    # k=1 is always valid.
    # k=2: phi(n) != phi(n+1).
    # phi(1)=1, phi(2)=1. Collision.
    # phi(2)=1, phi(3)=2. Distinct. So s=2, k=2 is valid.
    # So we should at least return k=2 if we can't find 200.
    
    # Let's add a fallback for small k.
    
    # Check s=2, k=2
    if check_window(2, 2):
         # This is likely true
         pass
         
    # Let's just do a quick search for small k to ensure we get > 1.
    # But the score is k.
    # So 200 is much better than 2.
    
    # The main risk is timeout.
    # 1000 random checks for k=200.
    # If each takes 0.05s, that's 50s.
    # We have 118s.
    # So we can do ~2000 checks.
    
    # Let's refine the check_window to be as fast as possible.
    # Using local variables.
    
    def check_window_fast(s, k, primes):
        remaining = [s + i for i in range(k)]
        phi_val = [1] * k
        
        # local vars for speed
        rem = remaining
        ph = phi_val
        ps = primes
        
        for p in ps:
            if p * p > s + k - 1:
                break
            
            start = ((s + p - 1) // p) * p
            
            # if start > s + k - 1, no multiples
            if start > s + k - 1:
                continue
                
            for j in range(start - s, k, p):
                if rem[j] % p == 0:
                    ph[j] -= ph[j] // p
                    while rem[j] % p == 0:
                        rem[j] //= p
        
        for i in range(k):
            if rem[i] > 1:
                ph[i] -= ph[i] // rem[i]
                
        # Check distinct
        seen = set()
        for v in ph:
            if v in seen:
                return False
            seen.add(v)
        return True

    # Use this faster version
    
    # Retry logic
    # 1. Top 100
    for offset in range(0, 100):
        if time.time() > deadline:
            break
        s = max_n - max_k + 1 - offset
        if s < 1: break
        if check_window_fast(s, max_k, primes):
            return {'start': s, 'length': max_k}
            
    # 2. Random 2000
    for i in range(2000):
        if time.time() > deadline:
            break
        s = random.randint(1, max_n - max_k + 1)
        if check_window_fast(s, max_k, primes):
            return {'start': s, 'length': max_k}
            
    # 3. Try smaller k?
    # If we have time left, try k=199, etc.
    # But we need to be careful.
    # Let's just return 1 if we didn't find 200.
    # Wait, the seed program returns 1.
    # But we can do better.
    # Let's check if s=2, k=2 is valid.
    # phi(2)=1, phi(3)=2. Yes.
    # So return k=2 is better than k=1.
    
    # Let's verify s=2, k=2
    if check_window_fast(2, 2, primes):
        return {'start': 2, 'length': 2}
        
    return {'start': 1, 'length': 1}
