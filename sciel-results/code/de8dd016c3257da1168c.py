"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

def _prime_factors(n):
    """Return dictionary of prime -> exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

def _is_prime_power(n):
    if _is_prime(n): return True
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        x = n
        while x % p == 0:
            x //= p
        if x == 1:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s kill
    
    # 1. Generate candidate integers in [2, 40]
    candidates = list(range(2, 41))
    
    # 2. Categorize candidates
    primes_list = []
    prime_powers_list = []
    others_list = []
    
    for num in candidates:
        if _is_prime(num):
            primes_list.append(num)
        elif _is_prime_power(num):
            prime_powers_list.append(num)
        else:
            others_list.append(num)
    
    # Sort each group
    primes_list.sort()
    prime_powers_list.sort()
    others_list.sort()
    
    # Strategy:
    # 1. Start with a strong base set of prime powers and primes that are known to be "independent" in a decoupled sense.
    #    We can safely take:
    #    - All primes (12): 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    #    Wait, if we take all primes, can we add prime powers?
    #    If we take 2 and 4, collision? No. 2*2=4 not a subset.
    #    But if we take 2, 4, 8, collision 2*4=8.
    #    So we can take at most 2 powers of 2 if we have 2?
    #    Actually, let's use the greedy check. It's robust.
    
    # To beat 16, we need 17 or 18.
    # Let's try to construct a set of 18 directly using a smart heuristic and then verify.
    
    # Candidate heuristic:
    # Pick primes and "large" prime powers that don't conflict.
    # Let's try a specific set that is likely valid.
    # Set S:
    # Primes: 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 (10 primes, excluding 2 and 3 to allow their powers)
    # 2-powers: 2, 4, 16, 32? 
    #   {2, 4, 16, 32} exponents 1, 2, 4, 5.
    #   Sums: 
    #   0, 1, 2, 4, 5 (from 5), 3(1+2), 6(1+5), 7(2+5), 9(4+5), 1+4=5 (collision with 5!).
    #   So {2, 4, 16, 32} invalid.
    #   {2, 4, 16} exponents 1, 2, 4. Sums 0..7 distinct. Valid.
    #   Can we add 32 (exp 5)? {1, 2, 4, 5}. 1+4=5 collision. No.
    #   So max 3 twos: {2, 4, 16}.
    # 3-powers: 3, 9, 27?
    #   Exponents 1, 2, 3. Sums: 0, 1, 2, 3, 3(1+2), collision.
    #   So max 2 threes: {3, 9} or {3, 27} or {9, 27}.
    #   {3, 9} sums 0, 1, 2, 3. Valid.
    # 5-powers: 5 is prime. 25 is power.
    #   If we take 5 (prime), can we take 25?
    #   {5, 25} exponents 1, 2. Sums 0, 1, 2, 3. Valid.
    #   So we can take 5 AND 25. That's 2 items for prime 5.
    #   Similarly for other primes?
    #   7: 49 > 40. Only 7.
    #   11: 121 > 40. Only 11.
    #   ...
    #   So only 2, 3, 5 have powers in range.
    
    # Let's build a set:
    # Twos: {2, 4, 16} (3 items)
    # Threes: {3, 9} (2 items)
    # Fives: {5, 25} (2 items)
    # Other Primes: {7, 11, 13, 17, 19, 23, 29, 31, 37} (9 items)
    # Total: 3 + 2 + 2 + 9 = 16.
    # We need 2 more.
    # Can we add any composite?
    # Or can we add another prime power?
    # No other prime powers <= 40.
    # Can we add a composite that doesn't conflict?
    # Conflict means: prod(subset S) = x * prod(subset T).
    # Since our set is decoupled by prime (mostly), x must involve primes 2, 3, 5.
    # Let's try 6 (2*3).
    # If we have 2 and 3, then 2*3=6 collision.
    # We have 2 and 3. So 6 is out.
    # Let's try 10 (2*5).
    # We have 2 and 5. 2*5=10 collision.
    # Let's try 12 (2^2 * 3).
    # We have 4 (2^2) and 3 (3^1). 4*3=12 collision.
    # Let's try 14 (2*7).
    # We have 2 and 7. 2*7=14 collision.
    # Let's try 15 (3*5).
    # We have 3 and 5. 3*5=15 collision.
    # Let's try 18 (2*3^2).
    # We have 2 and 9. 2*9=18 collision.
    # Let's try 20 (2^2 * 5).
    # We have 4 and 5. 4*5=20 collision.
    # Let's try 21 (3*7).
    # We have 3 and 7. 3*7=21 collision.
    # Let's try 22 (2*11).
    # We have 2 and 11. 2*11=22 collision.
    # Let's try 24 (2^3 * 3).
    # We have 2, 4, 16 (2^1, 2^2, 2^4) and 3, 9 (3^1, 3^2).
    # 24 = 2^3 * 3^1.
    # Can we make 2^3 * 3^1 from subset?
    # 2^3: can we make exp 3 from {1, 2, 4}? Yes, 1+2=3. So {2, 4} gives 2^3.
    # 3^1: can we make exp 1 from {1, 2}? Yes, {3} gives 3^1.
    # So {2, 4, 3} product is 2^3 * 3^1 = 24.
    # Collision.
    # Let's try 26 (2*13).
    # We have 2 and 13. 2*13=26 collision.
    # Let's try 28 (2^2 * 7).
    # We have 4 and 7. 4*7=28 collision.
    # Let's try 30 (2*3*5).
    # We have 2, 3, 5. 2*3*5=30 collision.
    # Let's try 33 (3*11).
    # We have 3 and 11. 3*11=33 collision.
    # Let's try 34 (2*17).
    # We have 2 and 17. 2*17=34 collision.
    # Let's try 35 (5*7).
    # We have 5 and 7. 5*7=35 collision.
    # Let's try 36 (2^2 * 3^2).
    # We have 4 (2^2) and 9 (3^2). 4*9=36 collision.
    # Let's try 38 (2*19).
    # We have 2 and 19. 2*19=38 collision.
    # Let's try 39 (3*13).
    # We have 3 and 13. 3*13=39 collision.
    
    # It seems our "decoupled" set of 16 is "saturated" with respect to composites.
    # Can we restructure to allow 17 or 18?
    # Maybe drop a "weak" prime power to allow more composites?
    # Or use a different set of prime powers?
    
    # Let's try to find a 17-set or 18-set via search.
    # Greedy with multiple restarts and randomization.
    
    import random
    random.seed(42)
    
    best_set = []
    
    # Heuristic orders
    orders = [
        sorted(candidates),
        sorted(candidates, reverse=True),
        primes_list + prime_powers_list + others_list,
        prime_powers_list + primes_list + others_list,
        others_list + prime_powers_list + primes_list,
    ]
    
    # Add some random shuffles
    for _ in range(20):
        shuffled = candidates[:]
        random.shuffle(shuffled)
        orders.append(shuffled)
    
    for order in orders:
        current_set = []
        products = {1}
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Check collision
            new_products = set()
            collision = False
            for p in products:
                np = p * x
                if np in products:
                    collision = True
                    break
                new_products.add(np)
            
            if not collision:
                current_set.append(x)
                products.update(new_products)
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        if len(best_set) >= 18:
            break
    
    # Verify best_set
    if len(best_set) > 0:
        prods = {1}
        valid = True
        for x in best_set:
            new_prods = {p * x for p in prods}
            if prods.intersection(new_prods):
                valid = False
                break
            prods.update(new_prods)
        
        if valid:
            return [int(x) for x in best_set]
    
    # Fallback: The 16-set we constructed manually
    fallback = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify fallback
    prods = {1}
    for x in fallback:
        new_prods = {p * x for p in prods}
        if prods.intersection(new_prods):
            return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        prods.update(new_prods)
    
    return [int(x) for x in fallback]
# EVOLVE-BLOCK-END
