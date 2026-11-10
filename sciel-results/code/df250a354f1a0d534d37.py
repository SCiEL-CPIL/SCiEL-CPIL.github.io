import time
import math
from itertools import combinations

def _tau_perp(n):
    """Compute the number of adjacent coprime pairs among divisors of n."""
    if n < 2:
        return 0
    # Factorize n
    factors = []
    temp = n
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            exp = 0
            while temp % d == 0:
                exp += 1
                temp //= d
            factors.append((d, exp))
        d += 1
    if temp > 1:
        factors.append((temp, 1))
    
    # Generate all divisors
    divisors = [1]
    for p, e in factors:
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    
    divisors.sort()
    
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i + 1]) == 1:
            count += 1
    return count

def _sieve(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start = time.time()
    deadline = start + 140.0  # 140s deadline, 10s safety margin
    
    # Primes up to 200
    primes = _sieve(200)
    
    # Start with the product of the first 15 primes (known good baseline)
    # 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47 = 614889782588491410 (approx 6.1e17)
    base_primes = primes[:15]
    best_n = 1
    for p in base_primes:
        best_n *= p
    best_score = _tau_perp(best_n)
    
    # Greedy local search:
    # 1. Try swapping one prime in the current set with a larger prime not in the set.
    # 2. Accept if the score improves.
    # 3. Repeat until no improvement or time runs out.
    
    current_primes = set(base_primes)
    current_n = best_n
    current_score = best_score
    
    # Precompute which primes are in the current set
    in_set = [False] * len(primes)
    for p in current_primes:
        if p < len(primes):
            in_set[primes.index(p)] = True
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Try swapping each prime in the current set with a prime not in the set
        # We only consider primes up to index 50 to keep it manageable
        max_idx = min(50, len(primes))
        
        for i in range(max_idx):
            if not in_set[i]:
                # Try adding primes[i] and removing some prime in the set
                # To keep the product <= 10^18, we should remove the largest prime in the set
                # or a prime that allows the new prime to fit
                # Actually, let's just try swapping the largest prime in the set with primes[i]
                # This maintains the count of primes (15) and tries to change the structure
                if len(current_primes) == 0:
                    break
                
                # Find the largest prime in current_primes
                max_p_in_set = max(current_primes)
                
                # If the new prime is smaller than the max, the product decreases, so it fits.
                # If the new prime is larger, the product increases.
                # Let's check if product * primes[i] / max_p_in_set <= 10^18
                new_product = current_n * primes[i] // max_p_in_set
                if new_product <= 10**18:
                    # Generate new n
                    new_n = new_product
                    # We need to recalculate the score
                    score = _tau_perp(new_n)
                    if score > current_score:
                        # Accept the swap
                        current_primes.remove(max_p_in_set)
                        current_primes.add(primes[i])
                        current_n = new_n
                        current_score = score
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
                        
                        # Update in_set
                        in_set[primes.index(max_p_in_set)] = False
                        in_set[i] = True
                        improved = True
                        # Restart the loop to try more swaps from this new state
                        break
        
        # Also try removing a prime to see if a smaller set with better structure helps?
        # Unlikely, since more primes usually means more divisors and more potential coprime pairs.
        # But let's stick to swapping for now.
        
        # If no swap improved the score, try a random restart with a different initial set?
        # Or try a different local search neighborhood.
        
        # Let's try a different neighborhood: swap two primes in the set with two primes not in the set.
        # This is more expensive but might escape local optima.
        if not improved:
            # Try 2-swaps
            # This is C(15, 2) * C(50-15, 2) which is large.
            # Let's just try a few random 2-swaps
            import random
            random.seed(int(time.time() * 1000) % 10000)
            
            for _ in range(100): # Try 100 random 2-swaps
                if time.time() > deadline - 10:
                    break
                
                # Pick 2 primes to remove
                if len(current_primes) < 2:
                    break
                rem1, rem2 = random.sample(list(current_primes), 2)
                
                # Pick 2 primes to add from outside
                outside_primes = [p for j, p in enumerate(primes[:50]) if not in_set[j]]
                if len(outside_primes) < 2:
                    break
                add1, add2 = random.sample(outside_primes, 2)
                
                new_product = current_n * add1 * add2 // (rem1 * rem2)
                if new_product <= 10**18:
                    score = _tau_perp(new_product)
                    if score > current_score:
                        current_primes.discard(rem1)
                        current_primes.discard(rem2)
                        current_primes.add(add1)
                        current_primes.add(add2)
                        current_n = new_product
                        current_score = score
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
                        
                        # Update in_set
                        in_set[primes.index(rem1)] = False
                        in_set[primes.index(rem2)] = False
                        in_set[primes.index(add1)] = True
                        in_set[primes.index(add2)] = True
                        improved = True
                        break
    
    return best_n
