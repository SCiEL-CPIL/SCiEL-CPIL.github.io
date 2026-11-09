import time
import random
import math

# Precompute small primes for faster factorization
_SMALL_PRIMES = []
def _init_primes():
    is_prime = [True] * 200
    is_prime[0] = is_prime[1] = False
    for i in range(2, 200):
        if is_prime[i]:
            _SMALL_PRIMES.append(i)
            for j in range(i*i, 200, i):
                is_prime[j] = False
_init_primes()

def get_primes(n):
    """Return the set of prime factors of n."""
    if n == 1:
        return set()
    primes = set()
    for p in _SMALL_PRIMES:
        if p * p > n:
            break
        while n % p == 0:
            primes.add(p)
            n //= p
    if n > 1:
        primes.add(n)
    return primes

def evaluate(set_a):
    """Calculate the union of prime factors of all pairwise sums."""
    all_primes = set()
    n = len(set_a)
    for i in range(n):
        for j in range(i + 1, n):
            s = set_a[i] + set_a[j]
            all_primes.update(get_primes(s))
    return all_primes

def construct():
    start_time = time.time()
    timeout = 50  # Tighter timeout to ensure we finish well within 60s
    
    # Initial candidates
    best_set = None
    best_primes = None
    best_count = float('inf')
    
    # Heuristic candidates
    candidates = []
    
    # 1. Multiples of 6 (ensures 2 and 3 are factors)
    candidates.append(list(range(6, 6*12+1, 6)))
    
    # 2. Multiples of 30 (ensures 2, 3, 5)
    candidates.append(list(range(30, 30*12+1, 30)))
    
    # 3. Powers of 2 (suggested by inspiration 1)
    candidates.append([2**i for i in range(12)])
    
    # 4. Small multiples of 3
    candidates.append(list(range(3, 3*12+1, 3)))
    
    # 5. Small multiples of 4
    candidates.append(list(range(4, 4*12+1, 4)))
    
    # 6. Small multiples of 12
    candidates.append(list(range(12, 12*12+1, 12)))
    
    # 7. Residue classes mod small numbers
    for mod in range(2, 20):
        for res in range(mod):
            nums = [x for x in range(1, 10001) if x % mod == res]
            if len(nums) >= 12:
                candidates.append(nums[:12])
    
    # 8. Random initializations
    for _ in range(5):
        candidates.append([random.randint(1, 10000) for _ in range(12)])
    
    # Evaluate all candidates
    for c in candidates:
        if time.time() - start_time > timeout:
            break
        # Ensure distinctness and bounds
        c = [max(1, min(10000, x)) for x in c]
        if len(set(c)) < 12:
            # Fix duplicates
            while len(set(c)) < 12:
                c.append(random.randint(1, 10000))
        c = c[:12]
        
        primes = evaluate(c)
        if len(primes) < best_count:
            best_count = len(primes)
            best_primes = primes
            best_set = c[:]
    
    if best_set is None:
        return list(range(1, 13))
    
    # Local Search
    current_set = best_set[:]
    current_primes = best_primes
    current_count = best_count
    
    # Precompute a list of "good" deltas to try
    deltas = list(range(-50, 51)) + list(range(-500, 501, 50)) + list(range(-2000, 2001, 500))
    random.shuffle(deltas)
    
    # Use a set for fast membership check
    current_set_as_set = set(current_set)
    
    while time.time() - start_time < timeout - 2:
        # Pick a random index to change
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Pick a new value
        if random.random() < 0.5:
            d = random.choice(deltas)
            new_val = old_val + d
        else:
            new_val = random.randint(1, 10000)
            
        if new_val < 1 or new_val > 10000:
            continue
            
        if new_val == old_val:
            continue
            
        # Check distinctness using set
        if new_val in current_set_as_set:
             continue
        
        # Swap
        new_set = current_set[:]
        new_set[idx] = new_val
        
        # Evaluate
        new_primes = evaluate(new_set)
        new_count = len(new_primes)
        
        if new_count < current_count:
            current_set = new_set
            current_set_as_set = set(current_set)
            current_primes = new_primes
            current_count = new_count
            if current_count < best_count:
                best_count = current_count
                best_primes = current_primes
                best_set = current_set[:]
                
    # Return the best set found
    return best_set
