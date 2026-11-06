import random
import math

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def construct():
    candidates = list(range(2, 129))
    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    best_seq = []
    best_len = 0
    
    # Helper to check validity and return new prods if valid
    def try_insert(seq, prods, x, k):
        L = len(seq)
        
        # Calculate left suffix products ending at k-1
        # left_prods[i] = prod(seq[i:k])
        # left_prods[0] = prod(seq[0:k]), ..., left_prods[k] = 1
        left_prods = [1] * (k + 1)
        acc = 1
        for i in range(k - 1, -1, -1):
            acc *= seq[i]
            left_prods[i] = acc
            
        # Calculate right prefix products starting at k
        # right_prods[j] = prod(seq[k:j+1])
        # right_prods[0] = prod(seq[k:k+1]) = seq[k], ..., right_prods[L-k] = prod(seq[k:L])
        # Let's index by offset from k
        # right_prods[0] = 1 (empty)
        # right_prods[1] = seq[k]
        # ...
        right_prods = [1] * (L - k + 1)
        acc = 1
        for j in range(k, L):
            acc *= seq[j]
            right_prods[j - k + 1] = acc
            
        new_prods = set()
        
        # 1. Singleton x
        new_prods.add(x)
        
        # 2. Left extensions: left_prods[i] * x
        for i in range(k):
            new_prods.add(left_prods[i] * x)
            
        # 3. Right extensions: x * right_prods[j]
        for j in range(1, L - k + 1):
            new_prods.add(x * right_prods[j])
            
        # 4. Combined: left_prods[i] * x * right_prods[j]
        for i in range(k):
            lp = left_prods[i] * x
            for j in range(1, L - k + 1):
                new_prods.add(lp * right_prods[j])
                
        # Check collisions with existing
        if prods.intersection(new_prods):
            return None
            
        # Check internal collisions
        # We added to a set, so we just need to ensure we didn't lose any?
        # No, if two generated values are same, set size is smaller.
        # But we don't know the expected size easily.
        # However, if we just return the set, the caller can use it.
        # The condition "products ... must be distinct" means the SET of all products must have size equal to number of subarrays.
        # If we merge new_prods into prods, the size should increase by len(new_prods) - (collisions with prods).
        # We already checked collisions with prods.
        # We need to ensure new_prods are distinct from each other.
        # Since we used a set, if there was a collision among new_prods, the set size would be smaller than the number of generated values.
        # How many values did we generate?
        # 1 (singleton) + k (left) + (L-k) (right) + k*(L-k) (combined)
        # Total = 1 + k + L - k + kL - k^2 = 1 + L + kL - k^2.
        # This is the number of subarrays involving x.
        # If len(new_prods) < expected_count, there was a collision.
        
        expected_count = 1 + L + k * (L - k)
        if len(new_prods) < expected_count:
            return None
            
        return new_prods

    # Run trials
    # Trial 0: Deterministic
    # Trials 1-19: Random shuffles of composites
    
    for trial in range(20):
        comps = composites[:]
        if trial > 0:
            random.shuffle(comps)
            
        order = primes + comps
        
        seq = []
        prods = set()
        
        for x in order:
            inserted = False
            # Try positions
            # Randomize position order for diversity?
            # Or just 0 to len(seq)
            # Let's try 0 to len(seq)
            for k in range(len(seq) + 1):
                new_prods = try_insert(seq, prods, x, k)
                if new_prods is not None:
                    seq.insert(k, x)
                    prods.update(new_prods)
                    inserted = True
                    break
            
            # If not inserted, skip x
            
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
    return best_seq
