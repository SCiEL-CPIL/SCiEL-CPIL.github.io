<<<<<<< SEARCH
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    
    # Precompute primes up to 128
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
    
    all_numbers = list(range(2, 129))
    primes = [x for x in all_numbers if is_prime(x)]
    composites = [x for x in all_numbers if not is_prime(x)]
    
    # Fast product-check using incremental computation with early termination
    def try_insertion(seq, c):
        """Try to insert c into seq maintaining strict increase and unique block products."""
        n = len(seq)
        
        # Find valid insertion positions (where seq[i-1] < c < seq[i])
        for i in range(n + 1):
            if i > 0 and seq[i-1] >= c:
                continue
            if i < n and c >= seq[i]:
                continue
            
            # Build new sequence
            new_seq = seq[:i] + [c] + seq[i:]
            m = len(new_seq)
            expected_count = m * (m + 1) // 2
            
            # Check all consecutive block products for uniqueness
            prods = set()
            valid = True
            for a in range(m):
                p = 1
                for b in range(a, m):
                    p *= new_seq[b]
                    if p in prods:
                        valid = False
                        break
                    prods.add(p)
                if not valid:
                    break
            
            if valid and len(prods) == expected_count:
                return new_seq
        
        return None
    
    # Strategy: multiple randomized greedy builds, then try to improve
    def build_seq(seed_order):
        """Build a sequence starting with primes, then greedily add composites."""
        seq = primes[:]
        used = set(seq)
        remaining = [x for x in composites if x not in used]
        
        # Apply seed_order shuffle
        random.Random(seed_order).shuffle(remaining)
        
        progress = True
        while progress:
            progress = False
            for c in remaining[:]:
                new_seq = try_insertion(seq, c)
                if new_seq is not None:
                    seq = new_seq
                    remaining.remove(c)
                    progress = True
                    break
        return seq
    
    best_seq = primes[:]
    best_len = len(best_seq)
    
    # Try many different random seeds/orders
    while time.time() < deadline:
        seed_val = random.randint(0, 2**31 - 1)
        seq = build_seq(seed_val)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
            # Try to further improve best_seq with local search
            improved = True
            while improved and time.time() < deadline:
                improved = False
                used = set(best_seq)
                remaining2 = [x for x in composites if x not in used]
                random.shuffle(remaining2)
                for c in remaining2:
                    new_seq = try_insertion(best_seq, c)
                    if new_seq is not None:
                        best_seq = new_seq
                        best_len = len(best_seq)
                        improved = True
                        break
    
    return best_seq
=======
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    
    # Precompute primes up to 128
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
    
    all_numbers = list(range(2, 129))
    primes = [x for x in all_numbers if is_prime(x)]
    composites = [x for x in all_numbers if not is_prime(x)]
    
    # Precompute prime factorization for each number 2..128
    def factorize(n):
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
    
    factor_cache = {n: factorize(n) for n in all_numbers}
    
    # Count distinct prime factors for ordering heuristic
    def num_distinct_primes(n):
        return len(factor_cache[n])
    
    # Sort composites by number of distinct prime factors (descending) for better insertion order
    composites_sorted = sorted(composites, key=num_distinct_primes, reverse=True)
    
    # Fast product-check using incremental computation with early termination
    def get_all_products(seq):
        """Compute all consecutive block products of seq."""
        m = len(seq)
        prods = set()
        for a in range(m):
            p = 1
            for b in range(a, m):
                p *= seq[b]
                prods.add(p)
        return prods
    
    def try_insertion(seq, c):
        """Try to insert c into seq maintaining strict increase and unique block products."""
        n = len(seq)
        
        # Find valid insertion positions (where seq[i-1] < c < seq[i])
        for i in range(n + 1):
            if i > 0 and seq[i-1] >= c:
                continue
            if i < n and c >= seq[i]:
                continue
            
            # Build new sequence
            new_seq = seq[:i] + [c] + seq[i:]
            m = len(new_seq)
            expected_count = m * (m + 1) // 2
            
            # Check all consecutive block products for uniqueness
            prods = set()
            valid = True
            for a in range(m):
                p = 1
                for b in range(a, m):
                    p *= new_seq[b]
                    if p in prods:
                        valid = False
                        break
                    prods.add(p)
                if not valid:
                    break
            
            if valid and len(prods) == expected_count:
                return new_seq
        
        return None
    
    def try_swap(seq, i, j):
        """Try swapping elements at positions i and j, maintaining strict increase and unique products."""
        if seq[i] >= seq[j]:
            return None
        new_seq = seq[:]
        new_seq[i], new_seq[j] = seq[j], seq[i]
        m = len(new_seq)
        expected_count = m * (m + 1) // 2
        
        prods = set()
        valid = True
        for a in range(m):
            p = 1
            for b in range(a, m):
                p *= new_seq[b]
                if p in prods:
                    valid = False
                    break
                prods.add(p)
            if not valid:
                break
        
        if valid and len(prods) == expected_count:
            return new_seq
        return None
    
    # Strategy: multiple randomized greedy builds, then try to improve
    def build_seq(seed_order):
        """Build a sequence starting with primes, then greedily add composites."""
        seq = primes[:]
        used = set(seq)
        remaining = [x for x in composites if x not in used]
        
        # Apply seed_order shuffle with bias toward numbers with more prime factors
        rng = random.Random(seed_order)
        # Weight by number of distinct prime factors
        weights = [num_distinct_primes(x) + 1 for x in remaining]
        # Shuffle with some randomness
        rng.shuffle(remaining)
        
        progress = True
        while progress and time.time() < deadline:
            progress = False
            for c in remaining[:]:
                new_seq = try_insertion(seq, c)
                if new_seq is not None:
                    seq = new_seq
                    remaining.remove(c)
                    progress = True
                    break
        return seq
    
    def local_search(seq, time_limit):
        """Try to improve seq by inserting remaining numbers and doing swaps."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            used = set(seq)
            remaining = [x for x in composites if x not in used]
            # Sort by heuristic: more prime factors first
            remaining.sort(key=num_distinct_primes, reverse=True)
            random.shuffle(remaining)
            
            for c in remaining:
                new_seq = try_insertion(seq, c)
                if new_seq is not None:
                    seq = new_seq
                    improved = True
                    break
            
            # Try some random swaps to escape local optima
            if not improved:
                n = len(seq)
                for _ in range(50):
                    i, j = random.sample(range(n), 2)
                    if i > j:
                        i, j = j, i
                    new_seq = try_swap(seq, i, j)
                    if new_seq is not None:
                        seq = new_seq
                        improved = True
                        break
        
        return seq
    
    best_seq = primes[:]
    best_len = len(best_seq)
    
    # Try many different random seeds/orders
    while time.time() < deadline:
        seed_val = random.randint(0, 2**31 - 1)
        seq = build_seq(seed_val)
        
        # Do local search on this candidate
        seq = local_search(seq, deadline)
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
    
    return best_seq
>>>>>>> REPLACE
