import time
import random
import math

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin for 90s limit
    
    best_sequence = []
    best_length = 0
    
    def is_valid_sequence(seq):
        """Verify that all consecutive block products are distinct."""
        if not seq:
            return True
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True

    def greedy_construct(priority_func, time_limit=None):
        """
        Construct a sequence using a greedy approach based on priority_func.
        priority_func(x) returns a score; lower score is picked first.
        """
        values = []
        products = set()
        candidates = list(range(2, 129))
        
        # Sort candidates by priority
        # We need to dynamically pick the best available candidate
        # Since the set of available candidates shrinks, we can just iterate
        # but the "best" depends on the current state? No, the priority is static in this simple version.
        # However, the validity check depends on the current set of products.
        # So we should repeatedly pick the candidate with the lowest priority that is valid.
        
        available = set(range(2, 129))
        
        while available:
            if time_limit and (time.time() - start_time) > time_limit:
                break
            
            # Find the best valid candidate
            best_candidate = None
            best_priority = float('inf')
            
            # To speed up, we can pre-sort available by priority once, 
            # but since we remove elements, we just iterate the sorted list
            # Actually, sorting every iteration is O(N log N). 
            # N is at most 127. This is fine.
            
            sorted_candidates = sorted(available, key=priority_func)
            
            for x in sorted_candidates:
                # Check if adding x is valid
                tails = []
                prod = x
                tails.append(prod)
                for a in reversed(values):
                    prod *= a
                    tails.append(prod)
                
                if not products.intersection(tails):
                    best_candidate = x
                    break
            
            if best_candidate is None:
                break
            
            # Add the candidate
            values.append(best_candidate)
            available.remove(best_candidate)
            
            # Update products
            tails = []
            prod = best_candidate
            tails.append(prod)
            for a in reversed(values[:-1]): # values[:-1] is the prefix before adding best_candidate
                prod *= a
                tails.append(prod)
            products.update(tails)
            
            # Check time
            if time_limit and (time.time() - start_time) > time_limit:
                break
                
        return values

    # Strategy 1: Smallest first (Seed behavior)
    seq1 = greedy_construct(lambda x: x)
    if len(seq1) > best_length:
        best_sequence = seq1
        best_length = len(seq1)
        
    # Strategy 2: Largest first
    seq2 = greedy_construct(lambda x: -x)
    if len(seq2) > best_length:
        best_sequence = seq2
        best_length = len(seq2)
        
    # Strategy 3: Primes first (approximate by number of prime factors)
    def num_prime_factors(n):
        count = 0
        d = 2
        nn = n
        while d * d <= nn:
            while nn % d == 0:
                count += 1
                nn //= d
            d += 1
        if nn > 1:
            count += 1
        return count
            
    seq3 = greedy_construct(lambda x: num_prime_factors(x))
    if len(seq3) > best_length:
        best_sequence = seq3
        best_length = len(seq3)
        
    # Strategy 4: Random restarts
    random.seed(42)
    for _ in range(50):
        if (time.time() - start_time) > timeout:
            break
        def random_priority(x):
            return random.random()
        seq_r = greedy_construct(random_priority, timeout)
        if len(seq_r) > best_length:
            best_sequence = seq_r
            best_length = len(seq_r)
            
    # Strategy 5: Local Search / Hill Climbing on the best sequence found so far
    # Try to swap elements or insert new ones?
    # This is complex. Let's stick to multiple greedy runs.
    
    # One more strategy: Prefer numbers with large smallest prime factor
    def smallest_prime_factor(n):
        if n % 2 == 0: return 2
        d = 3
        while d * d <= n:
            if n % d == 0: return d
            d += 2
        return n
        
    seq4 = greedy_construct(lambda x: -smallest_prime_factor(x)) # Larger SPF is better
    if len(seq4) > best_length:
        best_sequence = seq4
        best_length = len(seq4)

    # Final check to ensure validity (just in case)
    if best_sequence and not is_valid_sequence(best_sequence):
        # Fallback to seed if something went wrong
        values, products = [], set()
        for x in range(2, 129):
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    return best_sequence
