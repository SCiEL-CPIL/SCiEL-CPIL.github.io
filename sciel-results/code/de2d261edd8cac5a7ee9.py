import random
import time
import sys

# EVOLVE-BLOCK-START

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds safety margin
    
    # Numbers in range 2..40
    candidates = list(range(2, 41))
    
    # Identify primes to start with a strong base
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    best_set = primes
    best_size = len(primes)
    
    # Function to check if a number can be added to a set with given product set
    def can_add(x, prod_set):
        # Check if x * p is in prod_set for any p in prod_set
        # Note: x itself is not in prod_set initially? 
        # If 1 is in prod_set, then x*1 = x. So x must not be in prod_set.
        # The check x*p in prod_set covers the case p=1 -> x in prod_set.
        for p in prod_set:
            if x * p in prod_set:
                return False
        return True

    # Function to update the product set
    def update_prod_set(prod_set, x):
        new_prods = set(prod_set)
        for p in prod_set:
            new_prods.add(x * p)
        return new_prods

    # Initial product set for primes
    # We can calculate this once, but since we restart, we do it in the loop.
    
    num_trials = 5000
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        # Randomize the order of composites
        shuffled_composites = composites.copy()
        random.shuffle(shuffled_composites)
        
        current_set = primes.copy()
        # Calculate initial product set for primes
        prod_set = {1}
        for p in primes:
            new_prods = set(prod_set)
            for v in prod_set:
                new_prods.add(p * v)
            prod_set = new_prods
            
        for c in shuffled_composites:
            if time.time() > deadline:
                break
            if can_add(c, prod_set):
                current_set.append(c)
                prod_set = update_prod_set(prod_set, c)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            # Optional: If we reach a very high size, we might want to break early or increase trials
            # But 18 is the max possible, and 12 is the seed. 14-16 is likely achievable.
            
    # Return the best set found
    # Ensure it's a list of ints
    return list(best_set)

# EVOLVE-BLOCK-END
