import time
import random

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 60s. Use 55s to be safe.
    deadline = start_time + 55

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def is_valid(s):
        n = len(s)
        if n == 0:
            return True
        prods = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= s[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    all_numbers = list(range(2, 129))
    primes = [x for x in all_numbers if is_prime(x)]
    composites = [x for x in all_numbers if not is_prime(x)]
    
    best_seq = primes[:]
    best_len = len(primes)
    
    # Multi-Start Greedy Strategy
    # Different starting configurations to explore different local optima
    
    # Start 1: Primes (known valid, high density of unique factors)
    seeds = [primes[:]]
    
    # Start 2: Smallest numbers (might pack more tightly?)
    seeds.append(all_numbers[:20])
    
    # Start 3: Random valid subset
    random.seed(42)
    rand_seed = random.sample(all_numbers, 15)
    rand_seed.sort()
    # Ensure the random seed is valid, if not, shrink it
    if not is_valid(rand_seed):
        # Fallback to a smaller valid random set
        rand_seed = []
        for x in sorted(random.sample(all_numbers, 15)):
            test = rand_seed + [x]
            # Insert in sorted order
            pos = 0
            while pos < len(test) and test[pos] < x:
                pos += 1
            test.insert(pos, x)
            if is_valid(test):
                rand_seed = test
            else:
                test.pop(pos)
    if rand_seed:
        seeds.append(rand_seed)
        
    # Start 4: Only even numbers? (Likely invalid due to powers of 2, but worth a try with filtering)
    # Actually, powers of 2 are dangerous. Let's try a mix.
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
            
        # Ensure seed is sorted and valid
        seed.sort()
        if not is_valid(seed):
            continue
            
        current_seq = seed[:]
        current_len = len(current_seq)
        
        # Greedy Insertion of remaining numbers
        remaining = [x for x in all_numbers if x not in set(current_seq)]
        remaining.sort()
        
        # Try to insert each remaining number
        # We iterate through remaining numbers. 
        # To add variety, we could shuffle 'remaining' but we must insert in sorted position.
        # The order we *check* them can be shuffled.
        
        check_order = remaining[:]
        random.shuffle(check_order)
        
        for c in check_order:
            if time.time() > deadline - 5:
                break
                
            # Find position to insert c to keep sequence sorted
            pos = 0
            while pos < len(current_seq) and current_seq[pos] < c:
                pos += 1
            
            # Check if c is already there (shouldn't be, but safe)
            if pos < len(current_seq) and current_seq[pos] == c:
                continue
                
            new_seq = current_seq[:pos] + [c] + current_seq[pos:]
            
            if is_valid(new_seq):
                current_seq = new_seq
                current_len += 1
        
        # Update best
        if current_len > best_len:
            best_len = current_len
            best_seq = current_seq[:]
            
    return best_seq
