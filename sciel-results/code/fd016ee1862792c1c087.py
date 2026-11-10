import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    LIMIT = 10**18
    
    # Cache for scores to avoid recalculating
    score_cache = {}
    
    best_n = 2
    best_score = 0
    best_exp = [1] + [0]*17 # Default
    
    def get_n_from_exponents(exp):
        n = 1
        for i, e in enumerate(exp):
            if e > 0:
                n *= primes[i] ** e
                if n > LIMIT:
                    return -1 # Early exit
        return n

    def calculate_score(n, exp):
        key = tuple(exp)
        if key in score_cache:
            return score_cache[key]
            
        # Generate divisors
        divisors = [1]
        for i, e in enumerate(exp):
            if e > 0:
                p = primes[i]
                new_divs = []
                power = 1
                for _ in range(e + 1):
                    for d in divisors:
                        new_divs.append(d * power)
                    power *= p
                divisors = new_divs
        
        if not divisors:
            return 0
            
        divisors.sort()
        
        score = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                score += 1
                
        score_cache[key] = score
        return score

    def update_best(exp):
        nonlocal best_n, best_score, best_exp
        n = get_n_from_exponents(exp)
        if n < 2 or n > LIMIT:
            return
        score = calculate_score(n, exp)
        if score > best_score:
            best_score = score
            best_n = n
            best_exp = exp[:]

    # Initialize with P15
    init_exp = [0]*18
    for i in range(15):
        init_exp[i] = 1
    update_best(init_exp)
    
    current_exp = best_exp[:]
    
    # Search
    while time.time() < deadline:
        new_exp = current_exp[:]
        op = random.choice(['inc', 'dec_inc', 'swap', 'restart'])
        
        if op == 'inc':
            i = random.randint(0, 17)
            new_exp[i] += 1
        elif op == 'dec_inc':
            j = random.randint(0, 17)
            i = random.randint(0, 17)
            if new_exp[j] > 0:
                new_exp[j] -= 1
                new_exp[i] += 1
        elif op == 'swap':
            # Remove a large prime, add to small
            if any(e > 0 for e in new_exp[8:]):
                j = random.choice([i for i in range(8, 18) if new_exp[i] > 0])
                i = random.randint(0, 3)
                # Calculate delta
                ratio = primes[j] / primes[i]
                delta = int(math.log(ratio) / math.log(primes[i])) + 1
                new_exp[j] = 0
                new_exp[i] += delta
        else: # restart
            # Generate random valid exp
            rand_exp = [0]*18
            k = random.randint(10, 15)
            for i in range(k):
                rand_exp[i] = 1
            for i in range(4):
                rand_exp[i] += random.randint(0, 4)
            
            n_rand = get_n_from_exponents(rand_exp)
            while n_rand > LIMIT:
                max_idx = -1
                for i in range(17, -1, -1):
                    if rand_exp[i] > 0:
                        max_idx = i
                        break
                if max_idx == -1: break
                rand_exp[max_idx] -= 1
                n_rand = get_n_from_exponents(rand_exp)
            
            if n_rand >= 2:
                new_exp = rand_exp
        
        # Check validity
        n_new = get_n_from_exponents(new_exp)
        if n_new < 2 or n_new > LIMIT:
            continue
            
        # Evaluate
        score_new = calculate_score(n_new, new_exp)
        
        # Accept if better, or with small prob if not (to explore)
        if score_new > best_score:
            update_best(new_exp)
            current_exp = new_exp[:]
        elif random.random() < 0.05:
            # Move to new state even if not better, to escape local minima
            # But only if it's valid
            current_exp = new_exp[:]
            
    return best_n
