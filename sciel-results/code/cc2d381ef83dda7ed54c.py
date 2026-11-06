import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 145  # seconds, safety margin for 150s limit

    # 1. Generate Primes up to 1000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    all_primes = get_primes(1000)

    # 2. Select Prime Set
    # Strategy: Use primes that sum to <= 1.
    # The seed used 13..997. Sum ~0.92.
    # We will stick to the seed's set but optimize residues.
    primes_set = [p for p in all_primes if p >= 13]

    L_MAX = 100000
    L_CHECK_LIMIT = 50000 # Check up to 50k to save time, cap score at 10^5 if reached
    # Using a larger limit allows finding better L if coverage extends further.
    # However, calculating coverage for 50k is more expensive.
    # Let's stick to a reasonable limit that balances speed and potential.
    # If we cover 1..50000, score is at least 50000.
    # The seed score 1.0 corresponds to L=999? No, score is raw value.
    # Wait, the prompt says "The score is the largest L...".
    # "search score is value/seed". Seed value is likely 999 or similar.
    # If seed is 1.0, and value is L, then seed L is 1.0? No.
    # "1.0 equals the seed program".
    # If seed program returns L=999, then seed score is 999/999 = 1.0.
    # So we want to maximize L.
    
    best_L = 0
    best_classes = None
    n_primes = len(primes_set)

    # Precompute ranges for each prime to speed up updates?
    # Or just use list slicing?
    
    def evaluate_coverage(residues, L_limit=L_CHECK_LIMIT):
        """
        Returns L (max prefix covered)
        """
        cover_count = [0] * (L_limit + 1)
        for i, p in enumerate(primes_set):
            r = residues[i]
            start = r
            if start == 0:
                start = p
            elif start > L_limit:
                continue
            for n in range(start, L_limit + 1, p):
                cover_count[n] += 1
        
        # Find max L such that cover_count[1..L] > 0
        L = 0
        for n in range(1, L_limit + 1):
            if cover_count[n] == 0:
                break
            L = n
        return L, cover_count

    def get_holes(cover_count, L_limit=L_CHECK_LIMIT):
        holes = []
        for n in range(1, L_limit + 1):
            if cover_count[n] == 0:
                holes.append(n)
        return holes

    def init_residues(seed_style=True):
        res = [0] * n_primes
        if seed_style:
            for i, p in enumerate(primes_set):
                target = i + 1
                res[i] = target % p
        else:
            for i in range(n_primes):
                res[i] = random.randint(0, primes_set[i] - 1)
        return res

    # Main Search Loop
    for restart in range(50): # More restarts if fast
        if time.time() - start_time > DEADLINE:
            break
            
        if restart == 0:
            residues = init_residues(seed_style=True)
        else:
            residues = init_residues(seed_style=False)
            
        L, cover_count = evaluate_coverage(residues)
        
        if L > best_L:
            best_L = L
            best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
            
        # Local Search
        stuck_counter = 0
        # To speed up, we can limit the number of local search steps per restart
        # or just run until time is up.
        while time.time() - start_time < DEADLINE:
            if L >= L_CHECK_LIMIT:
                break
                
            # Find a hole
            holes = get_holes(cover_count)
            if not holes:
                break
            
            # Pick a random hole
            hole = random.choice(holes)
            
            # Pick a random prime index
            idx = random.randint(0, n_primes - 1)
            p = primes_set[idx]
            r_old = residues[idx]
            r_new = hole % p
            
            if r_old == r_new:
                continue
                
            # Apply change
            # Decrement
            start = r_old
            if start == 0:
                start = p
            elif start > L_CHECK_LIMIT:
                start = 0 # No change
            else:
                for n in range(start, L_CHECK_LIMIT + 1, p):
                    cover_count[n] -= 1
            
            # Increment
            start = r_new
            if start == 0:
                start = p
            elif start > L_CHECK_LIMIT:
                start = 0
            else:
                for n in range(start, L_CHECK_LIMIT + 1, p):
                    cover_count[n] += 1
            
            residues[idx] = r_new
            
            # Evaluate new L
            new_L = 0
            for n in range(1, L_CHECK_LIMIT + 1):
                if cover_count[n] == 0:
                    break
                new_L = n
            
            if new_L > L:
                L = new_L
                stuck_counter = 0
                if L > best_L:
                    best_L = L
                    best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
            else:
                # Revert
                # Decrement new
                start = r_new
                if start == 0:
                    start = p
                elif start > L_CHECK_LIMIT:
                    start = 0
                else:
                    for n in range(start, L_CHECK_LIMIT + 1, p):
                        cover_count[n] -= 1
                
                # Increment old
                start = r_old
                if start == 0:
                    start = p
                elif start > L_CHECK_LIMIT:
                    start = 0
                else:
                    for n in range(start, L_CHECK_LIMIT + 1, p):
                        cover_count[n] += 1
                
                residues[idx] = r_old
                stuck_counter += 1
                
                if stuck_counter > 2000:
                    break

    if best_classes is None:
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_set)]

    return best_classes

# EVOLVE-BLOCK-END
