import math
import time
import random

def construct():
    N = 511
    # Precompute prime factors mask for each number 1..511
    # Primes < 512
    primes = []
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    
    # Map prime to index for bitmask
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Precompute mask and score for each number
    masks = [0] * (N + 1)
    scores = [0.0] * (N + 1)
    for a in range(1, N + 1):
        scores[a] = 1.0 / (512 - a)
        x = a
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                masks[a] |= (1 << prime_to_idx[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            masks[a] |= (1 << prime_to_idx[x])
    
    # Initial set: All primes
    # For each prime, we might want to use its largest power, but starting with primes is a safe baseline
    # Actually, let's start with the largest power of each prime
    initial_set = []
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        initial_set.append(power)
    
    def get_score(s):
        return sum(scores[a] for a in s)
    
    def is_valid_swap(current_set, a_old, a_new):
        # a_new must be coprime to all in current_set except a_old
        # And a_new not in current_set
        if a_new in current_set:
            return False
        mask_new = masks[a_new]
        # Check against all others
        for a in current_set:
            if a == a_old:
                continue
            if mask_new & masks[a]:
                return False
        return True

    best_set = list(initial_set)
    best_score = get_score(best_set)
    
    start_time = time.time()
    deadline = start_time + 100.0 # 20s margin
    
    # Local Search
    # We will use a simple hill climbing with random restarts
    # Move types:
    # 1. Swap a in S with b not in S
    # 2. Merge a, b in S into c (multiple of a*b)
    # 3. Split a in S into its prime factors (or subset)
    
    all_candidates = list(range(1, N + 1))
    
    for _ in range(50): # 50 restarts/iterations
        if time.time() > deadline:
            break
            
        current_set = list(best_set)
        current_score = best_score
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            
            # Try all swaps
            for i in range(len(current_set)):
                a_old = current_set[i]
                # Try a few random candidates to swap in
                for _ in range(10):
                    a_new = random.choice(all_candidates)
                    if a_new == a_old:
                        continue
                    if is_valid_swap(current_set, a_old, a_new):
                        new_score = current_score - scores[a_old] + scores[a_new]
                        if new_score > current_score:
                            current_set[i] = a_new
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(current_set)
            
            # Try Merges
            # For each pair a, b in current_set, see if we can merge
            n = len(current_set)
            for i in range(n):
                for j in range(i + 1, n):
                    a = current_set[i]
                    b = current_set[j]
                    # They are coprime
                    # Find largest c < 512 such that c is multiple of a*b?
                    # Not necessarily a*b. c must have prime factors superset of a and b.
                    # Simplest: c is a multiple of lcm(a,b) = a*b.
                    base = a * b
                    if base > N:
                        # Can't merge into a multiple of a*b
                        # But maybe a different combination?
                        # For now, skip if base > N
                        continue
                    # Find largest multiple of base <= N
                    c = (N // base) * base
                    if c == a or c == b:
                        continue
                    # Check if c is valid replacement for {a, b}
                    # c must be coprime to all other elements
                    valid = True
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if masks[c] & masks[current_set[k]]:
                            valid = False
                            break
                    if valid:
                        new_score = current_score - scores[a] - scores[b] + scores[c]
                        if new_score > current_score:
                            # Replace a, b with c
                            # Remove b, replace a with c
                            current_set[i] = c
                            current_set.pop(j)
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(current_set)
                            break # Restart pair loop as indices changed
                    # Also try splitting?
                    # If a is composite, try replacing a with its prime factors
                    # This is handled by swaps if we consider adding primes and removing a
                    # But swaps are 1-for-1. Split is 1-for-k.
                    # We can simulate split by:
                    # 1. Remove a
                    # 2. Add prime factors
                    # This changes set size. My current loop structure assumes fixed size for swaps.
                    # Let's add a specific split check.
            
            # Try Splits
            for i in range(len(current_set)):
                a = current_set[i]
                if a in is_prime and is_prime[a]:
                    continue # Primes can't be split
                # Get prime factors of a
                factors = []
                x = a
                for p in primes:
                    if p * p > x:
                        break
                    if x % p == 0:
                        factors.append(p)
                        while x % p == 0:
                            x //= p
                if x > 1:
                    factors.append(x)
                
                # Check if we can add all factors to the set (replacing a)
                # New set: (current_set - {a}) + factors
                # Must be pairwise coprime
                # Factors are distinct primes, so they are coprime to each other.
                # They must be coprime to (current_set - {a})
                valid = True
                for k in range(len(current_set)):
                    if k == i:
                        continue
                    other = current_set[k]
                    for f in factors:
                        if masks[f] & masks[other]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    new_score = current_score - scores[a] + sum(scores[f] for f in factors)
                    if new_score > current_score:
                        # Replace a with factors
                        current_set.pop(i)
                        current_set.extend(factors)
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(current_set)
                        break # Restart loop

    return best_set
