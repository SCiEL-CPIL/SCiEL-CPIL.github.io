import time
import random
import math

# Precompute Smallest Prime Factor (SPF) for numbers up to 20000
MAX_VAL = 10000
MAX_SUM = 2 * MAX_VAL
SPF = list(range(MAX_SUM + 1))

def init_spf():
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if SPF[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if SPF[j] == j:
                    SPF[j] = i

init_spf()

def get_prime_factors(n):
    if n <= 1:
        return []
    factors = []
    while n > 1:
        p = SPF[n]
        factors.append(p)
        while n % p == 0:
            n //= p
    return factors

def construct():
    n = 12
    max_val = 10000
    deadline = time.time() + 50  # 50s limit to be safe
    
    def evaluate_set(S):
        primes = set()
        for i in range(n):
            for j in range(i+1, n):
                s = S[i] + S[j]
                # Factor s
                temp = s
                while temp > 1:
                    p = SPF[temp]
                    primes.add(p)
                    while temp % p == 0:
                        temp //= p
        return len(primes)

    def get_prime_counts(S):
        counts = {}
        for i in range(n):
            for j in range(i+1, n):
                s = S[i] + S[j]
                temp = s
                last_p = 0
                while temp > 1:
                    p = SPF[temp]
                    if p != last_p:
                        counts[p] = counts.get(p, 0) + 1
                        last_p = p
                    while temp % p == 0:
                        temp //= p
        return counts

    def update_counts(S, idx, old_val, new_val, counts):
        # Remove old_val contributions
        for j in range(n):
            if j == idx:
                continue
            s = old_val + S[j]
            temp = s
            last_p = 0
            while temp > 1:
                p = SPF[temp]
                if p != last_p:
                    counts[p] = counts.get(p, 0) - 1
                    last_p = p
                while temp % p == 0:
                    temp //= p
        # Add new_val contributions
        for j in range(n):
            if j == idx:
                continue
            s = new_val + S[j]
            temp = s
            last_p = 0
            while temp > 1:
                p = SPF[temp]
                if p != last_p:
                    counts[p] = counts.get(p, 0) + 1
                    last_p = p
                while temp % p == 0:
                    temp //= p
        return counts

    def current_cost(counts):
        return sum(1 for c in counts.values() if c > 0)

    best_set = None
    best_cost = float('inf')
    
    # Multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Initialization
        if restart == 0:
            S = list(range(1, 13))
        else:
            S = [random.randint(1, max_val) for _ in range(n)]
            # Ensure distinct
            while len(set(S)) < n:
                S[random.randint(0, n-1)] = random.randint(1, max_val)
        
        counts = get_prime_counts(S)
        cost = current_cost(counts)
        
        if cost < best_cost:
            best_cost = cost
            best_set = list(S)
            
        if best_cost == 2:
            break
            
        T = 1.0
        T_min = 0.01
        cooling = 0.9999
        
        while time.time() < deadline:
            if T < T_min:
                break
                
            idx = random.randint(0, n-1)
            old_val = S[idx]
            new_val = random.randint(1, max_val)
            
            if new_val == old_val:
                continue
                
            # Check distinctness
            if new_val in S:
                # Remove old, check if new is in rest?
                # Simpler: just skip if new_val is in S and new_val != old_val
                # But we are replacing old_val. So new_val must not be in S without old_val.
                # S contains old_val.
                # If new_val is in S, and new_val != old_val, it's a duplicate.
                continue
            
            # Calculate delta
            # We need to compute the change in cost.
            # Instead of full recount, we can compute the set of primes affected.
            # But counting unique primes is hard to do with deltas easily if a prime count goes 1->0 or 0->1.
            # Let's just update counts and compute new cost.
            
            new_counts = update_counts(S, idx, old_val, new_val, counts)
            new_cost = current_cost(new_counts)
            
            delta = new_cost - cost
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                S[idx] = new_val
                counts = new_counts
                cost = new_cost
                
                if cost < best_cost:
                    best_cost = cost
                    best_set = list(S)
                    if best_cost == 2:
                        break
                
            T *= cooling
            
        if best_cost == 2:
            break

    return best_set
