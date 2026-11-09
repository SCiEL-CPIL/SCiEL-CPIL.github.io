import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Generate primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    
    # We will search up to a cap. 
    # The problem states cap 10^5, but with sum(1/p) <= 1, 
    # the maximum possible L is limited. 
    # Heuristically, L is likely < 2000. 
    # Let's set CAP to 5000 to be safe but fast.
    CAP = 5000
    
    # State
    covered = bytearray(CAP + 1)  # 1-indexed, covered[i] is 1 if i is covered
    used_primes = set()
    current_sum = Fraction(0, 1)
    current_L = 0
    
    # Precompute prime indices for quick access? Not needed, list is small.
    
    def try_add_prime(p, r):
        """
        Simulates adding prime p with residue r.
        Returns the new covered length if this prime is added,
        or -1 if it doesn't extend the current length.
        Also updates a temporary covered state? 
        No, we just simulate the walk.
        """
        # The new prime covers x if x % p == r
        # We start from current_L + 1
        # We need to check if the sequence current_L+1, current_L+2, ... is covered
        # by (existing covered) OR (new prime)
        
        y = current_L + 1
        # If the new prime doesn't cover the first gap, it's useless for extending the prefix
        if y % p != r:
            return current_L
        
        new_L = y
        y += 1
        while y <= CAP:
            if covered[y]:
                new_L = y
                y += 1
            elif y % p == r:
                new_L = y
                y += 1
            else:
                break
        return new_L

    # Greedy Construction Loop
    while time.time() < deadline:
        best_p = None
        best_r = None
        best_new_L = current_L
        best_cost = Fraction(0, 1) # We want to minimize cost for same L, so maximize p? 
                                    # Actually, for same L, smaller p is "better" for future? 
                                    # No, smaller p costs more. 
                                    # If L is same, we prefer the one that leaves more budget? 
                                    # So prefer larger p (smaller 1/p).
        
        for p in primes:
            if p in used_primes:
                continue
            
            # Check budget
            cost = Fraction(1, p)
            if current_sum + cost > Fraction(1, 1):
                continue
                
            # To extend the prefix, the new prime MUST cover current_L + 1
            # So r is fixed
            r = (current_L + 1) % p
            
            # Simulate
            new_L = try_add_prime(p, r)
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_p = p
                best_r = r
                best_cost = cost
            elif new_L == best_new_L and new_L > current_L:
                # Tie-breaker: Prefer larger p (cheaper) to save budget
                # Or prefer smaller p? 
                # Let's stick with larger p (cheaper) for now.
                if p > best_p: # best_p is initialized to None, handle carefully
                    best_p = p
                    best_r = r
                    best_cost = cost
                elif best_p is None:
                    best_p = p
                    best_r = r
                    best_cost = cost

        if best_p is None or best_new_L <= current_L:
            # No improvement found
            break
            
        # Apply the best choice
        used_primes.add(best_p)
        current_sum += best_cost
        current_L = best_new_L
        
        # Update covered array
        # Mark all x in 1..CAP such that x % best_p == best_r
        # Optimization: Only mark from 1 to CAP
        if best_r == 0:
            start = best_p
        else:
            start = best_r
            
        for x in range(start, CAP + 1, best_p):
            covered[x] = 1

    # Construct result
    # We need to return [[p, r], ...]
    # We have used_primes, but we didn't store the residues in a dict during the loop.
    # We need to re-derive or store them.
    # Let's re-run the greedy selection to store residues, or just store them in the loop.
    
    # Re-do with storage
    covered = bytearray(CAP + 1)
    used_primes = set()
    current_sum = Fraction(0, 1)
    current_L = 0
    final_classes = []
    
    while time.time() < deadline:
        best_p = None
        best_r = None
        best_new_L = current_L
        
        for p in primes:
            if p in used_primes:
                continue
            
            cost = Fraction(1, p)
            if current_sum + cost > Fraction(1, 1):
                continue
                
            r = (current_L + 1) % p
            
            new_L = try_add_prime(p, r)
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_p = p
                best_r = r
            elif new_L == best_new_L and new_L > current_L:
                if best_p is None or p > best_p:
                    best_p = p
                    best_r = r

        if best_p is None or best_new_L <= current_L:
            break
            
        used_primes.add(best_p)
        current_sum += Fraction(1, best_p)
        current_L = best_new_L
        final_classes.append([best_p, best_r])
        
        if best_r == 0:
            start = best_p
        else:
            start = best_r
            
        for x in range(start, CAP + 1, best_p):
            covered[x] = 1

    return final_classes
