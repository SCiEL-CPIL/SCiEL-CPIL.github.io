import math
import random
import time

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    primes = [m for m in moduli if _is_prime(m)]
    composites = [m for m in moduli if not _is_prime(m)]
    
    best_solution = []
    best_count = 0
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Primes first (random residues), Composites Descending
    strategies.append(('primes_rand_comp_desc', primes, sorted(composites, reverse=True)))
    
    # Strategy 2: Primes first (residue 0), Composites Descending
    # We can't easily force residue 0 in the generic loop, so we just rely on the random restarts
    # Or we can modify the loop to try specific residue assignments for primes.
    
    # Strategy 3: All moduli sorted by "difficulty" (ascending number of divisors)
    def num_divisors(n):
        count = 0
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    strategies.append(('divisors_asc', sorted(moduli, key=num_divisors), []))
    
    # Strategy 4: Random order
    for _ in range(5):
        random.shuffle(moduli)
        strategies.append(('rand', moduli, []))
        
    for name, p_list, c_list in strategies:
        if time.time() > deadline:
            break
            
        # Re-initialize for this strategy
        # We need to handle the order.
        # If p_list is provided, use it, then c_list.
        # Else use the single list.
        
        if c_list:
            order = p_list + c_list
        else:
            order = p_list
            
        selected = []
        
        for m in order:
            if time.time() > deadline:
                break
                
            found = False
            # Try residues in random order to break symmetry
            residues = list(range(m))
            random.shuffle(residues)
            
            for r in residues:
                valid = True
                for r2, m2 in selected:
                    if not _disjoint(r, m, r2, m2):
                        valid = False
                        break
                if valid:
                    selected.append([r, m])
                    found = True
                    break
            
            if found:
                pass # added
            else:
                pass # skipped
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_solution = selected
            
    # If we didn't find anything (should not happen), return seed
    if not best_solution:
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
        
    return best_solution
