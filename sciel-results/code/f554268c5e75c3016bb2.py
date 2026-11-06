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

def _can_add(r, m, selected):
    for r2, m2 in selected:
        if not _disjoint(r, m, r2, m2):
            return False
    return True

def _find_residue(m, selected):
    for r in range(m):
        if _can_add(r, m, selected):
            return r
    return -1

def _local_search(selected, all_moduli, deadline):
    selected_set = {m for _, m in selected}
    unselected = [m for m in all_moduli if m not in selected_set]
    
    # Pass 1: Try to add
    while time.time() < deadline:
        added_any = False
        to_remove = []
        for m in unselected:
            if time.time() > deadline: break
            r = _find_residue(m, selected)
            if r != -1:
                selected.append([r, m])
                selected_set.add(m)
                to_remove.append(m)
                added_any = True
        for m in to_remove:
            unselected.remove(m)
        if not added_any:
            break
            
    # Pass 2: Try swaps (limited to first few unselected)
    if time.time() < deadline:
        for m in unselected[:10]:
            if time.time() > deadline: break
            r = _find_residue(m, selected)
            if r != -1:
                continue
            
            for i, (r2, m2) in enumerate(selected):
                if time.time() > deadline: break
                selected_temp = selected[:i] + selected[i+1:]
                r_new = _find_residue(m, selected_temp)
                if r_new != -1:
                    selected[i] = [r_new, m]
                    selected_set.remove(m2)
                    selected_set.add(m)
                    unselected.remove(m)
                    unselected.append(m2)
                    break
                    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    best_solution = []
    best_count = 0
    
    # Helper to count prime factors
    def omega(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
        
    primes = [m for m in moduli if _is_prime(m)]
    composites = [m for m in moduli if not _is_prime(m)]
    
    strategies = [
        primes + sorted(composites, key=omega),
        sorted(composites, key=omega) + primes,
        sorted(moduli, key=omega),
        sorted(moduli, key=omega, reverse=True),
        primes + sorted(composites, reverse=True),
        sorted(composites, reverse=True) + primes,
    ]
    
    # Add random shuffles
    for _ in range(10):
        random.shuffle(moduli)
        strategies.append(moduli[:])
        
    for order in strategies:
        if time.time() > deadline:
            break
            
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            r = _find_residue(m, selected)
            if r != -1:
                selected.append([r, m])
        
        # Apply local search
        selected = _local_search(selected, moduli, deadline)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_solution = selected
            
    if not best_solution:
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
        
    return best_solution
