import time
import random
from fractions import Fraction

def _get_primes(limit=1000):
    """Return list of primes up to limit."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    t0 = time.time()
    MAX_L = 100000
    deadline = t0 + 115  # Safety margin
    
    primes = _get_primes(1000)
    prime_set = set(primes)
    
    # State
    count = [0] * (MAX_L + 2)
    classes = []  # list of [p, r]
    used = set()
    current_sum = Fraction(0)
    L = 0
    
    best_L = 0
    best_classes = []
    best_sum = Fraction(0)
    
    def update_count(p, r, delta):
        # delta = 1 for add, -1 for remove
        start = r if r >= 1 else p
        for n in range(start, MAX_L + 1, p):
            count[n] += delta
            
    def get_L():
        # Find largest L such that 1..L are covered (count[n] > 0)
        for n in range(1, MAX_L + 1):
            if count[n] == 0:
                return n - 1
        return MAX_L

    def save_best():
        nonlocal best_L, best_classes, best_sum
        if L > best_L:
            best_L = L
            best_classes = [c[:] for c in classes]
            best_sum = current_sum

    # 1. Greedy Initialization
    while L < MAX_L:
        if time.time() > deadline - 5:
            break
            
        G = L + 1
        best_p = None
        best_r = None
        best_new_L = L
        
        # Try adding a new prime
        for p in primes:
            if p in used:
                continue
            new_sum = current_sum + Fraction(1, p)
            if new_sum > Fraction(1):
                continue
            
            r = G % p
            # Simulate Add
            update_count(p, r, 1)
            new_L = get_L()
            update_count(p, r, -1)
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_p = p
                best_r = r
        
        if best_p is None:
            break
        
        # Apply best
        classes.append([best_p, best_r])
        used.add(best_p)
        current_sum += Fraction(1, best_p)
        update_count(best_p, best_r, 1)
        L = best_new_L
        save_best()

    # 2. Local Search
    random.seed(42)
    
    while time.time() < deadline - 2:
        G = L + 1
        improved = False
        
        best_candidate = None
        best_new_L = L
        
        # 1. Residue Change
        # Iterate through existing classes
        for idx in range(len(classes)):
            p, r_old = classes[idx]
            r_new = G % p
            if r_new == r_old:
                continue
            
            # Simulate Change
            update_count(p, r_old, -1)
            update_count(p, r_new, 1)
            new_L = get_L()
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_candidate = ('change', idx, p, r_new)
            
            # Revert
            update_count(p, r_new, -1)
            update_count(p, r_old, 1)
            
            # Time check inside loop
            if time.time() > deadline - 2:
                break

        # 2. Add
        if time.time() < deadline - 2:
            for p in primes:
                if p in used:
                    continue
                new_sum = current_sum + Fraction(1, p)
                if new_sum > Fraction(1):
                    continue
                
                r = G % p
                # Simulate Add
                update_count(p, r, 1)
                new_L = get_L()
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_candidate = ('add', p, r)
                
                # Revert
                update_count(p, r, -1)
                
                if time.time() > deadline - 2:
                    break

        if best_candidate:
            # Apply
            if best_candidate[0] == 'change':
                _, idx, p, r_new = best_candidate
                r_old = classes[idx][1]
                classes[idx] = [p, r_new]
                update_count(p, r_old, -1)
                update_count(p, r_new, 1)
            elif best_candidate[0] == 'add':
                _, p, r = best_candidate
                classes.append([p, r])
                used.add(p)
                current_sum += Fraction(1, p)
                update_count(p, r, 1)
            
            L = best_new_L
            improved = True
            save_best()
            
        if not improved:
            # Stuck. Try a random perturbation to escape local optimum.
            # We accept the perturbation even if it lowers L, hoping to find a better path later.
            # But we must keep the best found so far.
            if classes:
                idx = random.randint(0, len(classes) - 1)
                p = classes[idx][0]
                r_old = classes[idx][1]
                r_new = random.randint(0, p - 1)
                if r_new != r_old:
                    classes[idx] = [p, r_new]
                    update_count(p, r_old, -1)
                    update_count(p, r_new, 1)
                    L = get_L()
                    # We don't save best here unless it's better, which is unlikely
                    # but get_L is called so L is updated.
            else:
                break

    # Return best found
    # Ensure budget is exact
    # The code uses Fraction, so it is exact.
    
    if not best_classes:
        # Fallback
        return [[2, 1]]
        
    return best_classes

if __name__ == "__main__":
    r = construct()
    from fractions import Fraction
    s = Fraction(0)
    for p, rr in r:
        s += Fraction(1, p)
    print("num classes", len(r), "sum", s)
