"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        f = 5
        while f * f <= n:
            if n % f == 0 or n % (f + 2) == 0:
                return False
            f += 6
        return True
    
    primes = [p for p in range(2, 1001) if is_prime(p)]
    
    # Initial solution: primes from 13 upward, covering i+1
    initial_primes = [p for p in primes if p >= 13]
    initial_residues = [(i + 1) % p for i, p in enumerate(initial_primes)]
    
    classes = list(zip(initial_primes, initial_residues))
    
    # Budget check with exact rationals
    budget = sum(Fraction(1, p) for p in initial_primes)
    assert budget <= 1, f"Initial budget exceeded: {float(budget)}"
    
    max_L = 100000
    
    def build_count(classes, max_L):
        count = [0] * (max_L + 1)
        for p, r in classes:
            start = r if r >= 1 else p
            for n in range(start, max_L + 1, p):
                count[n] += 1
        return count
    
    count = build_count(classes, max_L)
    
    def find_prefix(count):
        for n in range(1, max_L + 1):
            if count[n] == 0:
                return n - 1
        return max_L
    
    best_L = find_prefix(count)
    best_classes = [list(c) for c in classes]
    
    # Local search: optimize residues one at a time
    def local_search(classes, count, best_L, deadline):
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(len(classes)):
                p, r = classes[i]
                best_r = r
                best_L_local = best_L
                for new_r in range(p):
                    if new_r == r:
                        continue
                    # Update counts
                    old_start = r if r >= 1 else p
                    for n in range(old_start, max_L + 1, p):
                        count[n] -= 1
                    new_start = new_r if new_r >= 1 else p
                    for n in range(new_start, max_L + 1, p):
                        count[n] += 1
                    L_local = find_prefix(count)
                    if L_local > best_L_local:
                        best_L_local = L_local
                        best_r = new_r
                    # Revert
                    for n in range(new_start, max_L + 1, p):
                        count[n] -= 1
                    for n in range(old_start, max_L + 1, p):
                        count[n] += 1
                    if time.time() > deadline - 10:
                        break
                if best_r != r:
                    old_start = r if r >= 1 else p
                    for n in range(old_start, max_L + 1, p):
                        count[n] -= 1
                    new_start = best_r if best_r >= 1 else p
                    for n in range(new_start, max_L + 1, p):
                        count[n] += 1
                    classes[i] = (p, best_r)
                    best_L = best_L_local
                    improved = True
                if time.time() > deadline - 10:
                    break
        return best_L, classes
    
    best_L, classes = local_search(classes, count, best_L, deadline)
    
    # Iterated Local Search with multi-kick perturbations
    for iteration in range(50):
        if time.time() > deadline - 15:
            break
        
        # Multi-kick perturbation: randomly change k residues
        k = random.randint(2, 5)
        perturbed = [list(c) for c in classes]
        indices = random.sample(range(len(perturbed)), k)
        for idx in indices:
            p = perturbed[idx][0]
            perturbed[idx][1] = random.randint(0, p - 1)
        
        # Rebuild count for perturbed
        perturbed_count = build_count(perturbed, max_L)
        perturbed_L = find_prefix(perturbed_count)
        
        # Greedy repair: add/remove primes if helpful
        # Try to add a prime that covers the gap
        if perturbed_L < best_L:
            # Try adding a prime not already used
            used_primes = set(p for p, r in perturbed)
            gap = perturbed_L + 1
            if gap <= max_L:
                for p in primes:
                    if p in used_primes:
                        continue
                    # Check budget
                    current_budget = sum(Fraction(1, pp) for pp in [pp for pp, rr in perturbed])
                    if current_budget + Fraction(1, p) > 1:
                        continue
                    # Find best residue to cover gap
                    best_r = gap % p
                    # Check if this residue covers the gap
                    if (gap - best_r) % p == 0 and best_r < p:
                        # Add this prime
                        perturbed.append([p, best_r])
                        # Rebuild and check
                        new_count = build_count(perturbed, max_L)
                        new_L = find_prefix(new_count)
                        if new_L > perturbed_L:
                            perturbed_L = new_L
                            perturbed_count = new_count
                        break
        
        # Local search on perturbed
        perturbed_L, perturbed = local_search(perturbed, perturbed_count, perturbed_L, deadline)
        
        # Accept if better
        if perturbed_L > best_L:
            best_L = perturbed_L
            best_classes = [list(c) for c in perturbed]
            classes = perturbed
            count = perturbed_count
    
    return [[p, r] for p, r in best_classes]
# EVOLVE-BLOCK-END
