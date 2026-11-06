import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
        
    primes = sieve(1000)
    
    # Seed initialization: use primes from 13 upwards
    # Find index of 13
    start_idx = primes.index(13)
    initial_primes = primes[start_idx:]
    
    # Initial residues: cover 1, 2, 3...
    # Seed logic: i-th prime covers i+1?
    # "The i-th prime from 13 upward covers the integer i"
    # Wait, seed code: return [[p, (i + 1) % p] for i, p in enumerate(primes)]
    # So prime[0] (13) covers 1. prime[1] (17) covers 2.
    
    classes = [[p, (i + 1) % p] for i, p in enumerate(initial_primes)]
    
    # Helper to calculate coverage length
    def get_coverage_length(cls, limit=20000):
        covered = [False] * (limit + 1)
        for p, r in cls:
            for k in range(r if r > 0 else p, limit + 1, p):
                # 1..L, so k >= 1
                if k >= 1:
                    covered[k] = True
        
        L = 0
        for i in range(1, limit + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    current_L = get_coverage_length(classes)
    
    # Local Search
    no_improve_count = 0
    max_no_improve = 10000
    
    while time.time() < deadline:
        if current_L >= 100000:
            break
            
        # Try to improve by changing a single residue
        improved = False
        # Shuffle classes to avoid bias
        cls_indices = list(range(len(classes)))
        random.shuffle(cls_indices)
        
        for idx in cls_indices:
            p, r = classes[idx]
            # Target the first gap
            # We don't know the first gap exactly without checking, but get_coverage_length gives L.
            # The first gap is L + 1.
            target = current_L + 1
            
            # Try to cover 'target' with this prime
            new_r = target % p
            
            if new_r == r:
                continue
                
            # Save old state
            old_r = r
            classes[idx][1] = new_r
            
            new_L = get_coverage_length(classes)
            
            if new_L > current_L:
                current_L = new_L
                improved = True
                no_improve_count = 0
                break
            else:
                # Revert
                classes[idx][1] = old_r
        
        if not improved:
            no_improve_count += 1
            # If stuck, try a random move to escape local minimum?
            # Or try adding/removing primes?
            # Let's try a random perturbation if stuck for a while
            if no_improve_count > 500:
                # Randomly change a residue
                idx = random.randint(0, len(classes) - 1)
                p, r = classes[idx]
                classes[idx][1] = random.randint(0, p - 1)
                current_L = get_coverage_length(classes)
                no_improve_count = 0
                
            # If stuck a lot, try swapping a prime
            if no_improve_count > 5000:
                # Find an unused prime smaller than the largest used prime
                used_primes = set(p for p, r in classes)
                unused_primes = [p for p in primes if p not in used_primes]
                # Sort by size, prefer smaller
                unused_primes.sort()
                
                if unused_primes:
                    # Try to swap the largest used prime with the smallest unused
                    # Check budget
                    sum_inv = sum(1.0 / p for p, r in classes)
                    max_p_used = max(p for p, r in classes)
                    min_p_unused = unused_primes[0]
                    
                    if 1.0 / max_p_used > 1.0 / min_p_unused:
                        # Swapping to smaller prime increases sum, might violate budget
                        # Check if new sum <= 1
                        new_sum = sum_inv - 1.0 / max_p_used + 1.0 / min_p_unused
                        if new_sum <= 1.0:
                            # Perform swap
                            # Remove max_p_used
                            for i, (p, r) in enumerate(classes):
                                if p == max_p_used:
                                    classes.pop(i)
                                    break
                            # Add min_p_unused with random residue
                            new_r = random.randint(0, min_p_unused - 1)
                            classes.append([min_p_unused, new_r])
                            current_L = get_coverage_length(classes)
                            no_improve_count = 0

    return classes
