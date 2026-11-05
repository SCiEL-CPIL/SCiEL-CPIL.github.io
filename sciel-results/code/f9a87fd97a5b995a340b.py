import time
import random
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    MAX_L = 100000
    
    def optimize(prime_set, time_limit_per_set=10.0):
        # Check budget
        s = sum(Fraction(1, p) for p in prime_set)
        if s > 1:
            return 0, []
            
        P_sorted = sorted(prime_set)
        if not P_sorted:
            return 0, []
            
        residues = {p: 0 for p in P_sorted}
        
        # Use int32 for counts
        counts = np.zeros(MAX_L + 1, dtype=np.int32)
        
        # Initial application
        for p in P_sorted:
            r = residues[p]
            counts[r::p] += 1
            
        counts[0] = 10 # Ensure 0 is covered

        # Find initial L
        def find_first_gap(start_idx=1):
            # Scan from start_idx to MAX_L
            limit = min(MAX_L, start_idx + 5000)
            if limit <= start_idx:
                return MAX_L
            zero_indices = np.where(counts[start_idx:limit] == 0)[0]
            if len(zero_indices) == 0:
                # Check if the rest is covered
                if start_idx + 5000 < MAX_L:
                    if np.all(counts[start_idx + 5000:MAX_L] > 0):
                        return MAX_L
                    else:
                        zero_indices_rest = np.where(counts[start_idx + 5000:MAX_L] == 0)[0]
                        if len(zero_indices_rest) == 0:
                            return MAX_L
                        return int(start_idx + 5000 + zero_indices_rest[0])
                else:
                    return MAX_L
            return int(start_idx + zero_indices[0])

        current_L = find_first_gap(1)
        
        no_improve_count = 0
        max_no_improve = 20
        start_opt_time = time.time()
        
        while time.time() < start_opt_time + time_limit_per_set and time.time() < deadline - 5:
            if current_L == MAX_L:
                break
                
            n = current_L # First uncovered
            best_move = None
            best_L = current_L
            
            # Try changing each prime's residue to cover n
            for p in P_sorted:
                r_new = n % p
                if residues[p] == r_new:
                    continue
                
                r_old = residues[p]
                
                # Apply move
                counts[r_old::p] -= 1
                counts[r_new::p] += 1
                
                # Check if n is covered
                if counts[n] > 0:
                    new_L = find_first_gap(n + 1)
                else:
                    new_L = find_first_gap(n + 1)
                
                # Revert move
                counts[r_new::p] -= 1
                counts[r_old::p] += 1
                
                if new_L > best_L:
                    best_L = new_L
                    best_move = (p, r_old, r_new)
                    
            if best_move:
                p, r_old, r_new = best_move
                residues[p] = r_new
                counts[r_old::p] -= 1
                counts[r_new::p] += 1
                current_L = best_L
                no_improve_count = 0
            else:
                no_improve_count += 1
                # If stuck, perform a random perturbation
                if no_improve_count > max_no_improve:
                    num_perturb = random.randint(1, 2)
                    for _ in range(num_perturb):
                        p_rand = random.choice(P_sorted)
                        r_old = residues[p_rand]
                        r_new = random.randint(0, p_rand - 1)
                        if r_old == r_new:
                            continue
                        counts[r_old::p_rand] -= 1
                        counts[r_new::p_rand] += 1
                        residues[p_rand] = r_new
                    
                    current_L = find_first_gap(1)
                    no_improve_count = 0

        classes = [[p, residues[p]] for p in P_sorted]
        return current_L, classes

    # Generate Candidates
    candidates = []
    
    def build_set(base, skip):
        base_set = set(base)
        skip_set = set(skip)
        s = sum(Fraction(1, p) for p in base_set)
        if s > 1:
            return []
        
        final_set = list(base_set)
        for p in primes:
            if p in base_set or p in skip_set:
                continue
            if s + Fraction(1, p) <= 1:
                final_set.append(p)
                s += Fraction(1, p)
        return sorted(final_set)

    # High priority sets
    candidates.append(build_set([2, 3, 5], []))
    candidates.append(build_set([2, 3], []))
    candidates.append(build_set([2, 5], []))
    candidates.append(build_set([3, 5], []))
    candidates.append(build_set([2], []))
    candidates.append(build_set([], []))
    
    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    best_L = 0
    best_classes = []
    
    # Time management
    for i, P in enumerate(unique_candidates):
        if time.time() > deadline - 10:
            break
        if not P:
            continue
            
        remaining_time = deadline - 10 - time.time()
        candidates_left = len(unique_candidates) - i
        time_per_candidate = remaining_time / candidates_left
        time_per_candidate = min(time_per_candidate, 15.0)
        time_per_candidate = max(time_per_candidate, 2.0)
        
        L, classes = optimize(P, time_limit_per_set=time_per_candidate)
        if L > best_L:
            best_L = L
            best_classes = classes
            
    return best_classes
