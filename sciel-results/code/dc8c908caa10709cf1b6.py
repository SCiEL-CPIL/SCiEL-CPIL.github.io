import time
import random
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

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
        
        # Use int32 for counts to save memory and speed up operations
        counts = np.zeros(MAX_L + 1, dtype=np.int32)
        
        # Initial application
        for p in P_sorted:
            r = residues[p]
            counts[r::p] += 1
            
        counts[0] = 10 # Ensure 0 is covered, though irrelevant for L>=1
        
        def get_L_from_counts(c):
            # Find first index i >= 1 where c[i] == 0
            # c[1:] is the slice for 1..MAX_L
            # If all > 0, return MAX_L
            # Else return first index where 0 + 1
            zero_indices = np.where(c[1:] == 0)[0]
            if len(zero_indices) == 0:
                return MAX_L
            return int(zero_indices[0]) + 1

        current_L = get_L_from_counts(counts)
        
        # Local Search Loop with Perturbation
        no_improve_count = 0
        max_no_improve = 50
        start_opt_time = time.time()
        
        while time.time() < start_opt_time + time_limit_per_set and time.time() < deadline - 2:
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
                
                new_L = get_L_from_counts(counts)
                
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
                # If stuck, perform a random perturbation to escape local minimum
                if no_improve_count > max_no_improve:
                    # Randomly change residues of 1-2 primes
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
                    
                    current_L = get_L_from_counts(counts)
                    no_improve_count = 0 # Reset counter after perturbation

        classes = [[p, residues[p]] for p in P_sorted]
        return current_L, classes

    # Generate Candidates
    candidates = []
    
    # Helper to build a set given base and skip
    def build_set(base, skip):
        base_set = set(base)
        skip_set = set(skip)
        s = sum(Fraction(1, p) for p in base_set)
        if s > 1:
            return []
        
        final_set = list(base_set)
        # Fill with smallest available primes not in base or skip
        for p in primes:
            if p in base_set or p in skip_set:
                continue
            if s + Fraction(1, p) <= 1:
                final_set.append(p)
                s += Fraction(1, p)
        return sorted(final_set)

    # 1. Standard Greedy (Smallest primes)
    candidates.append(build_set([], []))
    
    # 2. Force 2
    candidates.append(build_set([2], []))
    
    # 3. Force 2, 3
    candidates.append(build_set([2, 3], []))
    
    # 4. Force 2, 5
    candidates.append(build_set([2, 5], []))
    
    # 5. Force 3, 5
    candidates.append(build_set([3, 5], []))
    
    # 6. Force 2, 3, 5
    candidates.append(build_set([2, 3, 5], []))
    
    # 7. Skip 3
    candidates.append(build_set([], [3]))
    
    # 8. Skip 5
    candidates.append(build_set([], [5]))
    
    # 9. Force 2, Skip 3
    candidates.append(build_set([2], [3]))
    
    # 10. Force 2, Skip 5
    candidates.append(build_set([2], [5]))
    
    # 11. Force 2, 3, Skip 5
    candidates.append(build_set([2, 3], [5]))
    
    # 12. Random sets
    random.seed(42)
    for _ in range(10):
        s = Fraction(0)
        res = []
        available = list(primes)
        random.shuffle(available)
        for p in available:
            if s + Fraction(1, p) <= 1:
                res.append(p)
                s += Fraction(1, p)
        candidates.append(res)

    best_L = 0
    best_classes = []
    
    # Sort candidates by size or some heuristic? 
    # Let's just iterate. We have ~15 candidates.
    # Time budget per candidate: (140 - 5) / 15 ~ 9 seconds.
    # But some candidates might be better. 
    # Let's run them all with a time slice.
    
    # To maximize score, we should prioritize candidates that are likely to be good.
    # Sets with small primes (2,3,5) usually cover more.
    # Let's run the "forced small" ones first.
    
    # Reorder: Put high-potential sets first
    priority_candidates = [
        build_set([2, 3, 5], []),
        build_set([2, 3], []),
        build_set([2, 5], []),
        build_set([3, 5], []),
        build_set([2], []),
        build_set([], []),
        build_set([2, 3], [5]),
        build_set([2], [3]),
        build_set([2], [5]),
        build_set([], [3]),
        build_set([], [5])
    ]
    
    # Add random ones at the end
    for c in candidates:
        if c not in priority_candidates:
            priority_candidates.append(c)
    
    # Remove duplicates (by set content)
    unique_candidates = []
    seen = set()
    for c in priority_candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    # Time management:
    # Total time ~ 135s.
    # Number of candidates ~ 11 + 10 random = 21.
    # Time per candidate ~ 6s.
    
    for P in unique_candidates:
        if time.time() > deadline - 10:
            break
        if not P:
            continue
            
        # Allocate time based on remaining time and candidates left
        remaining_time = deadline - 10 - time.time()
        candidates_left = len(unique_candidates) - unique_candidates.index(P)
        time_per_candidate = remaining_time / candidates_left
        # Cap time per candidate to 10s to ensure we try more if they finish fast
        time_per_candidate = min(time_per_candidate, 10.0)
        # Ensure at least 1s
        time_per_candidate = max(time_per_candidate, 1.0)
        
        L, classes = optimize(P, time_limit_per_set=time_per_candidate)
        if L > best_L:
            best_L = L
            best_classes = classes
            
    return best_classes

