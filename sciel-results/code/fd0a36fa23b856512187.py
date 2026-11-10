"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 5000
DEADLINE = 315.0  # safety margin before 330s kill


def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def construct():
    start_time = time.time()
    
    all_primes = _primes(N + 1)
    prime_set = set(all_primes)
    
    best_points = None
    best_size = 0
    
    def _evaluate_and_refine(points_set, label):
        """Refine a starting set: repair admissibility, then greedily add back."""
        nonlocal best_points, best_size
        nonlocal all_primes, prime_set, N, start_time, DEADLINE
        
        points = set(points_set)
        points = {x for x in points if 0 <= x <= N}
        
        if not points:
            return
        
        # Build residue counts
        residue_counts = {}
        missing_counts = {}
        for p in all_primes:
            counts = [0] * p
            residue_counts[p] = counts
            missing_counts[p] = p
        
        for x in points:
            for p in all_primes:
                r = x % p
                if residue_counts[p][r] == 0:
                    missing_counts[p] -= 1
                residue_counts[p][r] += 1
        
        # Repair phase: ensure admissibility
        changed = True
        while changed:
            changed = False
            if time.time() - start_time > DEADLINE:
                break
            n = len(points)
            for p in all_primes:
                if p > n:
                    break
                if missing_counts[p] == 0:
                    min_count = min(residue_counts[p])
                    target_residues = [r for r, c in enumerate(residue_counts[p]) if c == min_count]
                    target = target_residues[0]
                    to_remove = [x for x in points if x % p == target]
                    if to_remove:
                        for x in to_remove:
                            points.remove(x)
                            for pp in all_primes:
                                r = x % pp
                                residue_counts[pp][r] -= 1
                                if residue_counts[pp][r] == 0:
                                    missing_counts[pp] += 1
                        changed = True
                        break
        
        # Greedy add-back with batch processing
        added_any = True
        while added_any:
            added_any = False
            if time.time() - start_time > DEADLINE:
                break
            
            n = len(points)
            new_n = n + 1
            new_prime = new_n if new_n in prime_set else None
            
            # Collect all valid candidates in one pass
            valid_to_add = []
            for x in range(N + 1):
                if x in points:
                    continue
                
                valid = True
                # Check existing primes p <= n
                for p in all_primes:
                    if p > n:
                        break
                    r = x % p
                    if residue_counts[p][r] == 0 and missing_counts[p] == 1:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check new prime bound
                if new_prime is not None:
                    p = new_prime
                    r_x = x % p
                    if residue_counts[p][r_x] == 0:
                        final_missing = missing_counts[p] - 1
                    else:
                        final_missing = missing_counts[p]
                    if final_missing == 0:
                        valid = False
                
                if valid:
                    valid_to_add.append(x)
            
            # Add all valid candidates at once
            if valid_to_add:
                for x in valid_to_add:
                    points.add(x)
                    for pp in all_primes:
                        r = x % pp
                        if residue_counts[pp][r] == 0:
                            missing_counts[pp] -= 1
                        residue_counts[pp][r] += 1
                added_any = True
        
        # Targeted removal: for primes with exactly 1 missing, remove least-populated
        changed = True
        while changed:
            changed = False
            if time.time() - start_time > DEADLINE:
                break
            n = len(points)
            for p in all_primes:
                if p > n:
                    break
                if missing_counts[p] == 1:
                    min_count = min(residue_counts[p])
                    target_residues = [r for r, c in enumerate(residue_counts[p]) if c == min_count]
                    target = target_residues[0]
                    to_remove = [x for x in points if x % p == target]
                    if to_remove and len(to_remove) <= 5:
                        for x in to_remove:
                            points.remove(x)
                            for pp in all_primes:
                                r = x % pp
                                residue_counts[pp][r] -= 1
                                if residue_counts[pp][r] == 0:
                                    missing_counts[pp] += 1
                        changed = True
                        break
        
        # Second add-back pass
        added_any = True
        while added_any:
            added_any = False
            if time.time() - start_time > DEADLINE:
                break
            n = len(points)
            new_n = n + 1
            new_prime = new_n if new_n in prime_set else None
            
            valid_to_add = []
            for x in range(N + 1):
                if x in points:
                    continue
                valid = True
                for p in all_primes:
                    if p > n:
                        break
                    r = x % p
                    if residue_counts[p][r] == 0 and missing_counts[p] == 1:
                        valid = False
                        break
                if not valid:
                    continue
                if new_prime is not None:
                    p = new_prime
                    r_x = x % p
                    if residue_counts[p][r_x] == 0:
                        final_missing = missing_counts[p] - 1
                    else:
                        final_missing = missing_counts[p]
                    if final_missing == 0:
                        valid = False
                if valid:
                    valid_to_add.append(x)
            
            if valid_to_add:
                for x in valid_to_add:
                    points.add(x)
                    for pp in all_primes:
                        r = x % pp
                        if residue_counts[pp][r] == 0:
                            missing_counts[pp] -= 1
                        residue_counts[pp][r] += 1
                added_any = True
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = list(points)
    
    # Strategy 1: Start with all points, remove minority classes
    if time.time() - start_time < DEADLINE:
        _evaluate_and_refine(set(range(N + 1)), "full")
    
    # Strategy 2: Start with 1 and primes > sqrt(N)
    if time.time() - start_time < DEADLINE:
        sqrt_N = int(N ** 0.5) + 1
        points = {1}
        for p in all_primes:
            if p > sqrt_N:
                points.add(p)
        _evaluate_and_refine(points, "primes")
    
    # Strategy 3: Random subsets
    for seed in range(20):
        if time.time() - start_time > DEADLINE:
            break
        rng = random.Random(seed)
        subset_size = max(50, N // 8)
        points = set(rng.sample(range(N + 1), subset_size))
        _evaluate_and_refine(points, f"rand{seed}")
    
    # Strategy 4: Evens only
    if time.time() - start_time < DEADLINE:
        points = set(range(0, N + 1, 2))
        _evaluate_and_refine(points, "evens")
    
    # Strategy 5: Odds only
    if time.time() - start_time < DEADLINE:
        points = set(range(1, N + 1, 2))
        _evaluate_and_refine(points, "odds")
    
    # Strategy 6: Multiples of small primes removed
    if time.time() - start_time < DEADLINE:
        points = set(range(N + 1))
        for x in list(points):
            if x % 2 == 0 or x % 3 == 0:
                points.discard(x)
        _evaluate_and_refine(points, "no23")
    
    # Final verification
    if best_points is None:
        # Fallback
        points = set(range(N + 1))
        _evaluate_and_refine(points, "fallback")
        if best_points is None:
            best_points = list(points)
    
    # Verify admissibility of best_points
    n = len(best_points)
    points_set = set(best_points)
    for p in all_primes:
        if p > n:
            break
        residues = set(x % p for x in best_points)
        if len(residues) == p:
            # Not admissible, need to fix
            # Remove from the most populated residue
            counts = [0] * p
            for x in best_points:
                counts[x % p] += 1
            max_idx = max(range(p), key=lambda i: counts[i])
            to_remove = [x for x in best_points if x % p == max_idx]
            for x in to_remove:
                best_points.remove(x)
            # Re-verify
            n = len(best_points)
            for p2 in all_primes:
                if p2 > n:
                    break
                residues2 = set(x % p2 for x in best_points)
                if len(residues2) == p2:
                    counts2 = [0] * p2
                    for x in best_points:
                        counts2[x % p2] += 1
                    max_idx2 = max(range(p2), key=lambda i: counts2[i])
                    to_remove2 = [x for x in best_points if x % p2 == max_idx2]
                    for x in to_remove2:
                        best_points.remove(x)
    
    return best_points
# EVOLVE-BLOCK-END
