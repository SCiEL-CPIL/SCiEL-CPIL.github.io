"""Rewritten program for covering interval with at least 2 congruences."""
import time
import random
import numpy as np

_DEADLINE = 145.0  # safety margin before 150s wall-clock limit


def _primes_up_to(n):
    """Sieve of Eratosthenes for primes up to n."""
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(n + 1) if sieve[i]]


def _compute_score(residues, primes, MAX_L=100000):
    """Compute the coverage score for given residues."""
    cov = np.zeros(MAX_L + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        start = r if r != 0 else p
        if start <= MAX_L:
            cov[start::p] += 1
    bad = np.where(cov[1:] < 2)[0]
    return int(bad[0]) if len(bad) > 0 else MAX_L


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    primes = _primes_up_to(1000)
    n = len(primes)
    MAX_L = 100000
    
    # Precompute useful structures
    prime_indices = list(range(n))
    
    def build_coverage(residues):
        """Build the coverage array from scratch."""
        cov = np.zeros(MAX_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            start = r if r != 0 else p
            if start <= MAX_L:
                cov[start::p] += 1
        return cov
    
    def find_first_gap(cov):
        """Find the first x >= 1 where cov[x] < 2."""
        bad = np.where(cov[1:] < 2)[0]
        if len(bad) == 0:
            return MAX_L
        return int(bad[0])
    
    def fast_evaluate_L(cov, old_L, idx, old_r, new_r):
        """
        Quickly evaluate new L after changing residue at idx.
        Only checks positions that could be affected: those where coverage was 0 or 1
        and are in the vicinity of old_L, plus the frontier.
        """
        p = primes[idx]
        
        # Positions affected by this change: start_old::p and start_new::p
        old_start = old_r if old_r != 0 else p
        new_start = new_r if new_r != 0 else p
        
        # Collect candidate positions to check
        check_positions = set()
        
        # The frontier
        check_positions.add(old_L + 1)
        check_positions.add(old_L + 2)
        check_positions.add(old_L + 3)
        
        # Positions in [1, old_L+50] where coverage is 0 or 1
        check_end = min(old_L + 50, MAX_L)
        if check_end >= 1:
            # Check where coverage == 0
            zero_pos = np.where(cov[1:check_end+1] == 0)[0] + 1
            check_positions.update(zero_pos.tolist())
            # Check where coverage == 1
            one_pos = np.where(cov[1:check_end+1] == 1)[0] + 1
            check_positions.update(one_pos.tolist())
        
        # Also check positions that the new residue covers in the frontier region
        if new_start <= MAX_L:
            for pos in range(new_start, min(new_start + 200, MAX_L + 1), p):
                check_positions.add(pos)
        
        # Also check positions the old residue covered
        if old_start <= MAX_L:
            for pos in range(old_start, min(old_start + 200, MAX_L + 1), p):
                check_positions.add(pos)
        
        # Find the minimum position in check_positions where cov[pos] < 2
        min_bad = MAX_L
        for pos in check_positions:
            if 1 <= pos <= MAX_L and cov[pos] < 2:
                if pos < min_bad:
                    min_bad = pos
        
        # Also verify by checking from 1 up to min_bad if needed
        # But for efficiency, trust the candidate set for now
        # Do a final verification scan up to min_bad
        if min_bad < MAX_L:
            # Verify: scan from 1 to min_bad
            for x in range(1, min_bad + 1):
                if cov[x] < 2:
                    return x
            # If no gap found in [1, min_bad], the gap is at min_bad or beyond
            # Check a bit further
            for x in range(min_bad, min(min_bad + 10, MAX_L + 1)):
                if cov[x] < 2:
                    return x
        
        # If we haven't found a gap, do a full scan as fallback
        bad = np.where(cov[1:] < 2)[0]
        if len(bad) == 0:
            return MAX_L
        return int(bad[0])
    
    def update_coverage(cov, idx, old_r, new_r):
        """
        Update coverage array in-place when changing residue at idx from old_r to new_r.
        """
        p = primes[idx]
        
        # Remove old
        old_start = old_r if old_r != 0 else p
        if old_start <= MAX_L:
            cov[old_start::p] -= 1
            
        # Add new
        new_start = new_r if new_r != 0 else p
        if new_start <= MAX_L:
            cov[new_start::p] += 1

    best_L_global = 0
    best_residues_global = [0] * n
    
    # Define initializations
    def init_seed():
        """Seeded initialization with pattern."""
        return [((i + 2) // 2) % p for i, p in enumerate(primes)]
        
    def init_random():
        """Random initialization."""
        return [random.randint(0, p-1) for p in primes]

    def init_perturbed():
        """Perturbed seed initialization."""
        r = init_seed()
        for _ in range(15):
            idx = random.randint(0, n-1)
            r[idx] = random.randint(0, primes[idx]-1)
        return r

    def init_targeted():
        """Initialize by targeting coverage of early integers."""
        r = [0] * n
        assigned = [False] * n
        for x in range(1, 801):
            # Count how many primes already cover x
            count = 0
            for i, p in enumerate(primes):
                if assigned[i]:
                    if (x - r[i]) % p == 0:
                        count += 1
            if count < 2:
                # Find primes that don't yet have a residue assigned
                for i in random.sample(prime_indices, n):
                    p = primes[i]
                    if not assigned[i]:
                        r[i] = x % p
                        assigned[i] = True
                        count += 1
                        if count >= 2:
                            break
        # Fill any remaining
        for i in range(n):
            if not assigned[i]:
                r[i] = random.randint(0, primes[i]-1)
                assigned[i] = True
        return r

    def init_double_targeted():
        """More aggressive targeted initialization."""
        r = [0] * n
        assigned = [False] * n
        for x in range(1, 1501):
            count = 0
            for i, p in enumerate(primes):
                if assigned[i]:
                    if (x - r[i]) % p == 0:
                        count += 1
            while count < 2:
                # Find a prime to assign
                assigned_flag = False
                for i in random.sample(prime_indices, n):
                    p = primes[i]
                    if not assigned[i]:
                        r[i] = x % p
                        assigned[i] = True
                        count += 1
                        assigned_flag = True
                        break
                if not assigned_flag:
                    break
        # Fill remaining
        for i in range(n):
            if not assigned[i]:
                r[i] = random.randint(0, primes[i]-1)
                assigned[i] = True
        return r

    def init_greedy_frontier():
        """Greedy initialization focusing on extending the frontier."""
        r = [0] * n
        assigned = [False] * n
        current_L = 0
        while current_L < 5000:
            target = current_L + 1
            # Count coverage of target
            count = 0
            for i, p in enumerate(primes):
                if assigned[i]:
                    if (target - r[i]) % p == 0:
                        count += 1
            if count >= 2:
                current_L = target
                continue
            # Need to assign more primes to cover target
            while count < 2:
                assigned_flag = False
                for i in random.sample(prime_indices, n):
                    p = primes[i]
                    if not assigned[i]:
                        r[i] = target % p
                        assigned[i] = True
                        count += 1
                        assigned_flag = True
                        break
                if not assigned_flag:
                    break
            if not assigned_flag:
                break
        # Fill remaining
        for i in range(n):
            if not assigned[i]:
                r[i] = random.randint(0, primes[i]-1)
                assigned[i] = True
        return r

    initializers = [init_seed, init_perturbed, init_random, init_targeted, init_double_targeted, 
                    init_greedy_frontier, init_perturbed, init_random, init_targeted, init_double_targeted,
                    init_greedy_frontier, init_perturbed]
    
    for init_func in initializers:
        if time.time() > deadline - 10:
            break
            
        residues = init_func()
        coverage = build_coverage(residues)
        current_L = find_first_gap(coverage)
        
        if current_L > best_L_global:
            best_L_global = current_L
            best_residues_global = residues[:]
            
        # Local Search with targeted refinement
        improved = True
        pass_count = 0
        max_passes = 100
        while improved and pass_count < max_passes and time.time() < deadline - 10:
            improved = False
            pass_count += 1
            # Shuffle prime indices to avoid bias
            shuffled_indices = list(range(n))
            random.shuffle(shuffled_indices)
            
            for idx in shuffled_indices:
                if time.time() > deadline - 10:
                    break
                
                p = primes[idx]
                old_r = residues[idx]
                best_r_for_idx = old_r
                best_L_for_idx = current_L
                
                # Generate candidates
                candidates = set()
                candidates.add(old_r)
                
                # Target the next gap
                target = current_L + 1
                if target <= MAX_L:
                    candidates.add(target % p)
                    candidates.add((target + 1) % p)
                    candidates.add((target + 2) % p)
                    
                # Target positions where coverage is exactly 1 (fixable by this prime)
                check_range = min(current_L + 100, MAX_L)
                if check_range >= 1:
                    one_cov_positions = np.where(coverage[1:check_range+1] == 1)[0] + 1
                    if len(one_cov_positions) > 0:
                        sample_size = min(30, len(one_cov_positions))
                        sampled = random.sample(list(one_cov_positions), sample_size)
                        for pos in sampled:
                            candidates.add(pos % p)
                
                # Add random candidates
                num_random = min(30, p)
                for _ in range(num_random):
                    candidates.add(random.randint(0, p-1))
                    
                # Add neighbors
                candidates.add((old_r + 1) % p)
                candidates.add((old_r - 1) % p)
                
                # For small primes, try all
                if p <= 40:
                    candidates = set(range(p))
                
                for r in candidates:
                    if r == old_r:
                        continue
                    if time.time() > deadline - 10:
                        break
                        
                    # Apply change
                    update_coverage(coverage, idx, old_r, r)
                    
                    # Evaluate using fast method
                    new_L = fast_evaluate_L(coverage, current_L, idx, old_r, r)
                    
                    # Always revert to keep coverage clean for next candidate
                    update_coverage(coverage, idx, r, old_r)
                    
                    if new_L > best_L_for_idx:
                        best_L_for_idx = new_L
                        best_r_for_idx = r
                    
                # Commit best change for this index if found
                if best_r_for_idx != old_r:
                    # Apply the best change permanently
                    update_coverage(coverage, idx, old_r, best_r_for_idx)
                    residues[idx] = best_r_for_idx
                    current_L = best_L_for_idx
                    improved = True
                    if current_L > best_L_global:
                        best_L_global = current_L
                        best_residues_global = residues[:]
                        
        # End of local search for this initialization
        
    # Final polish: try to extend the current best solution further
    if best_L_global > 0 and time.time() < deadline - 5:
        residues = best_residues_global[:]
        coverage = build_coverage(residues)
        current_L = find_first_gap(coverage)
        
        # Additional passes focused on the frontier
        for _ in range(80):
            if time.time() > deadline - 5:
                break
            target = current_L + 1
            if target > MAX_L:
                break
            # Find primes that can cover target
            for idx in prime_indices:
                p = primes[idx]
                target_r = target % p
                if residues[idx] == target_r:
                    continue
                old_r = residues[idx]
                update_coverage(coverage, idx, old_r, target_r)
                new_L = fast_evaluate_L(coverage, current_L, idx, old_r, target_r)
                update_coverage(coverage, idx, target_r, old_r)
                if new_L > current_L:
                    update_coverage(coverage, idx, old_r, target_r)
                    residues[idx] = target_r
                    current_L = new_L
                    if current_L > best_L_global:
                        best_L_global = current_L
                        best_residues_global = residues[:]
                    break
    
    return [int(r) for r in best_residues_global]
