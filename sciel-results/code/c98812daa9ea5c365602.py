import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # Generate primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i in range(2, limit+1) if sieve[i]]

    primes = get_primes(1000)
    n = len(primes)  # 168

    # --- Fast numpy-based scoring ---
    def _build_count_array(residues, primes, max_L):
        """Build coverage count array for positions 1..max_L."""
        counts = np.zeros(max_L + 1, dtype=np.int32)
        for i in range(len(primes)):
            r = residues[i]
            p = primes[i]
            r_eff = p if r == 0 else r
            if r_eff <= max_L:
                positions = np.arange(r_eff, max_L + 1, p)
                counts[positions] += 1
        return counts

    def _find_L(counts, max_L):
        """Find largest L such that all positions 1..L have count >= 2."""
        if max_L <= 0:
            return 0
        deficient = np.where(counts[1:max_L+1] < 2)[0]
        if len(deficient) == 0:
            return max_L
        return int(deficient[0])

    def _score(residues, primes, max_L=20000):
        """Compute the largest L such that every integer 1..L is covered at least twice."""
        counts = _build_count_array(residues, primes, max_L)
        return _find_L(counts, max_L)

    def _score_full(residues, primes, max_L=100000):
        """Full range score for final verification."""
        counts = _build_count_array(residues, primes, max_L)
        return _find_L(counts, max_L)

    # --- Structural approach: use the fact that we need double coverage ---
    # Key insight: if we set r_p = 0 for all p, then every multiple of p is covered.
    # But we need each number covered at least twice.
    # 
    # Better structural insight: Think of this as a covering problem where we want
    # to maximize the initial segment [1, L] where each point is hit by at least 2 sets.
    #
    # New approach: Start with a clever initialization based on the structure of the problem.
    # For small primes, the residue class covers many early numbers. We want to ensure
    # that early numbers (1, 2, 3, ...) are covered by at least 2 residue classes.
    #
    # Strategy: 
    # 1. Initialize with r_p = 0 for all (covers multiples of p)
    # 2. For each position x from 1 upward, check if it's covered at least twice.
    #    If not, adjust residues of primes that could cover x.
    # 3. Use simulated annealing / local search to refine.

    # --- Phase 1: Smart initialization ---
    def _smart_init(primes):
        """
        Initialize residues with a strategy that aims for good early coverage.
        Use the idea: for each small position x, we want at least 2 primes p 
        where x % p == r_p.
        
        We'll use a greedy approach: process positions 1, 2, 3, ... and for each
        position that needs more coverage, pick a prime and set its residue.
        """
        n = len(primes)
        residues = [0] * n  # Start with all zeros
        max_L_check = 5000
        counts = np.zeros(max_L_check + 1, dtype=np.int32)
        
        # First pass: set all to 0 and compute initial counts
        for i in range(n):
            p = primes[i]
            positions = np.arange(p, max_L_check + 1, p)
            counts[positions] += 1
        
        # Now iterate through positions and fix deficiencies
        # For each position x with count < 2, find a prime p that can cover x
        # and set r_p = x % p (if not already set)
        
        # We'll do multiple passes
        for _pass in range(50):
            if time.time() > deadline:
                break
            
            improved = False
            for x in range(1, min(max_L_check, 2000) + 1):
                if counts[x] >= 2:
                    continue
                
                # This position needs more coverage
                # Find a prime p where setting r_p = x % p would help
                # and where r_p is currently not x % p
                best_i = -1
                best_gain = 0
                
                for i in range(n):
                    p = primes[i]
                    needed_r = x % p
                    current_r = residues[i]
                    
                    if current_r == needed_r:
                        continue  # Already covers x
                    
                    # If we set r_p = needed_r, how many additional positions does it cover?
                    # It covers positions: needed_r, needed_r+p, needed_r+2p, ...
                    # But it also UNCOVERS positions: current_r, current_r+p, ... (if current_r != 0)
                    
                    # Simple heuristic: just count how many deficient positions it would cover
                    r_eff_new = p if needed_r == 0 else needed_r
                    if r_eff_new > max_L_check:
                        continue
                    
                    # Count new coverage
                    new_positions = np.arange(r_eff_new, max_L_check + 1, p)
                    new_gain = 0
                    for pos in new_positions:
                        if counts[pos] < 2:
                            new_gain += 1
                    
                    # Count lost coverage (positions that were covered by old residue)
                    r_eff_old = p if current_r == 0 else current_r
                    if r_eff_old <= max_L_check:
                        old_positions = np.arange(r_eff_old, max_L_check + 1, p)
                        old_loss = 0
                        for pos in old_positions:
                            if counts[pos] == 1:  # Would become 0
                                old_loss += 1
                    
                    net_gain = new_gain - old_loss
                    if net_gain > best_gain:
                        best_gain = net_gain
                        best_i = i
                
                if best_i >= 0:
                    i = best_i
                    p = primes[i]
                    old_r = residues[i]
                    new_r = x % p
                    
                    # Update counts
                    r_eff_old = p if old_r == 0 else old_r
                    if r_eff_old <= max_L_check:
                        old_positions = np.arange(r_eff_old, max_L_check + 1, p)
                        counts[old_positions] -= 1
                    
                    r_eff_new = p if new_r == 0 else new_r
                    if r_eff_new <= max_L_check:
                        new_positions = np.arange(r_eff_new, max_L_check + 1, p)
                        counts[new_positions] += 1
                    
                    residues[i] = new_r
                    improved = True
            
            if not improved:
                break
        
        return residues

    # --- Phase 2: Pairwise swap optimization ---
    def _pairwise_optimize(residues, primes, max_L_check, deadline):
        """
        Try swapping residues between pairs of primes to improve the score.
        """
        n = len(primes)
        best_res = residues[:]
        best_L = _score(best_res, primes, max_L_check)
        
        no_improve = 0
        max_no_improve = 20
        
        random.seed(123)
        
        while time.time() < deadline and no_improve < max_no_improve:
            improved = False
            no_improve += 1
            
            # Try random pairs
            for _ in range(100):
                if time.time() > deadline:
                    break
                
                i, j = random.sample(range(n), 2)
                p_i = primes[i]
                p_j = primes[j]
                r_i = best_res[i]
                r_j = best_res[j]
                
                # Try swapping
                best_res[i] = r_j % p_i  # Note: r_j might be >= p_i
                best_res[j] = r_i % p_j  # Note: r_i might be >= p_j
                
                new_L = _score(best_res, primes, max_L_check)
                if new_L > best_L:
                    best_L = new_L
                    improved = True
                    no_improve = 0
                else:
                    # Revert
                    best_res[i] = r_i
                    best_res[j] = r_j
            
            if not improved:
                no_improve += 1
        
        return best_res, best_L

    # --- Phase 3: Local search with hill climbing ---
    def _local_search(residues, primes, max_L_check, deadline):
        """
        Local search: randomly perturb residues and accept improvements.
        Use simulated annealing style with temperature decay.
        """
        n = len(primes)
        best_res = residues[:]
        best_L = _score(best_res, primes, max_L_check)
        current_res = residues[:]
        current_L = best_L
        
        no_improve = 0
        max_no_improve = 50
        temp = 100.0
        cooling = 0.95
        
        random.seed(42)
        
        while time.time() < deadline and no_improve < max_no_improve:
            improved = False
            
            # Pick a random prime to perturb
            i = random.randint(0, n - 1)
            p = primes[i]
            old_r = current_res[i]
            
            # Try a random new residue
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            current_res[i] = new_r
            new_L = _score(current_res, primes, max_L_check)
            
            # Accept if better, or with probability based on temperature
            if new_L >= current_L:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_res = current_res[:]
                    improved = True
                    no_improve = 0
            else:
                # Metropolis criterion
                delta = new_L - current_L
                if random.random() < np.exp(delta / temp):
                    current_L = new_L
                else:
                    current_res[i] = old_r
            
            temp *= cooling
            
            if not improved:
                no_improve += 1
        
        return best_res, best_L

    # --- Phase 4: Multi-start with diverse initializations ---
    def _multi_start(primes, deadline):
        """
        Try multiple different initializations and keep the best result.
        """
        best_overall = None
        best_overall_L = 0
        
        # Different initialization strategies
        strategies = []
        
        # Strategy 1: All zeros
        strategies.append([0] * n)
        
        # Strategy 2: Random
        random.seed(42)
        strategies.append([random.randint(0, p - 1) for p in primes])
        
        # Strategy 3: Smart init
        strategies.append(_smart_init(primes))
        
        # Strategy 4: Based on small numbers coverage
        # For each position x from 1 to some limit, ensure it's covered
        # by setting residues of small primes
        res4 = [0] * n
        counts4 = np.zeros(5001, dtype=np.int32)
        for i in range(n):
            p = primes[i]
            positions = np.arange(p, 5001, p)
            counts4[positions] += 1
        
        # Fix early deficiencies
        for x in range(1, 500):
            if counts4[x] < 2:
                # Find a small prime that can cover x
                for i in range(min(n, 20)):  # Only small primes
                    p = primes[i]
                    needed_r = x % p
                    if res4[i] != needed_r:
                        old_r = res4[i]
                        # Update counts
                        r_eff_old = p if old_r == 0 else old_r
                        if r_eff_old <= 5000:
                            positions = np.arange(r_eff_old, 5001, p)
                            counts4[positions] -= 1
                        
                        res4[i] = needed_r
                        r_eff_new = p if needed_r == 0 else needed_r
                        if r_eff_new <= 5000:
                            positions = np.arange(r_eff_new, 5001, p)
                            counts4[positions] += 1
                        break
        strategies.append(res4)
        
        # For each strategy, run optimization
        for idx, init_res in enumerate(strategies):
            if time.time() > deadline:
                break
            
            # Quick score check
            quick_L = _score(init_res, primes, 5000)
            
            # Run local search with time budget
            time_budget = min(30, deadline - time.time())
            if time_budget < 5:
                break
            
            res, L = _local_search(init_res, primes, 20000, start_time + time_budget)
            
            if L > best_overall_L:
                best_overall_L = L
                best_overall = res[:]
        
        return best_overall, best_overall_L

    # --- Main algorithm ---
    # Phase 1: Get a good initial solution
    residues = _smart_init(primes)
    current_L = _score(residues, primes, 20000)
    
    # Phase 2: Local search refinement
    time_left = deadline - time.time()
    if time_left > 10:
        residues, current_L = _local_search(residues, primes, 20000, start_time + min(time_left - 5, 60))
    
    # Phase 3: Pairwise optimization
    time_left = deadline - time.time()
    if time_left > 10:
        residues, current_L = _pairwise_optimize(residues, primes, 20000, start_time + min(time_left - 5, 30))
    
    # Phase 4: Final local search polish
    time_left = deadline - time.time()
    if time_left > 10:
        residues, current_L = _local_search(residues, primes, 20000, start_time + min(time_left - 3, 40))
    
    # Final verification with full range
    final_L = _score_full(residues, primes, 100000)
    
    # Ensure pure Python ints
    return [int(r) for r in residues]
