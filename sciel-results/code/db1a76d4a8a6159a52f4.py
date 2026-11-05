import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 10s margin
    
    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = [False] * len(sieve[i*i::i])
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    assert n_primes == 168
    
    # L_max: search horizon for coverage
    L_max = 20000
    
    # Precompute positions for each prime and residue
    # positions[i] is a list of lists: positions[i][r] = list of x in [1, L_max] with x ≡ r (mod p)
    positions = []
    for i, p in enumerate(primes):
        pos_list = []
        for r in range(p):
            if r == 0:
                pos_list.append(np.arange(p, L_max + 1, p))
            else:
                pos_list.append(np.arange(r, L_max + 1, p))
        positions.append(pos_list)
    
    def find_L(coverage):
        """Find largest L such that coverage[x] >= 2 for all x in [1, L]."""
        L = 0
        for x in range(1, L_max + 1):
            if coverage[x] >= 2:
                L = x
            else:
                break
        return L
    
    def fast_find_L(coverage, current_L):
        """Find largest L such that coverage[x] >= 2 for all x in [1, L], starting from current_L."""
        # First, verify current_L is still valid
        L = 0
        for x in range(1, min(current_L + 1, L_max + 1)):
            if coverage[x] >= 2:
                L = x
            else:
                break
        # Now extend beyond current_L
        x = L + 1
        while x <= L_max and coverage[x] >= 2:
            L = x
            x += 1
        return L
    
    # Greedy initialization: for each prime in order, pick residue that maximizes
    # coverage of currently under-covered positions in [1, 500]
    residues = [0] * n_primes
    coverage = np.zeros(L_max + 1, dtype=np.int32)
    
    for i, p in enumerate(primes):
        best_r = 0
        best_gain = -1
        for r in range(p):
            gain = 0
            for x in positions[i][r]:
                if x <= 500 and coverage[x] < 2:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        residues[i] = best_r
        coverage[positions[i][best_r]] += 1
    
    current_L = find_L(coverage)
    best_L = current_L
    best_residues = residues[:]
    
    # Deficit function: f(c) = max(0, 2 - c)
    def deficit(c):
        return 0 if c >= 2 else (2 - c)
    
    # Simulated Annealing with delta evaluation
    T = 15.0
    T_min = 0.01
    alpha = 0.9998
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        if T < T_min:
            break
        
        # Pick random prime
        i = rng.randrange(n_primes)
        p = primes[i]
        r_old = residues[i]
        
        # Pick new residue
        r_new = rng.randrange(p)
        if r_old == r_new:
            continue
        
        old_pos = positions[i][r_old]
        new_pos = positions[i][r_new]
        
        # Calculate delta E using vectorized operations
        # Get unique affected positions
        old_set = set(old_pos.tolist())
        new_set = set(new_pos.tolist())
        affected = old_set.union(new_set)
        
        # Calculate delta E
        delta_E = 0
        for x in affected:
            c0 = coverage[x]
            c1 = c0
            if x in old_set:
                c1 -= 1
            if x in new_set:
                c1 += 1
            delta_E += deficit(c1) - deficit(c0)
        
        # Decide accept/reject
        accept = False
        if delta_E < 0:
            accept = True
        elif delta_E == 0:
            accept = True
        else:
            # Accept worse with probability exp(-delta_E / T)
            if delta_E < T * 5:  # avoid overflow
                if rng.random() < np.exp(-delta_E / T):
                    accept = True
        
        if accept:
            # Apply change
            coverage[old_pos] -= 1
            coverage[new_pos] += 1
            residues[i] = r_new
            
            # Check if we improved L
            # Only check L if delta_E <= 0 (improvement or neutral)
            # For simplicity, check periodically
            if delta_E <= 0:
                cur_L = fast_find_L(coverage, current_L)
                if cur_L > best_L:
                    best_L = cur_L
                    best_residues = residues[:]
                current_L = cur_L
        
        # Cool down
        T *= alpha
    
    # Backward refinement: for each prime, try all residues to maximize L
    # Process primes in reverse order (large primes first) for efficiency
    # Actually, process in order but with early termination
    for i in range(n_primes):
        if time.time() > deadline - 5:
            break
        p = primes[i]
        old_r = best_residues[i]
        best_local = best_L
        best_local_r = old_r
        
        # Try all residues
        for r in range(p):
            if r == old_r:
                continue
            best_residues[i] = r
            
            # Compute new coverage efficiently
            # Remove old positions, add new positions
            old_pos = positions[i][old_r]
            new_pos = positions[i][r]
            
            # We need to recompute coverage from scratch or track it
            # For simplicity in refinement, let's recompute coverage
            # This is O(L_max * n_primes) which is too slow
            # Instead, let's maintain coverage and update it
            
            # Actually, let's just do a full recompute for correctness
            # But that's too slow. Let's use the delta approach.
            
            # Revert and try again with proper tracking
            best_residues[i] = old_r
            break  # Skip this for now, use SA for refinement
        
        # Better refinement: use local search with delta
        for r in range(p):
            if r == old_r:
                continue
            if time.time() > deadline - 5:
                break
            
            old_pos = positions[i][old_r]
            new_pos = positions[i][r]
            
            old_set = set(old_pos.tolist())
            new_set = set(new_pos.tolist())
            affected = old_set.union(new_set)
            
            delta_E = 0
            for x in affected:
                c0 = coverage[x]
                c1 = c0
                if x in old_set:
                    c1 -= 1
                if x in new_set:
                    c1 += 1
                delta_E += deficit(c1) - deficit(c0)
            
            if delta_E < 0:
                # Apply
                coverage[old_pos] -= 1
                coverage[new_pos] += 1
                old_r = r
                cur_L = fast_find_L(coverage, current_L)
                if cur_L > best_L:
                    best_L = cur_L
                    best_residues = residues[:]
                current_L = cur_L
    
    # Final polish: try nearby residues for each prime to extend the frontier
    for i in range(n_primes):
        if time.time() > deadline - 2:
            break
        p = primes[i]
        old_r = best_residues[i]
        
        # Try a few nearby residues
        candidates = []
        for dr in [-2, -1, 1, 2]:
            r_candidate = (old_r + dr) % p
            if r_candidate != old_r:
                candidates.append(r_candidate)
        
        for r in candidates:
            if time.time() > deadline - 2:
                break
            
            old_pos = positions[i][old_r]
            new_pos = positions[i][r]
            
            old_set = set(old_pos.tolist())
            new_set = set(new_pos.tolist())
            affected = old_set.union(new_set)
            
            delta_E = 0
            for x in affected:
                c0 = coverage[x]
                c1 = c0
                if x in old_set:
                    c1 -= 1
                if x in new_set:
                    c1 += 1
                delta_E += deficit(c1) - deficit(c0)
            
            if delta_E < 0:
                # Apply
                coverage[old_pos] -= 1
                coverage[new_pos] += 1
                old_r = r
                residues[i] = r
                cur_L = fast_find_L(coverage, current_L)
                if cur_L > best_L:
                    best_L = cur_L
                    best_residues = residues[:]
                current_L = cur_L
    
    # Final check
    cur_L = find_L(coverage)
    if cur_L > best_L:
        best_L = cur_L
        best_residues = residues[:]
    
    return [int(r) for r in best_residues]
