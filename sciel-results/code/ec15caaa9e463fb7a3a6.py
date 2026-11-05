import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # leave 10s margin
    
    # Generate primes up to 1000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    nprimes = len(primes)
    
    # For evaluation, we need to find the largest L such that every integer 1..L
    # satisfies at least 2 congruences.
    # We'll use a vectorized approach with a running window.
    
    def compute_L(residues, primes, max_check=10000):
        """
        Compute the largest L such that every integer 1..L has count >= 2.
        residues[i] is the residue for primes[i].
        """
        # For each integer x in 1..max_check, count how many primes p_i 
        # satisfy x ≡ residues[i] (mod p_i)
        
        # We'll build the coverage array efficiently
        # coverage[x] = number of primes p_i where x % p_i == residues[i]
        
        coverage = np.zeros(max_check + 1, dtype=np.int32)
        
        for i, p in enumerate(primes):
            r = residues[i]
            # x ≡ r (mod p) means x = r, r+p, r+2p, ...
            # But x starts from 1, and r could be 0
            # If r == 0, then x = p, 2p, 3p, ... (since x >= 1, and 0 < 1)
            # Actually x = r + k*p for k >= 0, but x >= 1
            # If r == 0: x = p, 2p, 3p, ...
            # If r > 0: x = r, r+p, r+2p, ...
            
            if r == 0:
                xs = np.arange(p, max_check + 1, p)
            else:
                xs = np.arange(r, max_check + 1, p)
                xs = xs[xs >= 1]
            
            if len(xs) > 0:
                coverage[xs] += 1
        
        # Find the largest L such that coverage[1..L] >= 2
        # i.e., no x in 1..L has coverage[x] < 2
        bad = np.where(coverage[1:max_check + 1] < 2)[0]
        if len(bad) == 0:
            return max_check
        else:
            return int(bad[0])
    
    def compute_L_fast(residues, primes, current_L, max_check=10000):
        """
        Optimized: check from current_L+1 upward to find where coverage < 2.
        """
        # Just use the full computation for simplicity, it's fast enough
        return compute_L(residues, primes, max_check)
    
    # Initial greedy: align residues to cover small integers densely
    # Strategy: for each prime, pick residue to cover the "frontier"
    # Start with all zeros
    residues = np.zeros(nprimes, dtype=np.int32)
    
    # Better initialization: for each prime p_i, pick r_i to maximize coverage of early integers
    # Greedy approach: process primes in order, for each pick residue that maximizes 
    # the minimum coverage in 1..current_frontier
    
    # Even simpler: use a heuristic where we try to align all residues to cover 
    # a specific small number, then shift.
    
    # Let's try: for each prime, pick residue to cover ceil(some_target)
    # The seed uses ((i+2)//2) % p which covers ceil((i+1)/2)
    
    # Let me try a better greedy: 
    # For each prime in order, try all residues and pick the one that 
    # maximizes the minimum count in 1..min(200, current_best+100)
    
    residues = np.zeros(nprimes, dtype=np.int32)
    
    # Greedy initialization
    for i in range(nprimes):
        best_r = 0
        best_score = -1
        p = primes[i]
        
        # Try a subset of residues for speed (all if p is small)
        if p <= 50:
            candidates = range(p)
        else:
            # Sample residues
            candidates = np.random.choice(p, size=min(p, 50), replace=False)
        
        for r in candidates:
            residues[i] = r
            # Quick evaluation on a small window
            window = 200
            coverage = np.zeros(window + 1, dtype=np.int32)
            for j in range(i + 1):
                pj = primes[j]
                rj = residues[j]
                if rj == 0:
                    xs = np.arange(pj, window + 1, pj)
                else:
                    xs = np.arange(rj, window + 1, pj)
                    xs = xs[xs >= 1]
                if len(xs) > 0:
                    coverage[xs] += 1
            # Score: minimize the first position where count < 2, or maximize min count
            # We want to extend the doubly-covered prefix
            bad = np.where(coverage[1:window + 1] < 2)[0]
            if len(bad) == 0:
                score = window
            else:
                score = int(bad[0])
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[i] = best_r
    
    # Compute initial L
    current_L = compute_L(residues, primes)
    best_L = current_L
    best_residues = residues.copy()
    
    # Simulated Annealing
    np.random.seed(42)
    
    T = 50.0  # Initial temperature
    T_min = 0.01
    cooling = 0.999
    
    # Precompute index structures for faster evaluation
    # For each prime i and residue r, precompute the positions in 1..max_check
    max_check = 5000  # We'll check up to 5000 during SA
    idx_cache = []
    for p in primes:
        idx_list = []
        for r in range(p):
            if r == 0:
                xs = np.arange(p, max_check + 1, p)
            else:
                xs = np.arange(r, max_check + 1, p)
                xs = xs[xs >= 1]
            idx_list.append(xs)
        idx_cache.append(idx_list)
    
    # For SA, we need fast incremental updates
    # Maintain a coverage array for 1..max_check
    def full_coverage(residues):
        coverage = np.zeros(max_check + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            xs = idx_cache[i][r]
            if len(xs) > 0:
                coverage[xs] += 1
        return coverage
    
    coverage = full_coverage(residues)
    
    # Helper to get L from coverage
    def get_L_from_coverage(coverage):
        # Find first x in 1..max_check where coverage[x] < 2
        # Use searchsorted on where coverage < 2
        bad_mask = coverage[1:max_check + 1] < 2
        if not np.any(bad_mask):
            return max_check
        first_bad = np.argmax(bad_mask)
        return int(first_bad)
    
    # Track current L
    current_L = get_L_from_coverage(coverage)
    best_L = current_L
    best_residues = residues.copy()
    
    # SA loop
    steps = 0
    while time.time() < deadline:
        # Pick a random prime index
        i = np.random.randint(nprimes)
        p = primes[i]
        
        # Pick a new residue (different from current)
        old_r = residues[i]
        if p == 2:
            new_r = 1 - old_r
        else:
            new_r = np.random.randint(0, p)
            while new_r == old_r:
                new_r = np.random.randint(0, p)
        
        # Compute delta in coverage
        old_xs = idx_cache[i][old_r]
        new_xs = idx_cache[i][new_r]
        
        # Apply change
        if len(old_xs) > 0:
            coverage[old_xs] -= 1
        if len(new_xs) > 0:
            coverage[new_xs] += 1
        
        new_L = get_L_from_coverage(coverage)
        
        # Acceptance criterion
        if new_L >= best_L:
            # Always accept improvements or equals
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
                best_residues[i] = new_r
        else:
            # Accept with probability exp((new_L - current_L) / T)
            delta = new_L - current_L  # negative
            if np.random.random() < np.exp(delta / T):
                current_L = new_L
            else:
                # Reject: revert
                if len(new_xs) > 0:
                    coverage[new_xs] -= 1
                if len(old_xs) > 0:
                    coverage[old_xs] += 1
                new_r = old_r
        
        residues[i] = new_r
        steps += 1
        
        # Cool down
        T = max(T_min, T * cooling)
        
        # Periodic restart if stuck
        if steps % 10000 == 0 and T <= T_min * 1.5:
            # Reset temperature
            T = 50.0
            # Maybe try a few random moves
            for _ in range(10):
                i2 = np.random.randint(nprimes)
                p2 = primes[i2]
                old_r2 = residues[i2]
                new_r2 = np.random.randint(0, p2)
                old_xs2 = idx_cache[i2][old_r2]
                new_xs2 = idx_cache[i2][new_r2]
                if len(old_xs2) > 0:
                    coverage[old_xs2] -= 1
                if len(new_xs2) > 0:
                    coverage[new_xs2] += 1
                residues[i2] = new_r2
            current_L = get_L_from_coverage(coverage)
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
    
    # Final evaluation with full precision
    final_L = compute_L(best_residues, primes, max_check=10000)
    if final_L > best_L:
        best_L = final_L
    
    return best_residues.tolist()
