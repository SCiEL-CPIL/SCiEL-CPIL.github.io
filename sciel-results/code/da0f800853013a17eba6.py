"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _primes_upto(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n+1) if sieve[i]]

def _compute_score(residues, primes, max_L=100000):
    """Compute the largest L such that every integer 1..L satisfies at least 2 congruences."""
    n = len(primes)
    # For each integer x in 1..max_L, count how many primes p_i have x ≡ r_i (mod p_i)
    # Use a difference-array / sieve approach
    # For prime p with residue r, the covered integers in [1, max_L] are r, r+p, r+2p, ...
    # We need count >= 2 for all x in 1..L
    
    # Build count array
    counts = [0] * (max_L + 1)
    for i, (p, r) in enumerate(zip(primes, residues)):
        # Mark all x in [1, max_L] where x ≡ r (mod p)
        start = r if r >= 1 else r + p
        if start > max_L:
            continue
        for x in range(start, max_L + 1, p):
            counts[x] += 1
    
    # Find largest L where all 1..L have count >= 2
    L = 0
    for x in range(1, max_L + 1):
        if counts[x] >= 2:
            L = x
        else:
            break
    return L

def _compute_score_fast(residues, primes, max_L=2000):
    """Faster score computation for smaller L range."""
    n = len(primes)
    counts = [0] * (max_L + 1)
    for i, (p, r) in enumerate(zip(primes, residues)):
        start = r if r >= 1 else r + p
        if start > max_L:
            continue
        for x in range(start, max_L + 1, p):
            counts[x] += 1
    
    L = 0
    for x in range(1, max_L + 1):
        if counts[x] >= 2:
            L = x
        else:
            break
    return L

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # Strategy: Greedy coverage deficit
    # 1. Initialize residues to 0
    # 2. Iteratively, for each prime (in some order), choose the residue that maximizes
    #    coverage of currently-undercovered integers in [1, 2000]
    # 3. Then do local search on bottleneck primes
    
    # Phase 1: Greedy assignment
    # Process primes in order of increasing p (small primes have more frequent hits)
    # For each prime, try all residues and pick the one that gives best coverage
    
    # Use a working range for greedy: [1, 2000]
    WORK_L = 2000
    residues = [0] * n_primes
    
    # Initialize counts
    counts = [0] * (WORK_L + 1)
    
    # Process primes in order: small primes first (they have more hits per period)
    # Actually, process in order of prime value
    # For each prime, evaluate all possible residues
    
    # To speed up, we'll use a scoring function:
    # For residue r of prime p, the gain is sum over x in [1, WORK_L] where x ≡ r (mod p)
    # of (1 if counts[x] == 1 else 0)  -- i.e., how many currently-undercovered integers we fix
    
    # Process primes in increasing order
    order = sorted(range(n_primes), key=lambda i: primes[i])
    
    for idx in order:
        p = primes[idx]
        # Try all residues 0..p-1
        # For each residue r, compute: number of x in [1, WORK_L] with x ≡ r (mod p) and counts[x] < 2
        # This is the "deficit coverage"
        
        # To speed up: for prime p, the residues that matter are those where r (or r+p, etc.) falls in [1, WORK_L]
        # For small p, all residues 0..p-1 are relevant
        # For large p > WORK_L, only residue r (if 1 <= r <= WORK_L) matters, and it covers just that one integer
        
        best_r = 0
        best_score = -1
        
        if p <= WORK_L:
            # Try all residues
            for r in range(p):
                # Count how many x in [1, WORK_L] with x ≡ r (mod p) have counts[x] < 2
                # x values: r (if r>=1), r+p, r+2p, ...
                gain = 0
                start_x = r if r >= 1 else r + p
                for x in range(start_x, WORK_L + 1, p):
                    if counts[x] < 2:
                        gain += 1
                if gain > best_score:
                    best_score = gain
                    best_r = r
        else:
            # p > WORK_L: each residue r in [1, WORK_L] covers exactly one integer x=r
            # residue 0 covers no integer in [1, WORK_L]
            # Best residue is the one with minimum count (preferably 0 or 1)
            best_r = 0
            best_score = -1
            for r in range(1, min(p, WORK_L + 1)):
                gain = 1 if counts[r] < 2 else 0
                if gain > best_score:
                    best_score = gain
                    best_r = r
            # If no r in [1, WORK_L] has counts[r] < 2, keep 0
            if best_score <= 0:
                best_r = 0
        
        residues[idx] = best_r
        
        # Update counts
        for x in range(best_r if best_r >= 1 else best_r + p, WORK_L + 1, p):
            counts[x] += 1
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    # Phase 2: Local search / hill climbing
    # Try to improve by changing residues of individual primes
    current_score = _compute_score_fast(residues, primes, WORK_L)
    
    # Check time
    if time.time() > deadline - 10.0:
        # Compute full score if time allows
        if time.time() < deadline - 2.0:
            final_score = _compute_score(residues, primes)
        else:
            final_score = current_score
        return residues
    
    # Hill climbing: try changing one residue at a time
    improved = True
    while improved and time.time() < deadline - 15.0:
        improved = False
        # Try primes in random order for diversity
        trial_order = list(range(n_primes))
        random.shuffle(trial_order)
        
        for idx in trial_order:
            if time.time() > deadline - 15.0:
                break
            p = primes[idx]
            old_r = residues[idx]
            
            # Try a few candidate residues
            # Focus on residues that might help: try all if p is small, sample if p is large
            if p <= 50:
                candidates = list(range(p))
            else:
                # Sample some candidates
                candidates = [old_r]
                # Try residues near the current bottleneck
                # Find first x where count < 2
                # For efficiency, just try a few random ones
                for _ in range(min(20, p)):
                    candidates.append(random.randint(0, p - 1))
                candidates = list(set(candidates))
            
            best_r = old_r
            best_score = current_score
            
            for r in candidates:
                if r == old_r:
                    continue
                # Compute new score quickly
                # Update counts: remove old, add new
                # For efficiency, just compute the delta
                
                # Delta: for x where x ≡ old_r (mod p) and x ≡ r (mod p) in [1, WORK_L]
                # These are the only x whose count changes
                # x ≡ old_r (mod p) and x ≡ r (mod p) => x ≡ r (mod p) and old_r ≡ r (mod p)
                # Since 0 <= old_r, r < p, old_r ≡ r (mod p) means old_r == r
                # So if old_r != r, the sets are disjoint!
                
                # So: remove coverage of old_r, add coverage of r
                # New score = old score + (number of x in r-set that were at count 1) - (number of x in old_r-set that were at count 2)
                
                # Count x in [1, WORK_L] with x ≡ r (mod p) and counts[x] == 1
                # Count x in [1, WORK_L] with x ≡ old_r (mod p) and counts[x] == 2
                
                gain_r = 0
                for x in range(r if r >= 1 else r + p, WORK_L + 1, p):
                    if counts[x] == 1:
                        gain_r += 1
                
                loss_old = 0
                for x in range(old_r if old_r >= 1 else old_r + p, WORK_L + 1, p):
                    if counts[x] == 2:
                        loss_old += 1
                
                new_score = current_score + gain_r - loss_old
                
                if new_score > best_score:
                    best_score = new_score
                    best_r = r
            
            if best_r != old_r:
                # Apply change
                # Remove old coverage
                for x in range(old_r if old_r >= 1 else old_r + p, WORK_L + 1, p):
                    counts[x] -= 1
                # Add new coverage
                for x in range(best_r if best_r >= 1 else best_r + p, WORK_L + 1, p):
                    counts[x] += 1
                residues[idx] = best_r
                current_score = best_score
                improved = True
        
        # Occasionally recompute actual score to verify
        if time.time() < deadline - 15.0:
            actual = _compute_score_fast(residues, primes, WORK_L)
            if actual != current_score:
                current_score = actual
                # Rebuild counts
                counts = [0] * (WORK_L + 1)
                for i, (p, r) in enumerate(zip(primes, residues)):
                    for x in range(r if r >= 1 else r + p, WORK_L + 1, p):
                        counts[x] += 1
    
    # Final: compute full score
    if time.time() < deadline - 2.0:
        _compute_score(residues, primes)
    
    return residues
# EVOLVE-BLOCK-END
