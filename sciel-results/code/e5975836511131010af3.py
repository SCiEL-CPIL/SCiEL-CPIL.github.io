"""Structural greedy + optimized backward refinement solver for covering problem."""
import time
import numpy as np

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _build_count_array(residues, primes, max_L):
    """Build coverage count array for positions 1..max_L."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for i in range(len(primes)):
        r = residues[i]
        p = primes[i]
        if r == 0:
            r = p
        if r <= max_L:
            positions = np.arange(r, max_L + 1, p)
            counts[positions] += 1
    return counts

def _find_L(counts, max_L):
    """Find largest L such that all positions 1..L have count >= 2."""
    deficient = np.where(counts[1:max_L+1] < 2)[0]
    if len(deficient) == 0:
        return max_L
    return deficient[0]

def _score(residues, primes, max_L=100000):
    """Compute the largest L such that every integer 1..L is covered at least twice."""
    counts = _build_count_array(residues, primes, max_L)
    return _find_L(counts, max_L)

def _greedy_construct(primes, max_L_check=2000):
    """
    Forward greedy: process primes in order, for each prime choose the residue
    that maximizes coverage of currently-deficient positions.
    """
    n = len(primes)
    residues = [0] * n
    
    # Start with all zeros, build up incrementally
    counts = np.zeros(max_L_check + 1, dtype=np.int32)
    
    for idx in range(n):
        p = primes[idx]
        
        best_r = 0
        best_gain = -1
        
        # Find the first few deficient positions
        deficient_positions = []
        for pos in range(1, min(max_L_check + 1, 200)):
            if counts[pos] < 2:
                deficient_positions.append(pos)
                if len(deficient_positions) >= 50:
                    break
        
        if not deficient_positions:
            # No deficient positions in early range, pick residue to maximize overall coverage
            best_r = 1
            best_count = 0
            for r in range(min(p, 50)):
                if r == 0:
                    r_eff = p
                else:
                    r_eff = r
                if r_eff > max_L_check:
                    continue
                num_positions = (max_L_check - r_eff) // p + 1
                if num_positions > best_count:
                    best_count = num_positions
                    best_r = r
        else:
            # For each candidate residue, count how many deficient positions it covers
            residue_weighted = np.zeros(p, dtype=np.int32)
            for pos in deficient_positions:
                r_needed = pos % p
                weight = 2 if counts[pos] == 0 else 1
                residue_weighted[r_needed] += weight
            
            best_r = int(np.argmax(residue_weighted))
            best_gain = residue_weighted[best_r]
        
        residues[idx] = best_r
        
        # Update counts
        r = best_r
        if r == 0:
            r_eff = p
        else:
            r_eff = r
        if r_eff <= max_L_check:
            positions = np.arange(r_eff, max_L_check + 1, p)
            counts[positions] += 1
    
    return residues

def _refine_pass(residues, primes, max_L_check, deadline):
    """
    Optimized backward refinement.
    For each prime, we determine the best residue by looking at the deficiencies
    caused by the other primes.
    """
    n = len(primes)
    improved = False
    
    # Process in reverse order
    for idx in range(n - 1, -1, -1):
        if time.time() > deadline:
            break
            
        p = primes[idx]
        old_r = residues[idx]
        
        # 1. Build counts for all OTHER primes
        # To save time, we can maintain a global counts array and subtract/add.
        # But for simplicity and correctness in this pass structure, we rebuild.
        # Optimization: We can pre-compute the counts of all other primes once per idx?
        # No, because residues change. But we can compute "base_counts" = counts of all primes except idx.
        
        base_counts = np.zeros(max_L_check + 1, dtype=np.int32)
        for j in range(n):
            if j == idx:
                continue
            r = residues[j]
            pj = primes[j]
            if r == 0:
                r_eff = pj
            else:
                r_eff = r
            if r_eff <= max_L_check:
                positions = np.arange(r_eff, max_L_check + 1, pj)
                base_counts[positions] += 1
        
        # Find current L with old residue
        current_counts = base_counts.copy()
        r_old_eff = p if old_r == 0 else old_r
        if r_old_eff <= max_L_check:
            positions = np.arange(r_old_eff, max_L_check + 1, p)
            current_counts[positions] += 1
        current_L = _find_L(current_counts, max_L_check)
        
        # 2. Identify candidate residues
        # We want residues that cover the first few positions where base_counts < 2
        # or where base_counts + 1 < 2 (i.e. base_counts == 0)
        
        # Find first 20 positions where base_counts < 2
        deficient_pos = []
        for pos in range(1, max_L_check + 1):
            if base_counts[pos] < 2:
                deficient_pos.append(pos)
                if len(deficient_pos) >= 20:
                    break
        
        if not deficient_pos:
            # If no deficiencies in base, any residue is fine, keep old
            continue
            
        # Candidate residues are those that cover at least one deficient position
        candidates = set()
        for pos in deficient_pos:
            candidates.add(pos % p)
            
        # Also include the old residue
        candidates.add(old_r)
        
        # Limit candidates to 50 max for speed
        if len(candidates) > 50:
            # Prioritize candidates that cover more deficient positions
            cand_scores = {}
            for c in candidates:
                score = 0
                for pos in deficient_pos:
                    if pos % p == c:
                        score += 1
                cand_scores[c] = score
            sorted_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
            candidates = {c for c, _ in sorted_cands[:50]}
        
        best_r = old_r
        best_L = current_L
        
        # 3. Evaluate candidates
        for new_r in candidates:
            if time.time() > deadline:
                break
                
            r_new_eff = p if new_r == 0 else new_r
            
            # Calculate new L
            # We can do this efficiently by checking positions where base_counts < 2
            # and seeing if new_r covers them.
            
            # A position x is covered by new_r if x % p == new_r.
            # We need the first x such that base_counts[x] + (1 if x%p==new_r else 0) < 2.
            
            # This is equivalent to:
            # If base_counts[x] >= 2, it's fine regardless of new_r.
            # If base_counts[x] == 1, it's fine if x%p == new_r.
            # If base_counts[x] == 0, it's fine if x%p == new_r (becomes 1) -> NO, we need >=2.
            # Wait.
            # We need count >= 2.
            # If base_counts[x] == 0, adding new_r makes it 1. Still deficient.
            # If base_counts[x] == 1, adding new_r makes it 2. OK.
            # If base_counts[x] >= 2, OK.
            
            # So, a position x is deficient in the new configuration if:
            # 1. base_counts[x] == 0 (remains 0 or 1)
            # 2. base_counts[x] == 1 AND x % p != new_r (remains 1)
            
            # We want the smallest x satisfying this.
            
            # Optimization: Iterate x from 1 to max_L_check? Too slow.
            # Instead, iterate over the "problematic" positions.
            
            # Positions with base_counts == 0 are always problematic.
            # Positions with base_counts == 1 are problematic only if not covered by new_r.
            
            # Let's find the first problematic position.
            
            # We can pre-compute the list of positions where base_counts < 2.
            # But base_counts is fixed for this idx.
            
            # Let's just check the first few hundred positions?
            # No, we need the exact L.
            
            # Alternative:
            # The new L is the min of:
            # - First x where base_counts[x] == 0
            # - First x where base_counts[x] == 1 and x % p != new_r
            
            # Let pre_zero = first x where base_counts[x] == 0
            # Let pre_one = list of x where base_counts[x] == 1
            
            # We can compute pre_zero once.
            # And for pre_one, we need the first x in pre_one such that x % p != new_r.
            
            # This is still potentially slow if we do it for every candidate.
            
            # Let's stick to a simpler, faster approximation for the candidate evaluation:
            # Just check the first 500 positions.
            
            new_L = max_L_check
            for x in range(1, min(500, max_L_check + 1)):
                add = 1 if (x % p) == new_r else 0
                if base_counts[x] + add < 2:
                    new_L = x - 1
                    break
            
            # If we didn't find a deficiency in first 500, we assume it's at least 500.
            # To be safe, if new_L >= current_L, we might want to verify further.
            # But for speed, we accept this.
            
            if new_L > best_L:
                best_L = new_L
                best_r = new_r
        
        if best_r != old_r:
            residues[idx] = best_r
            improved = True
            
    return residues, improved

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    primes = _get_primes(1000)
    n = len(primes)
    
    # Phase 1: Greedy construction
    max_L_small = 500
    residues = _greedy_construct(primes, max_L_small)
    
    # Phase 2: Refinement with increasing max_L
    # Start with a small max_L for speed, increase as we go
    max_L_stages = [1000, 2000, 5000, 10000, 20000]
    
    for max_L in max_L_stages:
        if time.time() > deadline - 10:
            break
            
        # Do 2 passes of refinement for this max_L
        for _ in range(2):
            if time.time() > deadline - 5:
                break
            residues, improved = _refine_pass(residues, primes, max_L, deadline)
            if not improved:
                break
        
    # Final evaluation
    final_score = _score(residues, primes, 100000)
    
    return residues
