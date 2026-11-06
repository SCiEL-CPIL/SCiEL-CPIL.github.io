"""Structural approach: greedy gap-closure with fast numpy coverage + local search."""
import time
import numpy as np

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    n_primes = len(primes)  # 168
    
    # Use a moderate range; extend if time permits
    MAX_L = 15000
    
    # NumPy coverage array
    coverage = np.zeros(MAX_L + 1, dtype=np.int32)
    residues = np.zeros(n_primes, dtype=np.int32)
    
    def add_congruence(idx, r, cov):
        p = primes[idx]
        start = r if r > 0 else p
        cov[start::p] += 1
        
    def remove_congruence(idx, r, cov):
        p = primes[idx]
        start = r if r > 0 else p
        cov[start::p] -= 1
    
    def find_first_gap(cov, start=1):
        """Find first index >= start where coverage < 2. Returns MAX_L+1 if none."""
        if start > MAX_L:
            return MAX_L + 1
        # Use numpy search for speed
        seg = cov[start:]
        if len(seg) == 0:
            return MAX_L + 1
        gaps = np.where(seg < 2)[0]
        if len(gaps) == 0:
            return MAX_L + 1
        return start + gaps[0]

    # Phase 1: Greedy Construction
    # Process primes in increasing order. For each, pick r to cover the current gap.
    current_gap = 1
    for idx, p in enumerate(primes):
        if time.time() > deadline - 40:
            break
            
        # Ensure current_gap is valid
        if current_gap <= MAX_L:
            # Find the true first gap from current position
            gap = find_first_gap(coverage, current_gap)
        else:
            gap = MAX_L + 1
            
        if gap > MAX_L:
            r = 0
        else:
            r = gap % p
        
        residues[idx] = r
        add_congruence(idx, r, coverage)
        
        # Update current_gap: after adding, the gap moves forward
        # We don't need to rescan from 1; scan from old gap
        if gap <= MAX_L:
            current_gap = find_first_gap(coverage, gap)
    
    # Phase 2: Local Search / Hill Climbing
    # Try to improve L by changing residues to close gaps.
    
    current_L = find_first_gap(coverage, 1) - 1
    if current_L > MAX_L:
        current_L = MAX_L
    
    improved = True
    no_improve_count = 0
    max_no_improve = 50
    
    while improved and time.time() < deadline - 15 and no_improve_count < max_no_improve:
        improved = False
        
        gap = find_first_gap(coverage, 1)
        
        if gap > MAX_L:
            break
        
        cov_gap = coverage[gap]
        
        # If cov_gap >= 2, gap isn't actually a gap (shouldn't happen)
        # If cov_gap == 1, we need one more prime to cover gap
        # If cov_gap == 0, we need two more primes
        
        # Strategy: find a prime whose residue change would cover gap
        # and verify it doesn't break earlier coverage.
        
        # Quick filter: only consider primes where old_r != gap % p
        # and prefer primes where the old residue covers few positions < gap
        
        candidates = []
        for idx in range(n_primes):
            p = primes[idx]
            target_r = gap % p
            old_r = residues[idx]
            if old_r == target_r:
                continue
            # Estimate impact: how many positions < gap does old_r cover?
            # Positions: old_r, old_r+p, ... < gap
            # Count = max(0, (gap - 1 - old_r) // p + 1) if old_r > 0 else (gap-1)//p
            if old_r > 0:
                old_count = max(0, (gap - 1 - old_r) // p + 1)
            else:
                old_count = max(0, (gap - 1) // p)
            candidates.append((old_count, idx))
        
        # Sort by old_count (prefer primes that cover few positions < gap)
        candidates.sort()
        
        found_improvement = False
        for old_count, idx in candidates[:30]:  # Limit to top 30 candidates
            if time.time() > deadline - 15:
                break
                
            p = primes[idx]
            target_r = gap % p
            old_r = residues[idx]
            
            # Try the change
            remove_congruence(idx, old_r, coverage)
            add_congruence(idx, target_r, coverage)
            
            new_L = find_first_gap(coverage, 1) - 1
            if new_L > current_L:
                current_L = new_L
                residues[idx] = target_r
                improved = True
                found_improvement = True
                no_improve_count = 0
                break
            else:
                # Revert
                remove_congruence(idx, target_r, coverage)
                add_congruence(idx, old_r, coverage)
        
        if not found_improvement:
            no_improve_count += 1
            
            # Try a different strategy: look at positions near the gap
            # and try to fix them with any prime
            if no_improve_count < max_no_improve:
                # Try changing residues for primes to cover positions in [gap, gap+10]
                for offset in range(10):
                    if time.time() > deadline - 15:
                        break
                    pos = gap + offset
                    if pos > MAX_L:
                        break
                    cov_pos = coverage[pos]
                    if cov_pos >= 2:
                        continue
                    # Need to increase coverage at pos
                    needed = 2 - cov_pos
                    # Find primes that don't currently cover pos
                    for idx in range(n_primes):
                        if time.time() > deadline - 15:
                            break
                        p = primes[idx]
                        target_r = pos % p
                        old_r = residues[idx]
                        if old_r == target_r:
                            continue
                        # Try change
                        remove_congruence(idx, old_r, coverage)
                        add_congruence(idx, target_r, coverage)
                        new_L = find_first_gap(coverage, 1) - 1
                        if new_L > current_L:
                            current_L = new_L
                            residues[idx] = target_r
                            improved = True
                            found_improvement = True
                            no_improve_count = 0
                            break
                        else:
                            remove_congruence(idx, target_r, coverage)
                            add_congruence(idx, old_r, coverage)
                        if improved:
                            break
                    if improved:
                        break
    
    return residues.tolist()
# EVOLVE-BLOCK-END
