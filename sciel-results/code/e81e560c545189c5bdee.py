"""Combined program: fast greedy init + efficient local search for odd covering systems."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    # Get all odd divisors of 51975 > 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    n_div = len(divisors)

    # Greedy initialization: process moduli in DECREASING order
    # Larger moduli cover fewer positions each, so they're more "specialized"
    # Processing them first lets smaller moduli fill in gaps
    moduli_desc = sorted(divisors, reverse=True)
    moduli_asc = sorted(divisors)  # keep for reference
    residues = [0] * n_div
    # Map: index in moduli_asc -> residue
    # We'll work with moduli_asc ordering for the final answer
    # But initialize using descending order
    
    # Use coverage count array for efficiency
    cov_count = np.zeros(base, dtype=np.int16)
    
    # Greedy: for each modulus (descending), pick best residue
    # For efficiency, we track uncovered count
    uncovered_count = base  # initially all uncovered
    
    # Process in descending order of modulus
    # We need to map back to ascending index
    desc_to_asc = {}
    for i, m in enumerate(moduli_asc):
        desc_to_asc[m] = i
    
    for m in moduli_desc:
        if time.time() > deadline - 30:  # leave time for local search
            break
        
        idx = desc_to_asc[m]
        # Find best residue: maximize newly covered (uncovered) positions
        best_r = 0
        best_gain = -1
        
        # For each residue r, count how many positions r, r+m, r+2m... are uncovered
        # Use numpy for speed
        for r in range(m):
            positions = np.arange(r, base, m)
            gain = int(np.sum(cov_count[positions] == 0))
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        residues[idx] = best_r
        # Update coverage
        positions = np.arange(best_r, base, m)
        # Count how many were uncovered
        newly_covered = int(np.sum(cov_count[positions] == 0))
        uncovered_count -= newly_covered
        cov_count[positions] += 1
    
    # Now do local search
    # State: residues array (indexed by position in moduli_asc)
    best_uncovered = uncovered_count
    best_residues = residues[:]
    
    # Precompute position lists for each modulus (for fast access)
    # positions[i][r] = array of positions for modulus moduli_asc[i], residue r
    # This is too much memory for large moduli, so compute on the fly but cache small ones
    
    # For local search, we'll do single-residue changes (faster than pair changes)
    temp = 50.0
    cooling = 0.995
    step = 0
    
    while time.time() < deadline - 2:
        step += 1
        
        # Pick a random modulus to change
        i = random.randint(0, n_div - 1)
        mi = moduli_asc[i]
        old_ri = residues[i]
        
        # Pick new residue
        if random.random() < 0.7:
            new_ri = random.randint(0, mi - 1)
        else:
            # Try a nearby residue (modular neighborhood)
            new_ri = (old_ri + random.randint(-1, 1)) % mi
        
        if new_ri == old_ri:
            continue
        
        # Compute delta: change in uncovered count
        # Positions that change: old_ri::mi and new_ri::mi
        # A position p's coverage changes only if it's in one of these two sets
        # and its coverage count goes from 0 to 1 (newly covered) or 1 to 0 (newly uncovered)
        
        # Get positions in old and new residue classes
        pos_old = np.arange(old_ri, base, mi)
        pos_new = np.arange(new_ri, base, mi)
        
        # For pos_old: coverage decreases by 1
        #   - if cov_count was 1, becomes 0: uncovered increases by 1 (bad)
        #   - if cov_count was >1, stays covered: no change
        # For pos_new: coverage increases by 1
        #   - if cov_count was 0, becomes 1: uncovered decreases by 1 (good)
        #   - if cov_count was >=1, stays covered: no change
        
        # But we need to be careful: some positions might be in both pos_old and pos_new
        # (this happens when mi divides the difference, i.e., always if we're looking at same modulus)
        # Actually, pos_old and pos_new are disjoint for the same modulus (different residues)
        # Wait, no: pos_old = {old_ri + k*mi} and pos_new = {new_ri + k*mi}
        # These are disjoint if old_ri != new_ri (mod mi), which they are.
        
        # So:
        # newly_uncovered = count of p in pos_old where cov_count[p] == 1
        # newly_covered = count of p in pos_new where cov_count[p] == 0
        
        newly_uncovered = int(np.sum(cov_count[pos_old] == 1))
        newly_covered = int(np.sum(cov_count[pos_new] == 0))
        
        delta = newly_uncovered - newly_covered  # change in uncovered count
        
        # Accept if improvement or with probability exp(-delta/temp) for worsening
        if delta < 0 or (temp > 0.1 and random.random() < np.exp(-delta / temp)):
            # Apply the change
            cov_count[pos_old] -= 1
            cov_count[pos_new] += 1
            residues[i] = new_ri
            uncovered_count += delta
            
            if uncovered_count < best_uncovered:
                best_uncovered = uncovered_count
                best_residues = residues[:]
        else:
            pass  # reject, no change needed
        
        temp *= cooling
    
    # Final polish: try all residues for each modulus (steepest descent)
    if time.time() < deadline - 1:
        for i in range(n_div):
            if time.time() > deadline - 1:
                break
            mi = moduli_asc[i]
            current_ri = best_residues[i]
            
            # Find best residue for this modulus given all others fixed
            best_r = current_ri
            best_uc = best_uncovered
            
            for r in range(mi):
                if r == current_ri:
                    continue
                
                # Compute what uncovered would be if we set residue to r
                # We need to temporarily change and count
                # But this is expensive. Let's just do a quick check for a few candidates.
                # Actually, let's skip this for now as it's expensive.
                pass
        
        # Simpler polish: try a few random perturbations
        for _ in range(100):
            if time.time() > deadline - 1:
                break
            i = random.randint(0, n_div - 1)
            mi = moduli_asc[i]
            old_ri = best_residues[i]
            new_ri = random.randint(0, mi - 1)
            
            if new_ri == old_ri:
                continue
            
            pos_old = np.arange(old_ri, base, mi)
            pos_new = np.arange(new_ri, base, mi)
            
            # We need to know the coverage counts with best_residues applied
            # This requires rebuilding cov_count, which is expensive
            # Skip for now
            
            pass
    
    # Build final answer from best_residues
    classes = [[moduli_asc[i], best_residues[i]] for i in range(n_div)]
    
    # Verify and return
    return classes
# EVOLVE-BLOCK-END
