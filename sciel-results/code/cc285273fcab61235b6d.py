"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Generate primes <= 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n = len(primes)
    
    # Initial residues: random or simple pattern
    # Let's start with r_p = 1 for all, then let search fix it.
    # Or better, r_p = (i+1) % p as in current, which spreads them out.
    residues = [(i + 1) % p for i, p in enumerate(primes)]
    
    # Dynamic cap
    cap = 1000
    # coverage[x] = number of primes p such that x % p == residues[idx]
    # We only care about x >= 1
    coverage = [0] * (cap + 2) # +2 for safety
    
    def build_coverage():
        nonlocal coverage
        coverage = [0] * (cap + 2)
        for r, p in zip(residues, primes):
            # first x in 1..cap such that x % p == r
            # if r == 0, first is p (since x>=1, 0 is not in 1..cap usually, but 0%p=0. 
            # Task says 0 <= r_p < p. x=1..L.
            # If r=0, x must be multiple of p. Smallest x>=1 is p.
            # If r>0, x must be r, r+p, ...
            # Note: if r > cap, no contribution? No, r < p <= 1000. cap grows.
            if r == 0:
                first = p
            else:
                first = r
            if first > cap:
                continue
            x = first
            while x <= cap:
                coverage[x] += 1
                x += p

    build_coverage()
    
    # Find current L
    current_L = 0
    for i in range(1, cap + 1):
        if coverage[i] >= 2:
            current_L = i
        else:
            break
            
    rng = random.Random(42)
    
    # Weights for selecting primes: prefer smaller primes as they affect more small numbers
    # Weight ~ 1/p
    weights = [1.0 / p for p in primes]
    total_weight = sum(weights)
    # Cumulative weights for random selection
    cum_weights = []
    s = 0
    for w in weights:
        s += w
        cum_weights.append(s)
        
    def select_prime():
        r = rng.random() * total_weight
        # Binary search or linear scan (n=168, linear is fine)
        for i, cw in enumerate(cum_weights):
            if r <= cw:
                return i
        return n - 1

    def get_first_gap(start):
        """Returns the first index >= start where coverage < 2, or cap+1 if none found in range."""
        for i in range(start, cap + 1):
            if coverage[i] < 2:
                return i
        return cap + 1

    moves = 0
    while time.time() < deadline:
        moves += 1
        
        # Determine strategy
        # If we are stuck (gap at current_L + 1), try to fix it.
        # First, check if current_L is still valid? 
        # We only accept moves that don't decrease L, so L is always valid.
        
        gap = get_first_gap(current_L + 1)
        
        if gap == cap + 1:
            # No gap in current cap. Expand cap.
            cap *= 2
            old_cap = cap // 2
            # Extend coverage array
            coverage.extend([0] * (cap - old_cap + 2))
            # Update coverage for the new range [old_cap+1, cap]
            for r, p in zip(residues, primes):
                # Find first x in [old_cap+1, cap] such that x % p == r
                # We can just iterate from old_cap+1 to cap
                # Or calculate start
                # start = smallest x >= old_cap+1 such that x % p == r
                # x = r + k*p
                # k = ceil((old_cap+1 - r) / p)
                if r == 0:
                    r_eff = p
                else:
                    r_eff = r
                
                # Calculate k
                # We want x >= old_cap + 1
                # x = r_eff + k*p
                # k >= (old_cap + 1 - r_eff) / p
                k = 0
                if r_eff < old_cap + 1:
                    k = (old_cap + 1 - r_eff + p - 1) // p
                else:
                    k = 0
                x = r_eff + k * p
                while x <= cap:
                    coverage[x] += 1
                    x += p
            
            # Now check if L increased
            new_L = get_first_gap(current_L + 1)
            if new_L == cap + 1:
                current_L = cap
            else:
                current_L = new_L - 1
            continue

        # We have a gap at 'gap'.
        # Strategy 1 (80%): Targeted fix
        # Pick a prime p that does NOT cover 'gap'
        # Change its residue to 'gap' % p
        
        if rng.random() < 0.8:
            # Find primes that don't cover gap
            # coverage[gap] < 2, so at most 1 prime covers it.
            # Actually, we want to add a cover to 'gap'.
            # So we need a prime p such that residues[idx] != gap % p.
            
            # Select a prime weighted by 1/p
            # To ensure we pick one that doesn't cover gap, we might need to retry or filter
            # Given 168 primes, and gap is covered by < 2, most primes don't cover it.
            
            idx = select_prime()
            p = primes[idx]
            old_r = residues[idx]
            
            # If this prime already covers gap, pick another? 
            # Or just skip? 
            # If old_r == gap % p, it already covers it. Changing it won't help gap.
            # We want to change a prime that DOESN'T cover gap.
            
            if old_r == gap % p:
                # This prime covers gap. We should pick one that doesn't.
                # Try a few times
                found = False
                for _ in range(5):
                    idx2 = select_prime()
                    p2 = primes[idx2]
                    if residues[idx2] != gap % p2:
                        idx = idx2
                        p = p2
                        old_r = residues[idx]
                        found = True
                        break
                if not found:
                    continue
            
            new_r = gap % p
            
            if new_r == old_r:
                continue
                
            # Apply move
            # Remove old
            old_first = old_r if old_r > 0 else p
            x = old_first
            while x <= cap:
                coverage[x] -= 1
                x += p
                
            # Add new
            new_first = new_r if new_r > 0 else p
            x = new_first
            while x <= cap:
                coverage[x] += 1
                x += p
            
            # Check if L improved
            # The gap at 'gap' is now fixed (coverage[gap] >= 2).
            # But did we break something before 'gap'?
            # We only changed residue for p.
            # The points affected are multiples of p.
            # If we broke a point y < gap, L decreases.
            # If we didn't break y < gap, L might increase if the next gap is further.
            
            # Check if any point in [1, gap] is now < 2?
            # Actually, we only need to check if the first gap is still >= gap?
            # No, if we broke a point y < gap, the first gap is y.
            
            # Efficient check:
            # We know gap was the first gap.
            # If we fixed gap, the new first gap is either:
            # 1. A point y < gap that we broke.
            # 2. The next gap after gap.
            
            # Let's just find the new first gap from 1? 
            # That's O(L). 
            # Better: check if any point in [1, gap-1] is < 2?
            # Since gap was the first gap, all points in [1, gap-1] were >= 2.
            # We only decreased coverage for points covered by old_r.
            # So we only need to check if any point in [1, gap-1] that was covered by old_r is now < 2.
            # And check the next gap after gap.
            
            # Simple approach:
            new_gap = get_first_gap(1)
            
            if new_gap > gap:
                # Improved
                residues[idx] = new_r
                current_L = new_gap - 1
            else:
                # Revert
                # Remove new
                x = new_first
                while x <= cap:
                    coverage[x] -= 1
                    x += p
                # Add old
                x = old_first
                while x <= cap:
                    coverage[x] += 1
                    x += p
        else:
            # Strategy 2 (20%): Random move
            idx = select_prime()
            p = primes[idx]
            old_r = residues[idx]
            new_r = rng.randint(0, p - 1)
            if new_r == old_r:
                continue
                
            old_first = old_r if old_r > 0 else p
            x = old_first
            while x <= cap:
                coverage[x] -= 1
                x += p
                
            new_first = new_r if new_r > 0 else p
            x = new_first
            while x <= cap:
                coverage[x] += 1
                x += p
                
            new_gap = get_first_gap(1)
            if new_gap > current_L + 1: # Actually, if new_gap - 1 > current_L
                residues[idx] = new_r
                current_L = new_gap - 1
            else:
                # Revert
                x = new_first
                while x <= cap:
                    coverage[x] -= 1
                    x += p
                x = old_first
                while x <= cap:
                    coverage[x] += 1
                    x += p

    return residues
# EVOLVE-BLOCK-END
