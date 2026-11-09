import time
import random

def construct():
    # Primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)
    n_primes = len(primes)
    
    # Initial residues from seed
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    MAX_L = 10000 # Cap for search, task says capped at 10^5 but 1000 is target. 10000 is safe.
    
    def compute_counts(res):
        counts = [0] * (MAX_L + 1)
        for i, p in enumerate(primes):
            r = res[i]
            # x in 1..MAX_L such that x % p == r
            # First x >= 1
            start = r
            if start < 1:
                start += p
            for x in range(start, MAX_L + 1, p):
                counts[x] += 1
        return counts

    def find_first_gap(counts):
        for x in range(1, MAX_L + 1):
            if counts[x] < 2:
                return x
        return MAX_L + 1

    counts = compute_counts(residues)
    current_L = find_first_gap(counts) - 1
    best_L = current_L
    best_residues = residues[:]
    
    start_time = time.time()
    time_limit = 145.0 # Leave 5s margin
    
    # Precompute indices for fast updates?
    # For each prime p, the indices x it covers.
    # But residues change, so the set of x changes.
    # However, the set of x covered by p with residue r is {x | x % p == r}.
    # We can generate this list on the fly.
    
    def get_covered_indices(p, r):
        start = r
        if start < 1:
            start += p
        return range(start, MAX_L + 1, p)

    while time.time() - start_time < time_limit:
        gap = find_first_gap(counts)
        if gap > MAX_L:
            break
        
        # Try to fix 'gap'
        # Case 1: counts[gap] == 1
        if counts[gap] == 1:
            # Find the prime p_cover that currently covers gap
            p_cover_idx = -1
            for i, p in enumerate(primes):
                if gap % p == residues[i]:
                    p_cover_idx = i
                    break
            
            # Try changing any other prime p to cover gap
            found = False
            # Randomize order to escape local minima?
            # Let's try all, pick the one with least "risk"
            best_p_idx = -1
            best_risk = float('inf')
            
            for i, p in enumerate(primes):
                if i == p_cover_idx:
                    continue
                old_r = residues[i]
                new_r = gap % p
                
                # Check if changing p to new_r breaks any x < gap
                # We remove coverage from x where x % p == old_r
                # We add coverage to x where x % p == new_r
                # Risk: Any x < gap where counts[x] == 2 and x % p == old_r
                # Because removing p's cover makes it 1.
                
                risk = 0
                # Iterate x < gap covered by old_r
                start = old_r
                if start < 1:
                    start += p
                # We only care about x < gap
                end = gap
                for x in range(start, end, p):
                    if counts[x] == 2:
                        risk += 1
                        # Early exit if risk is high?
                        if risk > best_risk:
                            break
                
                if risk < best_risk:
                    best_risk = risk
                    best_p_idx = i
            
            if best_p_idx != -1 and best_risk == 0:
                # Safe to change
                i = best_p_idx
                p = primes[i]
                old_r = residues[i]
                new_r = gap % p
                
                # Apply change
                # Decrement counts for old_r
                start = old_r
                if start < 1:
                    start += p
                for x in range(start, MAX_L + 1, p):
                    counts[x] -= 1
                # Increment counts for new_r
                start = new_r
                if start < 1:
                    start += p
                for x in range(start, MAX_L + 1, p):
                    counts[x] += 1
                
                residues[i] = new_r
                current_L = find_first_gap(counts) - 1
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
                found = True
        
        # Case 2: counts[gap] == 0
        elif counts[gap] == 0:
            # Need to change 2 primes
            # This is more complex. 
            # We can pick two primes p1, p2.
            # Change r_p1 to gap%p1, r_p2 to gap%p2.
            # Check risk for both.
            # Risk: x < gap where counts[x] == 2 and (x%p1==old_r1 or x%p2==old_r2)
            # And we must ensure we don't double count if p1==p2 (impossible).
            
            # To keep it simple, we can try pairs of primes.
            # But 168 choose 2 is ~14k pairs. For each pair, we check x < gap.
            # This might be slow if done every iteration.
            # Let's limit this to a smaller subset or only do it if stuck.
            
            # Alternative: Just pick one prime, change it. Now counts[gap]==1.
            # Next iteration will handle the second one.
            # This greedy step-by-step is safer.
            found = False
            best_p_idx = -1
            best_risk = float('inf')
            
            for i, p in enumerate(primes):
                old_r = residues[i]
                new_r = gap % p
                
                risk = 0
                start = old_r
                if start < 1:
                    start += p
                end = gap
                for x in range(start, end, p):
                    if counts[x] == 2:
                        risk += 1
                        if risk > best_risk:
                            break
                
                if risk < best_risk:
                    best_risk = risk
                    best_p_idx = i
            
            if best_p_idx != -1 and best_risk == 0:
                i = best_p_idx
                p = primes[i]
                old_r = residues[i]
                new_r = gap % p
                
                start = old_r
                if start < 1:
                    start += p
                for x in range(start, MAX_L + 1, p):
                    counts[x] -= 1
                start = new_r
                if start < 1:
                    start += p
                for x in range(start, MAX_L + 1, p):
                    counts[x] += 1
                
                residues[i] = new_r
                current_L = find_first_gap(counts) - 1
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
                found = True
        
        if not found:
            # Stuck. Try a random restart or perturbation?
            # Or just break to save time.
            # Let's try a few random changes to escape local minima.
            stuck_count = 0
            for _ in range(10):
                if time.time() - start_time > time_limit:
                    break
                i = random.randint(0, n_primes - 1)
                p = primes[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
                
                # Check risk
                risk = 0
                start = old_r
                if start < 1:
                    start += p
                end = gap
                for x in range(start, end, p):
                    if counts[x] == 2:
                        risk += 1
                        if risk > 0:
                            break
                
                if risk == 0:
                    # Apply
                    start = old_r
                    if start < 1:
                        start += p
                    for x in range(start, MAX_L + 1, p):
                        counts[x] -= 1
                    start = new_r
                    if start < 1:
                        start += p
                    for x in range(start, MAX_L + 1, p):
                        counts[x] += 1
                    
                    residues[i] = new_r
                    current_L = find_first_gap(counts) - 1
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues[:]
                    # Don't break, try to find more improvements
            # If still stuck, break to avoid infinite loop of no progress
            # Actually, the main loop will just keep finding the same gap and failing.
            # We should break if we haven't improved in a while?
            # For now, just let the time limit handle it.
            # But if `found` is false, the loop repeats with same gap.
            # We should break if no improvement possible.
            break

    return best_residues

construct()
