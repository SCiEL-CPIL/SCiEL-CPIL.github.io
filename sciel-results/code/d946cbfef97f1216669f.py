import time
import random
import numpy as np

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    # Generate primes <= 1000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    
    # Max L to consider
    MAX_L = 10000
    
    # Precompute indices for each prime and residue
    # indices[p_idx][r] = list of y in 1..MAX_L such that y % p == r
    indices = []
    for p in primes:
        idx_list = [[] for _ in range(p)]
        for y in range(1, MAX_L + 1):
            r = y % p
            idx_list[r].append(y)
        indices.append(idx_list)
    
    # Initial residues (seed)
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    # Compute initial cover
    cover = np.zeros(MAX_L + 1, dtype=int)
    for i, p in enumerate(primes):
        r = residues[i]
        for y in indices[i][r]:
            cover[y] += 1
    
    def find_L(cover):
        # Find largest L such that cover[1..L] >= 2
        # cover is 1-indexed in our logic, but array is 0-indexed.
        # We care about 1..MAX_L
        for x in range(1, MAX_L + 1):
            if cover[x] < 2:
                return x - 1
        return MAX_L
    
    L = find_L(cover)
    
    # Greedy Repair Loop
    while time.time() < deadline:
        x = L + 1
        if x > MAX_L:
            break
            
        best_p_idx = -1
        best_r_new = -1
        
        # Try to find a prime to fix x
        # Iterate primes in random order or decreasing p?
        # Larger p has fewer collisions, might be safer?
        # Let's try random order to avoid bias
        p_indices = list(range(n_primes))
        random.shuffle(p_indices)
        
        valid_moves = []
        
        for i in p_indices:
            p = primes[i]
            r_new = x % p
            r_old = residues[i]
            
            if r_new == r_old:
                continue
                
            # Check if changing r_old to r_new breaks any y < x
            # We need to ensure no y < x with y % p == r_old has cover[y] == 2
            # Because if cover[y] == 2 and we remove one cover, it becomes 1.
            
            broken = False
            # Iterate y in indices[i][r_old] that are < x
            # indices[i][r_old] is sorted
            for y in indices[i][r_old]:
                if y >= x:
                    break
                if cover[y] == 2:
                    broken = True
                    break
            
            if not broken:
                valid_moves.append((i, r_new))
                
        if valid_moves:
            # Pick a move. Heuristic: pick the one that adds coverage to more future points?
            # Or just pick the first one?
            # Let's pick the one with the largest p (least likely to interfere)
            # Or random?
            # Let's pick random to allow exploration
            i, r_new = random.choice(valid_moves)
            
            r_old = residues[i]
            
            # Apply move
            # Decrement cover for y in indices[i][r_old]
            for y in indices[i][r_old]:
                cover[y] -= 1
            # Increment cover for y in indices[i][r_new]
            for y in indices[i][r_new]:
                cover[y] += 1
                
            residues[i] = r_new
            
            # Update L
            new_L = find_L(cover)
            if new_L >= L:
                L = new_L
            else:
                # Should not happen if we checked validity for y < x
                # But maybe x itself is now covered?
                # If new_L < L, something went wrong or we broke an earlier point?
                # Our check ensured no y < x dropped to 1.
                # So L should be at least x-1 = L.
                # If L didn't increase, it means x is still not covered by 2?
                # We added cover to x (since r_new = x % p).
                # So cover[x] increased by 1.
                # If cover[x] was 1, it is now 2. So L should become at least x.
                # If cover[x] was 0, it is now 1. L stays x-1.
                # If cover[x] was >= 2, it is now >= 3. L stays or increases.
                # So L should not decrease.
                pass
        else:
            # Stuck. Try random perturbation?
            # Pick a random prime and change its residue randomly.
            # Then try to repair.
            # This is risky but might escape local minima.
            if random.random() < 0.1: # 10% chance to perturb
                i = random.randint(0, n_primes - 1)
                old_r = residues[i]
                new_r = random.randint(0, primes[i] - 1)
                
                # Apply
                for y in indices[i][old_r]:
                    cover[y] -= 1
                for y in indices[i][new_r]:
                    cover[y] += 1
                residues[i] = new_r
                
                # Repair
                # Just let the loop continue, it will try to fix the new L
                # But we might have dropped L significantly.
                # To prevent infinite loop of bad perturbations, we could keep a history.
                # For now, just continue.
            else:
                # If stuck and no perturbation, break?
                # Or try different heuristic?
                # Let's break to avoid wasting time if greedy is exhausted.
                break
                
    return residues

# EVOLVE-BLOCK-END
