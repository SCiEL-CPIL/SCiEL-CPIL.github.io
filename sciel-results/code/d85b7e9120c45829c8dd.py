import numpy as np
import time

def get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i:limit+1:i] = [False] * len(sieve[i*i:limit+1:i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    primes = get_primes(1000)
    n_primes = len(primes)
    L_max = 5000  # Search up to this L
    
    # Initialize residues
    # Heuristic: r_p = 1 is a start, but we can try r_p = p//2 or random
    residues = np.array([1] * n_primes, dtype=int)
    
    # Initialize counts array
    # counts[x] = number of primes p such that x % p == residues[i]
    counts = np.zeros(L_max + 1, dtype=int)
    
    def update_counts(idx, old_val, new_val, delta):
        """
        Update counts for prime at index idx.
        old_val: old residue
        new_val: new residue
        delta: +1 for adding, -1 for removing
        """
        p = primes[idx]
        # Find indices in [1, L_max] congruent to val mod p
        # We need x such that x % p == val
        # x = k*p + val
        # 1 <= k*p + val <= L_max
        
        # First valid k
        # If val == 0, x must be multiple of p. Smallest x >= 1 is p (if p>=1).
        # If val > 0, smallest x is val (if val >= 1).
        
        # Actually, simpler:
        # start = val if val >= 1 else p
        # But we need x % p == val.
        # If val == 0, x in {p, 2p, ...}
        # If val > 0, x in {val, val+p, ...}
        
        if val == 0:
            start = p
        else:
            start = val
            
        if start > L_max:
            return
            
        indices = np.arange(start, L_max + 1, p)
        counts[indices] += delta

    # Initialize counts
    for i in range(n_primes):
        update_counts(i, residues[i], residues[i], 1)
        
    def get_L():
        # Find largest L such that counts[1..L] >= 2
        # We can use cumulative min or just search
        # Since L_max is small, a loop is fine, but we can optimize.
        # Actually, we just need to check where the first < 2 occurs.
        # np.where(counts[1:L_max+1] < 2)[0]
        bad = np.where(counts[1:L_max+1] < 2)[0]
        if len(bad) == 0:
            return L_max
        return bad[0]
        
    current_L = get_L()
    best_L = current_L
    best_residues = residues.copy()
    
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # Local Search
    no_improve_count = 0
    
    while time.time() - start_time < timeout:
        current_L = get_L()
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # If stuck for a while, perturb
            if no_improve_count > 50:
                # Randomly change a few residues
                for _ in range(5):
                    idx = np.random.randint(n_primes)
                    old_val = residues[idx]
                    new_val = np.random.randint(primes[idx])
                    if new_val != old_val:
                        update_counts(idx, old_val, old_val, -1)
                        residues[idx] = new_val
                        update_counts(idx, new_val, new_val, 1)
                no_improve_count = 0
                continue
                
        # Try to fix the first failing integer
        if current_L >= L_max:
            break
            
        target_x = current_L + 1
        
        # Check if we can fix target_x by changing one residue
        improved = False
        # Iterate over primes in random order to avoid bias
        prime_indices = np.random.permutation(n_primes)
        
        for i in prime_indices:
            p = primes[i]
            old_val = residues[i]
            new_val = target_x % p
            
            if new_val == old_val:
                continue
                
            # Simulate change
            # We need to check if min(counts[1..current_L]) stays >= 2
            # And if counts[target_x] increases (which it will, by 1)
            
            # To be safe and fast, we can just apply, check, and revert if bad.
            # Applying is O(L_max/p). Checking min is O(current_L).
            # This is acceptable.
            
            update_counts(i, old_val, old_val, -1)
            residues[i] = new_val
            update_counts(i, new_val, new_val, 1)
            
            new_L = get_L()
            
            if new_L > current_L:
                improved = True
                break
            else:
                # Revert
                update_counts(i, new_val, new_val, -1)
                residues[i] = old_val
                update_counts(i, old_val, old_val, 1)
                
        if not improved:
            # Try to improve coverage of some random x in [1, L_max]
            # This helps "prepare" the system for extension
            if np.random.random() < 0.5:
                target_x = np.random.randint(1, L_max + 1)
                if counts[target_x] >= 2:
                    continue # Already good
                
                prime_indices = np.random.permutation(n_primes)
                for i in prime_indices:
                    p = primes[i]
                    old_val = residues[i]
                    new_val = target_x % p
                    if new_val == old_val:
                        continue
                    
                    update_counts(i, old_val, old_val, -1)
                    residues[i] = new_val
                    update_counts(i, new_val, new_val, 1)
                    
                    new_L = get_L()
                    if new_L >= current_L:
                        # Accept if L doesn't drop
                        break
                    else:
                        # Revert
                        update_counts(i, new_val, new_val, -1)
                        residues[i] = old_val
                        update_counts(i, old_val, old_val, 1)

    return best_residues.tolist()
