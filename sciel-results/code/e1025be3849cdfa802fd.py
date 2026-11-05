import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Generate primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i in range(2, limit+1) if sieve[i]]
    
    primes = get_primes(1000)
    n = len(primes)  # 168
    
    # Seed solution
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    def compute_L(res):
        """Compute the largest L such that every integer 1..L has at least 2 covers."""
        # For each x, count how many primes p have x ≡ res[i] (mod p)
        # We need to find the first x where count < 2
        # Use an array for counts
        max_check = 100000
        # For each prime i, it covers all x = res[i] + k*p for k >= 0
        # We need count[x] >= 2 for all x in 1..L
        counts = [0] * (max_check + 1)
        for i, p in enumerate(primes):
            r = res[i]
            # x = r, r+p, r+2p, ... but x must be >= 1
            # x = r + k*p >= 1 => k >= ceil((1-r)/p)
            # Start from the smallest x >= 1 in this residue class
            if r == 0:
                start_x = p  # x = p, 2p, 3p, ...
            else:
                start_x = r  # x = r, r+p, ...
            # But we need x >= 1, and r is in [0, p-1]
            # If r == 0, x = p, 2p, ...
            # If r > 0, x = r, r+p, ...
            for x in range(start_x, max_check + 1, p):
                counts[x] += 1
        # Find largest L where all 1..L have count >= 2
        L = 0
        for x in range(1, max_check + 1):
            if counts[x] >= 2:
                L = x
            else:
                break
        return L
    
    def compute_L_fast(res, max_check=20000):
        """Faster computation focusing on relevant range."""
        counts = [0] * (max_check + 1)
        for i, p in enumerate(primes):
            r = res[i]
            if r == 0:
                start_x = p
            else:
                start_x = r
            for x in range(start_x, max_check + 1, p):
                counts[x] += 1
        L = 0
        for x in range(1, max_check + 1):
            if counts[x] >= 2:
                L = x
            else:
                break
        return L
    
    best_res = residues[:]
    best_L = compute_L_fast(best_res)
    
    # Local search
    random.seed(42)
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        improved = False
        # Try random order of indices
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            p = primes[i]
            old_r = best_res[i]
            # Try a few random new values
            for _ in range(5):
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
                best_res[i] = new_r
                new_L = compute_L_fast(best_res)
                if new_L > best_L:
                    best_L = new_L
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    best_res[i] = old_r
            else:
                # No improvement from this index
                pass
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
        else:
            no_improve_count = 0
    
    # Final verification with full range
    final_L = compute_L(best_res)
    if final_L > best_L:
        best_L = final_L
    
    # Ensure pure Python ints
    return [int(r) for r in best_res]
