import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Generate primes <= 1000
    primes = []
    for i in range(2, 1001):
        is_prime = True
        for j in range(2, int(i**0.5) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    
    n_primes = len(primes)
    # Initial residues: use the seed's heuristic as a starting point
    # Seed: r_p = ((i + 2) // 2) % p
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    # We need to track coverage counts.
    # The maximum L we care about is likely around 1000-2000.
    # Let's set a dynamic limit.
    max_limit = 2000
    counts = [0] * (max_limit + 1)
    
    def update_counts(res, primes, counts, max_l):
        # Reset counts
        for i in range(max_l + 1):
            counts[i] = 0
        # Add contributions
        for i, p in enumerate(primes):
            r = res[i]
            # x = r, r+p, r+2p... <= max_l
            # Note: x must be >= 1.
            # r can be 0. If r=0, x=p, 2p...
            # If r>0, x=r, r+p...
            start = r
            if start == 0:
                start = p
            if start > max_l:
                continue
            for x in range(start, max_l + 1, p):
                counts[x] += 1

    # Initial update
    update_counts(residues, primes, counts, max_limit)
    
    def get_bottleneck(counts, max_l):
        for x in range(1, max_l + 1):
            if counts[x] < 2:
                return x
        return max_l + 1

    best_L = 0
    # Calculate initial best_L
    for x in range(1, max_limit + 1):
        if counts[x] >= 2:
            best_L = x
        else:
            break
            
    # Local search loop
    while time.time() < deadline:
        bottleneck = get_bottleneck(counts, max_limit)
        
        # If we've covered up to max_limit, we can try to extend, but for now, 
        # if bottleneck > best_L, update best_L.
        if bottleneck > best_L:
            best_L = bottleneck - 1
            
        if bottleneck > max_limit:
            # We covered everything in our window.
            # We could expand max_limit, but for simplicity, if we hit the cap,
            # we assume we are doing well. 
            # The task says score is capped at 10^5, but L >= 1000 is the goal.
            # If we reach 2000, we are very good.
            break
            
        # Try to fix the bottleneck
        fixed = False
        # Shuffle primes to avoid bias towards small primes
        p_indices = list(range(n_primes))
        random.shuffle(p_indices)
        
        for idx in p_indices:
            if time.time() > deadline:
                break
                
            p = primes[idx]
            r_old = residues[idx]
            r_new = bottleneck % p
            
            if r_old == r_new:
                continue
            
            # Check safety: 
            # We lose coverage on y where y == r_old (mod p) and y != r_new (mod p)
            # We gain coverage on y where y == r_new (mod p) and y != r_old (mod p)
            
            # Critical check: For all y < bottleneck, if counts[y] == 2 and we lose coverage, it's bad.
            # Loss set: y < bottleneck, y % p == r_old, y % p != r_new.
            # Since r_old != r_new, y % p == r_old implies y % p != r_new.
            # So we just need to check all y < bottleneck with y % p == r_old.
            
            safe = True
            start_y = r_old
            if start_y == 0:
                start_y = p
            if start_y > bottleneck - 1:
                # No y < bottleneck is affected by loss
                safe = True
            else:
                for y in range(start_y, bottleneck, p):
                    if counts[y] == 2:
                        safe = False
                        break
            
            if safe:
                # Apply change
                residues[idx] = r_new
                # Update counts efficiently
                # Subtract old
                for y in range(start_y, max_limit + 1, p):
                    counts[y] -= 1
                # Add new
                start_new = r_new
                if start_new == 0:
                    start_new = p
                for y in range(start_new, max_limit + 1, p):
                    counts[y] += 1
                fixed = True
                break
        
        if not fixed:
            # If we couldn't fix the bottleneck with a single move, 
            # we might be stuck in a local minimum.
            # Strategy: Randomly perturb a residue to escape.
            # Pick a random prime and a random residue.
            # Check if it breaks any y < bottleneck.
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            r_old = residues[idx]
            r_new = random.randint(0, p - 1)
            
            if r_old == r_new:
                continue
            
            safe = True
            start_y = r_old
            if start_y == 0:
                start_y = p
            if start_y <= bottleneck - 1:
                for y in range(start_y, bottleneck, p):
                    if counts[y] == 2:
                        safe = False
                        break
            
            if safe:
                residues[idx] = r_new
                for y in range(start_y, max_limit + 1, p):
                    counts[y] -= 1
                start_new = r_new
                if start_new == 0:
                    start_new = p
                for y in range(start_new, max_limit + 1, p):
                    counts[y] += 1

    # Final calculation of L
    # We need the exact L such that 1..L are all covered >= 2 times.
    L = 0
    for x in range(1, max_limit + 1):
        if counts[x] >= 2:
            L = x
        else:
            break
            
    # The task asks for residues.
    return residues
