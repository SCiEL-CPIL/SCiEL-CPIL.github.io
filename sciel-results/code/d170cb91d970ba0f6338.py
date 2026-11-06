import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Generate primes up to 1000 using sieve
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)

    # Precompute coverage arrays
    MAX_L = 10000  # Allow checking up to 10000 to be safe, but L likely < 1000
    # Actually, the task says "capped at 10^5", but "L >= 1000 would realize...". 
    # Let's set MAX_L to 2000 to be efficient. If we reach 2000, we can stop or expand.
    # But the score is the largest L. If we find L=1000, we want to see if we can go higher.
    # Let's use MAX_L = 5000.
    MAX_L = 5000
    
    cover = np.zeros((n_primes, 1000, MAX_L + 1), dtype=np.bool_)
    for i, p in enumerate(primes):
        for r in range(p):
            if r == 0:
                first = p
            else:
                first = r
            if first >= 1 and first <= MAX_L:
                cover[i, r, first::p] = True

    def find_L_from_counts(counts):
        """Find largest L from a counts array."""
        # We care about 1..MAX_L
        # If counts[1] < 2, L=0
        # Find first index x in 1..MAX_L where counts[x] < 2
        # If none, L = MAX_L
        gap_mask = (counts[1:MAX_L+1] < 2).astype(np.int32)
        if gap_mask.size == 0:
            return MAX_L
        first_gap_idx = np.argmax(gap_mask)
        if gap_mask[first_gap_idx] == 0:
            return MAX_L
        return first_gap_idx

    def get_counts(residues):
        """Compute full counts array for given residues."""
        counts = np.zeros(MAX_L + 1, dtype=np.int32)
        residues_arr = np.array(residues, dtype=np.int64)
        for i in range(n_primes):
            counts += cover[i, residues_arr[i], :].astype(np.int32)
        return counts

    # Initialization
    random.seed(42)
    
    # Try a few initial strategies
    candidates = []
    
    # 1. All zeros
    candidates.append([0] * n_primes)
    
    # 2. All ones
    candidates.append([1 % p for p in primes])
    
    # 3. r_p = p-1 (covers 0 mod p, i.e., multiples of p, wait. r=p-1 covers p-1, 2p-1... )
    # x % p == p-1.
    candidates.append([(p-1) % p for p in primes])
    
    # 4. Random
    for _ in range(20):
        candidates.append([random.randint(0, p - 1) for p in primes])
        
    # 5. Structured: r_p = 0 for small p, random for large
    for _ in range(5):
        res = [0 if p < 10 else random.randint(0, p-1) for p in primes]
        candidates.append(res)

    best_residues = None
    best_L = 0
    best_counts = None

    for res in candidates:
        counts = get_counts(res)
        L = find_L_from_counts(counts)
        if L > best_L:
            best_L = L
            best_residues = res[:]
            best_counts = counts

    current = best_residues[:]
    counts = best_counts.copy()
    current_L = best_L
    
    # Simulated Annealing / Local Search
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.995
    
    # Weights for selecting coordinates: smaller primes are more important
    # Weight ~ 1/p
    weights = [1.0 / p for p in primes]
    total_weight = sum(weights)
    
    no_improve = 0
    max_no_improve = 100
    
    # Precompute cumulative weights for selection
    cum_weights = np.cumsum(weights)
    cum_weights /= total_weight
    
    while time.time() < deadline - 2.0:
        # Select coordinate
        r_rand = random.random()
        i = np.searchsorted(cum_weights, r_rand)
        if i >= n_primes: i = n_primes - 1
        
        p = primes[i]
        old_r = current[i]
        
        # Generate new residue
        # Sometimes pick a random one, sometimes a "smart" one
        if random.random() < 0.5:
            new_r = random.randint(0, p - 1)
        else:
            # Try to fix the first hole if it exists
            if current_L < MAX_L:
                x_hole = current_L + 1
                # Check if prime i covers x_hole
                if (x_hole % p) != old_r:
                    # We don't cover it. If we set r to x_hole % p, we cover it.
                    # But does it cover it currently?
                    # cover[i, old_r, x_hole] is False if x_hole % p != old_r
                    # So setting new_r = x_hole % p will make cover[i, new_r, x_hole] True.
                    # This increases count[x_hole] by 1.
                    new_r = x_hole % p
                else:
                    # We already cover it. Changing it might break it.
                    # Better to change to something else or leave it?
                    # Let's pick a random one that is different
                    new_r = random.randint(0, p - 1)
                    if new_r == old_r:
                        new_r = (new_r + 1) % p
            else:
                new_r = random.randint(0, p - 1)
                
        if new_r == old_r:
            continue

        # Evaluate change
        # Delta counts = cover[i, new_r] - cover[i, old_r]
        delta = cover[i, new_r, :].astype(np.int32) - cover[i, old_r, :].astype(np.int32)
        
        # Apply temporarily to a copy? Or just compute new L?
        # Since MAX_L is 5000, we can just add delta to a temp counts and find L.
        # But finding L requires scanning.
        # Optimization: If current_L is small, we only care about 1..current_L + some buffer.
        # However, a change can create a hole at x=1. So we must check 1..MAX_L.
        
        temp_counts = counts + delta
        new_L = find_L_from_counts(temp_counts)
        
        # Acceptance criterion
        if new_L > current_L:
            # Accept
            counts = temp_counts
            current[i] = new_r
            current_L = new_L
            no_improve = 0
            if current_L > best_L:
                best_L = current_L
                best_residues = current[:]
                best_counts = counts.copy()
            T = max(T_min, T * cooling_rate)
        else:
            # Calculate delta L
            delta_L = new_L - current_L
            # Accept with probability exp(delta_L / T)
            # If delta_L is negative, this is < 1
            if delta_L < 0:
                prob = np.exp(delta_L / T)
                if random.random() < prob:
                    # Accept
                    counts = temp_counts
                    current[i] = new_r
                    current_L = new_L
                    no_improve += 1
                    T = max(T_min, T * cooling_rate)
                else:
                    # Reject
                    no_improve += 1
                    T = max(T_min, T * cooling_rate)
            else:
                # delta_L == 0
                # Accept with some probability to move around plateau
                if random.random() < 0.5:
                    counts = temp_counts
                    current[i] = new_r
                    # current_L unchanged
                    no_improve += 1
                    T = max(T_min, T * cooling_rate)
                else:
                    no_improve += 1
                    T = max(T_min, T * cooling_rate)
        
        # If stuck for a while, do a "Repair" move
        if no_improve > max_no_improve:
            # Find first hole
            if current_L < MAX_L:
                x_hole = current_L + 1
                # Find a prime that doesn't cover x_hole
                # c_p(x_hole) = 1 if x_hole % p == r_p
                # We want to pick a p where x_hole % p != r_p
                # And set r_p = x_hole % p
                # This will increase count[x_hole] by 1.
                # If count[x_hole] was 1, it becomes 2 -> Hole fixed.
                # If count[x_hole] was 0, it becomes 1 -> Hole remains, but maybe other holes fixed?
                
                # Pick a random prime from those not covering x_hole
                non_covering_primes = []
                for j in range(n_primes):
                    if (x_hole % primes[j]) != current[j]:
                        non_covering_primes.append(j)
                
                if non_covering_primes:
                    j = random.choice(non_covering_primes)
                    p_j = primes[j]
                    old_r_j = current[j]
                    new_r_j = x_hole % p_j
                    
                    # Apply this move forcefully
                    delta_j = cover[j, new_r_j, :].astype(np.int32) - cover[j, old_r_j, :].astype(np.int32)
                    counts += delta_j
                    current[j] = new_r_j
                    current_L = find_L_from_counts(counts)
                    
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = current[:]
                        best_counts = counts.copy()
                    
                    no_improve = 0
                    # Reset temperature to encourage further search
                    T = max(T, 5.0)
            else:
                no_improve = 0
                T = max(T, 5.0)

    return best_residues if best_residues is not None else [0] * n_primes
