import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # We use a coverage array 'counts' where counts[x] is the number of primes covering x.
    # We maintain this array incrementally.
    # Initial size M. We can extend it if needed, but for L < 10000, M=20000 is safe.
    M = 20000
    counts = np.zeros(M + 1, dtype=np.int64)
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # Initialize residues randomly
    for i, p in enumerate(primes):
        r = random.randint(0, p - 1)
        residues[i] = r
        if r == 0:
            counts[p::p] += 1
        else:
            counts[r::p] += 1
            
    def calc_L(counts, M):
        # Find largest L such that counts[1..L] >= 2
        if counts[1] < 2:
            return 0
        mask = counts[1:M+1] < 2
        if not np.any(mask):
            return M
        first_fail_idx = np.argmax(mask)
        return int(first_fail_idx)
        
    def extend_M(counts, residues, M, new_M):
        new_counts = np.zeros(new_M + 1, dtype=np.int64)
        new_counts[:M + 1] = counts
        for j, pj in enumerate(primes):
            rj = residues[j]
            if rj == 0:
                start_m = (M + 1 + pj - 1) // pj
            else:
                start_m = max(0, (M + 1 - rj + pj - 1) // pj)
            first_idx = rj + start_m * pj
            if first_idx <= new_M:
                new_counts[first_idx::pj] += 1
        return new_counts

    current_L = calc_L(counts, M)
    best_L = current_L
    best_residues = residues.copy()
    
    # Helper to get indices in [1, L] congruent to r mod p
    def get_indices(r, p, L):
        if r == 0:
            # Indices: p, 2p, ... <= L
            if L < p:
                return np.array([], dtype=np.int64)
            return np.arange(p, L + 1, p)
        else:
            # Indices: r, r+p, ... <= L
            if L < r:
                return np.array([], dtype=np.int64)
            return np.arange(r, L + 1, p)
            
    def is_safe_move(i, r_old, r_new, L):
        """
        Check if moving prime i from r_old to r_new keeps counts[1..L] >= 2.
        We only need to check if any y covered by r_old (and not r_new? No, r_old and r_new are distinct mod p)
        actually, if r_old != r_new, the sets of y covered are disjoint.
        The move removes coverage from {y | y % p == r_old} and adds to {y | y % p == r_new}.
        The risk is in the removed set. If any y in removed set has counts[y] == 2, it drops to 1.
        """
        if r_old == r_new:
            return True
        p = primes[i]
        # Get y's that will lose coverage
        y_indices = get_indices(r_old, p, L)
        if len(y_indices) == 0:
            return True
        # Check if any of these have count exactly 2
        if np.any(counts[y_indices] == 2):
            return False
        return True

    def apply_move(i, r_old, r_new):
        p = primes[i]
        if r_old == r_new:
            return
        if r_old == 0:
            counts[p::p] -= 1
        else:
            counts[r_old::p] -= 1
        if r_new == 0:
            counts[p::p] += 1
        else:
            counts[r_new::p] += 1
        residues[i] = r_new

    def revert_move(i, r_old, r_new):
        # Revert is same as apply with swapped old/new
        apply_move(i, r_new, r_old)

    # Main Loop
    stuck_count = 0
    MAX_STUCK = 50
    
    while time.time() < deadline:
        current_L = calc_L(counts, M)
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
            stuck_count = 0
            
        if current_L >= M - 10:
            # Extend M
            new_M = M * 2
            counts = extend_M(counts, residues, M, new_M)
            M = new_M
            continue

        target_x = current_L + 1
        
        # Identify candidates: primes that do NOT cover target_x
        candidates = []
        for i in range(n_primes):
            if (target_x - residues[i]) % primes[i] != 0:
                candidates.append(i)
                
        if not candidates:
            # All primes cover target_x? Impossible if target_x is prime and > 1000? 
            # Or if target_x is small.
            # If all cover it, counts[target_x] >= 168. So current_L should have advanced.
            # This case implies current_L is not the first gap?
            # calc_L finds first gap. So counts[target_x] < 2.
            # If counts[target_x] < 2, not all primes cover it.
            pass

        # Score candidates by criticality
        # Criticality: number of y in [1, current_L] such that y % p == residues[i] and counts[y] == 2
        # We want low criticality.
        
        # To speed this up, we can just sort by p (larger p = fewer y's = likely lower criticality)
        # Or compute a proxy.
        # Let's compute exact criticality for top candidates only? 
        # With 168 primes, computing for all is O(N * L/p). Sum(L/p) ~ L * log(log(N)).
        # For L=1000, this is small. For L=10000, still manageable.
        
        criticalities = np.zeros(n_primes, dtype=np.int64)
        for i in range(n_primes):
            if i in candidates:
                p = primes[i]
                r = residues[i]
                y_indices = get_indices(r, p, current_L)
                if len(y_indices) > 0:
                    criticalities[i] = np.sum(counts[y_indices] == 2)
                    
        # Sort candidates by criticality (ascending)
        candidates.sort(key=lambda i: criticalities[i])
        
        found_repair = False
        for idx in candidates:
            old_res = residues[idx]
            new_res = target_x % primes[idx]
            
            if old_res == new_res:
                continue
                
            # Check safety
            if is_safe_move(idx, old_res, new_res, current_L):
                apply_move(idx, old_res, new_res)
                # Update L
                new_L = calc_L(counts, M)
                if new_L >= current_L:
                    found_repair = True
                    break
                else:
                    # Should not happen if is_safe_move is correct, but revert just in case
                    revert_move(idx, old_res, new_res)
            else:
                # Move is not safe. 
                # Try 2-step repair?
                # For now, skip to next candidate.
                pass
        
        if found_repair:
            stuck_count = 0
        else:
            stuck_count += 1
            if stuck_count > MAX_STUCK:
                # Random kick
                kick_size = random.randint(5, 15)
                kick_indices = random.sample(range(n_primes), kick_size)
                for i in kick_indices:
                    old_r = residues[i]
                    new_r = random.randint(0, primes[i] - 1)
                    if old_r != new_r:
                        apply_move(i, old_r, new_r)
                stuck_count = 0
                # Recalculate L after kick
                current_L = calc_L(counts, M)
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()

    return best_residues.tolist()
