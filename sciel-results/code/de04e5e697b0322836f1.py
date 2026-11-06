import time
import random

N = 5000
DEADLINE = 300
START_TIME = time.time()

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _greedy_construct(seed):
    random.seed(seed)
    
    # Precompute primes up to N
    all_primes = _primes(N)
    
    # Initialize counts for all primes up to N
    # counts[p] is a list of length p
    counts = {}
    for p in all_primes:
        counts[p] = [0] * p
        for i in range(N + 1):
            counts[p][i % p] += 1
            
    points = list(range(N + 1))
    # We need to track which points are present
    present = [True] * (N + 1)
    
    while True:
        if time.time() - START_TIME > DEADLINE - 10:
            break
            
        M = sum(present)
        if M == 0:
            break
            
        # Get primes <= M
        # We can filter all_primes
        active_primes = [p for p in all_primes if p <= M]
        
        if not active_primes:
            break
            
        bad_primes = []
        for p in active_primes:
            c = counts[p]
            if min(c) > 0:
                min_count = min(c)
                candidates = [r for r, cnt in enumerate(c) if cnt == min_count]
                bad_primes.append((p, min_count, candidates))
                
        if not bad_primes:
            break
            
        min_removals = min(bp[1] for bp in bad_primes)
        candidates_for_removal = [bp for bp in bad_primes if bp[1] == min_removals]
        p_star, _, residues_star = random.choice(candidates_for_removal)
        r_star = random.choice(residues_star)
        
        # Remove all points congruent to r_star mod p_star
        # We need to update counts for ALL primes, not just p_star
        # Because removing x affects x % q for all q.
        
        to_remove = []
        for i in range(N + 1):
            if present[i] and i % p_star == r_star:
                to_remove.append(i)
                
        if not to_remove:
            # Should not happen
            break
            
        for i in to_remove:
            present[i] = False
            # Update counts for all active primes
            for q in active_primes:
                counts[q][i % q] -= 1
                
    # Add-back search
    # Try to add back removed elements
    # We need to check if adding i makes any prime "bad" (all residues covered)
    # Actually, adding i makes a prime "bad" if it was the last empty residue.
    # So we need to know which primes are currently "safe" (have >= 1 empty residue)
    # and if adding i fills the *only* empty residue for that prime.
    
    # Let's just try to add back in random order or sorted order.
    # To be efficient, we can just try all removed points.
    
    removed_points = [i for i in range(N + 1) if not present[i]]
    random.shuffle(removed_points)
    
    for i in removed_points:
        if time.time() - START_TIME > DEADLINE - 5:
            break
            
        # Check if we can add i
        can_add = True
        # New size will be M + 1
        new_M = sum(present) + 1
        # Primes to check: all primes <= new_M
        # But we only have counts for primes <= N.
        # If new_M > N, we need to handle primes between M and new_M.
        # But M is decreasing, so new_M is small.
        # Actually, M starts at N and decreases. So new_M <= N.
        
        # We need to check all primes p <= new_M.
        # For each p, if counts[p][i % p] == 0, then adding i makes it 1.
        # If it was 0, and all other counts[p][r] > 0, then p becomes bad.
        # So we need to check: for all p <= new_M, if counts[p][i % p] == 0, 
        # then we must ensure that there is another r' != i%p such that counts[p][r'] == 0.
        
        # Optimization: Precompute "empty counts" for each prime?
        # Let's just check.
        
        for p in all_primes:
            if p > new_M:
                break
                
            r = i % p
            if counts[p][r] == 0:
                # Check if there is another empty residue
                # If min(counts[p]) == 0 and count of zeros == 1, then adding i makes min > 0
                # We can check if sum(1 for c in counts[p] if c == 0) > 1
                # This is O(p).
                zeros = 0
                for c in counts[p]:
                    if c == 0:
                        zeros += 1
                        if zeros > 1:
                            break
                if zeros == 1:
                    can_add = False
                    break
                    
        if can_add:
            present[i] = True
            for q in all_primes:
                if q <= new_M:
                    counts[q][i % q] += 1
            # Note: if we add i, M increases.
            # We might need to check new primes?
            # The condition is for p <= len(tuple).
            # If len increases, we might need to check a new prime p_new.
            # If p_new is bad, we can't add i?
            # Yes, if adding i makes the tuple non-admissible, we can't add it.
            # So we must check p_new as well.
            # The loop above checks p <= new_M, so it covers p_new.
            
    return [i for i in range(N + 1) if present[i]]

def construct():
    best_points = []
    best_size = -1
    
    for seed in range(100):
        if time.time() - START_TIME > DEADLINE - 10:
            break
            
        pts = _greedy_construct(seed)
        if len(pts) > best_size:
            best_size = len(pts)
            best_points = pts
            
    return best_points
