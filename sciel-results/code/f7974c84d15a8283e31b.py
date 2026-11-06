"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 5000
DEADLINE = 300.0  # Safety margin for the 330s limit

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

# Precompute primes up to N. 
# The constraint is: for every prime p <= len(tuple), some residue class mod p is missed.
# As we add elements, len(tuple) increases. 
# If len(tuple) reaches a new prime P, we must ensure that not all residues mod P are covered.
# If len(tuple) is not a prime, the constraint for that specific prime is not active yet?
# Wait, the problem statement says: "for every prime p <= len(tuple) some residue class mod p is missed".
# This implies that if len(tuple) = 10, we must satisfy constraints for p=2, 3, 5, 7.
# If len(tuple) = 11, we must satisfy constraints for p=2, 3, 5, 7, 11.

# Strategy:
# 1. Start with an empty set.
# 2. Iterate through candidate integers in [0, N].
# 3. For each candidate, check if adding it to the current set S would violate the admissibility condition.
#    Violation occurs if for some prime p <= |S|+1, the new set S' covers all residues mod p.
#    Note: We only need to check primes p <= |S'|.
#    Actually, if p <= |S|, the condition must already be satisfied by S (since we maintain it).
#    Adding an element can only *add* residues. It cannot remove a missing residue.
#    So, if S is admissible, S' = S U {x} is admissible IF AND ONLY IF:
#    For all primes p <= |S'|, S' does not cover all residues mod p.
#    
#    Let's analyze the risk:
#    If p <= |S|, S already misses some residue r. Adding x adds x%p. 
#    S' covers all residues mod p iff S covered all residues except possibly x%p, AND x%p is that missing one?
#    No, S misses at least one residue. Let M_p be the set of missing residues in S mod p.
#    S' misses M_p \ {x%p}.
#    So S' is invalid for prime p iff M_p == {x%p}. i.e., S missed exactly one residue mod p, and x fills it.
#    
#    If p == |S'| (and p is prime), this is a new prime constraint.
#    We must ensure S' does not cover all residues mod p.
#    Since |S'| = p, S' has p elements. It covers all residues mod p iff they are distinct mod p.
#    
#    So the check is:
#    For each prime p <= |S'|:
#    1. If p <= |S|: Check if S missed exactly one residue mod p, and x fills it.
#    2. If p == |S'|: Check if all elements in S' are distinct mod p.

def construct():
    start_time = time.time()
    primes = _primes(N)
    # We need primes up to N because the set size can go up to N.
    # But the constraint is p <= len(tuple). Max len is N.
    
    # To speed up, we can precompute the "missing residues" state?
    # Or just maintain counts of residues for each prime.
    # Number of primes up to 5000 is ~669.
    # For each prime, we track count of each residue mod p.
    # Total memory: sum(p for p in primes) ~ 5000 * 669 / 2? No, sum of primes ~ N^2/(2lnN) ~ 1.5M.
    # This is manageable.
    
    # res_counts[p][r] = count of elements in S congruent to r mod p
    # We only need to track primes p that are <= current size + 1?
    # Actually, we need to track all primes p <= N, because if size grows, new primes become active.
    # But if p > current_size + 1, it's not active.
    # However, maintaining counts for all primes up to N for every addition might be slow.
    # Optimization: Only update counts for primes p <= current_size + 1?
    # No, if we don't update counts for p=1000 now, and size grows to 1000 later, we won't have the counts.
    # So we must update counts for all primes p <= N?
    # 5000 additions * 669 primes * (1 mod op + 1 array access) ~ 3.3M ops. Very fast in Python?
    # 3.3M ops is < 1s.
    
    # Let's use a list of dicts or lists of lists.
    # primes_list = [p1, p2, ...]
    # counts[i] is a list of length primes_list[i], initialized to 0.
    
    prime_indices = {p: i for i, p in enumerate(primes)}
    counts = [ [0] * p for p in primes ]
    
    S = []
    current_size = 0
    
    # Heuristic for adding: 
    # Just iterate 0..N? Or random?
    # Iterating 0..N might get stuck if early numbers block later ones?
    # Actually, admissibility is "hard" to violate if you are sparse.
    # The constraint is "miss a residue". It is easy to miss a residue.
    # It is hard to *cover* all residues.
    # So greedy addition should work well.
    # To maximize size, we should probably add numbers that are "least likely" to complete a residue class.
    # But any addition that doesn't complete a class is valid.
    # So we just scan 0..N and add if valid.
    
    # To improve speed, we can check validity quickly.
    # Validity check for adding x:
    # 1. New size k = current_size + 1.
    # 2. For each prime p in primes:
    #    if p > k: break (since we only care about p <= k)
    #    if p == k:
    #        # Check if S U {x} has distinct residues mod p
    #        # This is equivalent to: x mod p is not in {s mod p for s in S}
    #        # We can check this by looking at counts[p_idx][x%p] == 0?
    #        # Wait, if counts[p_idx][x%p] > 0, then x mod p is already present.
    #        # If x mod p is NOT present, then S U {x} has all distinct residues?
    #        # S has k-1 = p-1 elements. If they are all distinct mod p (which they must be if S was admissible for p? No, S didn't have constraint p yet if p>k-1).
    #        # Wait, if p = k, then for S (size k-1), p was not active.
    #        # So S might have collisions mod p.
    #        # We need S U {x} to miss a residue.
    #        # Since size is p, missing a residue means NOT all residues are present.
    #        # So we need to check if the set of residues of S U {x} mod p has size < p.
    #        # This is true if there is a collision in S U {x} mod p.
    #        # Collision exists if:
    #        a) x mod p is already in S (counts[p_idx][x%p] > 0) -> Collision. Valid.
    #        b) x mod p is new, but S has a collision?
    #           If S has a collision mod p, then S misses at least one residue (since size p-1 < p, it can't cover all, but it might cover p-1 distinct ones).
    #           Actually, if S has size p-1, it can cover at most p-1 residues.
    #           If S covers p-1 distinct residues, adding a new residue x%p makes it p distinct residues -> Covers ALL -> Invalid.
    #           If S has a collision (covers < p-1 distinct residues), adding x%p (even if new) results in < p distinct residues -> Valid.
    #           
    #    So for p == k:
    #       Let r = x % p.
    #       If counts[p_idx][r] > 0: Valid (collision with x).
    #       Else:
    #           We need to know if S covers p-1 distinct residues.
    #           If yes, adding new r makes p distinct -> Invalid.
    #           If no, adding new r makes < p distinct -> Valid.
    #           How to check if S covers p-1 distinct residues?
    #           We can maintain `distinct_counts[p_idx]` = number of residues with count > 0.
    #           If distinct_counts[p_idx] == p-1, then S covers p-1 distinct residues.
    #           Then adding new r makes it p -> Invalid.
    #           If distinct_counts[p_idx] < p-1, then adding new r makes it <= p-1 -> Valid.
    
    #    if p < k:
    #       # p was active for S. S missed at least one residue.
    #       # Let r = x % p.
    #       # If counts[p_idx][r] > 0: S' still misses the same residues (or more). Valid.
    #       # If counts[p_idx][r] == 0:
    #           # r was missing in S.
    #           # If r was the ONLY missing residue in S, then S' covers all -> Invalid.
    #           # How to check if r was the only missing residue?
    #           # distinct_counts[p_idx] == p - 1.
    #           # If distinct_counts[p_idx] == p - 1, and counts[p_idx][r] == 0, then r is the unique missing one.
    #           # Adding it makes distinct_counts p -> Invalid.
    #           # If distinct_counts[p_idx] < p - 1, then even if we add r, distinct_counts < p -> Valid.
    
    # Summary of Check for x:
    # k = current_size + 1
    # for i, p in enumerate(primes):
    #   if p > k: break
    #   r = x % p
    #   if counts[i][r] == 0:
    #       # r is new for S
    #       if distinct_counts[i] == p - 1:
    #           # S was missing exactly one residue (r), and we are adding it.
    #           # This makes S' cover all p residues.
    #           return False (Invalid)
    #       # else distinct_counts[i] < p - 1, so S' will still miss at least one.
    #   # if counts[i][r] > 0, S' still misses whatever S missed (since r was already present, we didn't fill a hole).
    #       # Wait, if S missed r, and counts[i][r] > 0, that's a contradiction.
    #       # If counts[i][r] > 0, r is NOT missing in S.
    #       # So S missed some other residue. Adding x (which is r) doesn't fill that other residue.
    #       # So S' still misses that other residue. Valid.
    
    # So the condition for INVALID is:
    # There exists a prime p <= k such that:
    # 1. x % p was NOT in S (counts[i][r] == 0)
    # 2. S covered exactly p-1 distinct residues mod p (distinct_counts[i] == p - 1)
    
    # This is very fast! O(number of primes <= k).
    
    distinct_counts = [0] * len(primes)
    
    for x in range(N + 1):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        k = current_size + 1
        valid = True
        # Check primes p <= k
        for i, p in enumerate(primes):
            if p > k:
                break
            r = x % p
            if counts[i][r] == 0:
                # r is new
                if distinct_counts[i] == p - 1:
                    valid = False
                    break
                # else: distinct_counts[i] < p - 1, so adding r keeps distinct_counts < p
            # else: counts[i][r] > 0, r is already present, so we didn't fill a missing slot
            #       (actually, if r was missing, counts would be 0. So if counts > 0, r was not missing.)
            #       So S missed some other residue. Adding x doesn't fill it. Valid.
        
        if valid:
            # Add x to S
            S.append(x)
            current_size = k
            
            # Update counts and distinct_counts
            for i, p in enumerate(primes):
                if p > k:
                    break
                r = x % p
                if counts[i][r] == 0:
                    distinct_counts[i] += 1
                counts[i][r] += 1
    
    return S

# EVOLVE-BLOCK-END
