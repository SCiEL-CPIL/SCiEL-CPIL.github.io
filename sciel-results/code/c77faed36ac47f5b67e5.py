"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import heapq

N = 5000

# EVOLVE-BLOCK-START
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

def construct():
    deadline = time.time() + 290  # safety margin
    
    points = set(range(N + 1))
    primes = _primes(N + 1)
    prime_set = set(primes)
    
    # 1. Greedy Removal Phase with improved heuristic
    # For each prime, if all residues are covered, remove the residue class that
    # maximizes the "flexibility" - we approximate this by removing the class
    # whose elements have the most diverse interactions with other small primes.
    for p in primes:
        if time.time() > deadline:
            break
        if p > len(points):
            break
            
        # Check if all residues are covered
        counts = [0] * p
        residue_elements = [set() for _ in range(p)]
        for x in points:
            r = x % p
            counts[r] += 1
            residue_elements[r].add(x)
            
        if min(counts) > 0:
            # All residues covered - need to remove one class
            # Heuristic: remove the class with minimum count, breaking ties by
            # choosing the class whose elements are most "spread out" modulo other small primes
            min_class_size = min(counts)
            candidates = [r for r in range(p) if counts[r] == min_class_size]
            
            if len(candidates) == 1:
                min_residue = candidates[0]
            else:
                # Score each candidate residue class by how many distinct residues
                # its elements occupy modulo small primes (higher = more spread = better to remove)
                best_residue = candidates[0]
                best_score = -1
                small_primes = [q for q in primes if q < p and q <= 20]
                for r in candidates:
                    score = 0
                    for q in small_primes:
                        residues_seen = set()
                        for x in residue_elements[r]:
                            residues_seen.add(x % q)
                        score += len(residues_seen)
                    if score > best_score:
                        best_score = score
                        best_residue = r
                min_residue = best_residue
            
            to_remove = residue_elements[min_residue]
            points -= to_remove

    # 2. Expansion Phase with priority queue
    k = len(points)
    
    # Relevant primes are those p <= k+1
    # We will maintain empty_res for these primes
    empty_res = {}
    
    def compute_empty(p):
        present = set()
        for x in points:
            present.add(x % p)
        return set(range(p)) - present

    # Initialize empty_res for primes <= k+1
    for p in primes:
        if p <= k + 1:
            empty_res[p] = compute_empty(p)
        else:
            break

    forbidden = [False] * (N + 1)
    
    def mark_forbidden(p, r):
        # Mark all x in [0, N] not in points such that x % p == r as forbidden
        for x in range(r, N + 1, p):
            if x not in points:
                forbidden[x] = True

    # Initial marking
    for p, empties in empty_res.items():
        if len(empties) == 1:
            r = next(iter(empties))
            mark_forbidden(p, r)

    # Build priority queue of candidates
    # Priority: negative of number of constraints satisfied (more constraints = higher priority)
    candidate_heap = []
    for x in range(N + 1):
        if x not in points and not forbidden[x]:
            # Count how many primes have x in a non-empty residue class
            compat = 0
            for p in empty_res:
                if x % p not in empty_res[p]:
                    compat += 1
            heapq.heappush(candidate_heap, (-compat, x))
    
    # Iterate through candidates in priority order
    added = set()
    while candidate_heap:
        if time.time() > deadline:
            break
        
        neg_compat, x = heapq.heappop(candidate_heap)
        
        if x in points or x in added:
            continue
        if forbidden[x]:
            continue
            
        new_k = k + 1
        valid = True
        
        # Check if new_k is a prime and if x violates it
        if new_k in prime_set:
            # Compute empty residues for new_k based on current points (size new_k - 1)
            present = set()
            for y in points:
                present.add(y % new_k)
            empties_new = set(range(new_k)) - present
            
            if len(empties_new) == 1:
                r_new = next(iter(empties_new))
                if x % new_k == r_new:
                    valid = False
        
        if valid:
            points.add(x)
            added.add(x)
            k = new_k
            
            # Update empty_res for existing primes
            for p in list(empty_res.keys()):
                if p > k:
                    continue
                r_x = x % p
                if r_x in empty_res[p]:
                    empty_res[p].remove(r_x)
                    if len(empty_res[p]) == 1:
                        r_new = next(iter(empty_res[p]))
                        mark_forbidden(p, r_new)
            
            # If new_k is prime, add it to tracking
            if k in prime_set:
                empty_res[k] = compute_empty(k)
                if len(empty_res[k]) == 1:
                    r_new = next(iter(empty_res[k]))
                    mark_forbidden(k, r_new)
            
            # Add new candidates that became un-forbidden or were not previously in heap
            # This is expensive, so we'll do a limited scan
            if time.time() < deadline - 10:
                # Scan a subset of candidates that might now be valid
                for x2 in range(N + 1):
                    if x2 in points or x2 in added or forbidden[x2]:
                        continue
                    # Check if this candidate was already in the heap by checking if it's compatible
                    # For efficiency, we'll just add a limited number of new candidates
                    compat = 0
                    for p in empty_res:
                        if x2 % p not in empty_res[p]:
                            compat += 1
                    heapq.heappush(candidate_heap, (-compat, x2))
                    if len(candidate_heap) > 10000:
                        break

    return list(points)
# EVOLVE-BLOCK-END
