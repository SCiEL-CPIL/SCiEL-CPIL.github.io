"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

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
    
    # 1. Greedy Removal Phase
    # We iterate through primes in increasing order.
    # If a prime p <= len(points) has all residues covered, we remove the least populated class.
    
    # We need to be careful: removing points changes len(points), which might make larger primes irrelevant.
    # However, if we process primes in increasing order, and we remove points, the set of "active" primes (p <= len(points)) shrinks.
    # But we must ensure that for ALL p <= len(points), the condition holds.
    # If we fix p=2, then p=3, etc., does that guarantee correctness for the final set?
    # Let's say we have a set S. We check p=2. If bad, fix. Check p=3. If bad, fix.
    # When we fix p=3, we remove some points. This might make p=2 bad again?
    # If p=2 was good (one class empty), removing points can only make the empty class larger or keep it empty. It cannot make it full.
    # So fixing a larger prime never breaks a smaller prime's admissibility.
    # Therefore, a single pass in increasing order is sufficient for the removal phase.
    
    for p in primes:
        if time.time() > deadline:
            break
        if p > len(points):
            break
            
        # Check if all residues are covered
        residues = set()
        for x in points:
            residues.add(x % p)
            if len(residues) == p:
                break
                
        if len(residues) == p:
            # Violation found. Find least populated class.
            counts = [0] * p
            for x in points:
                counts[x % p] += 1
                
            min_class_size = min(counts)
            min_residue = counts.index(min_class_size)
            
            # Remove all x in points such that x % p == min_residue
            to_remove = {x for x in points if x % p == min_residue}
            points -= to_remove
            
    # 2. Expansion Phase
    # We have a valid set. Now try to add back points.
    # We need to track empty residues for each prime p <= len(points).
    # If we add x, it is valid if for all p <= len(points)+1, x does not fill the LAST empty residue for p.
    
    # Precompute empty residues for current points
    # We only care about primes p <= len(points) + 1 potentially.
    # Actually, if p > len(points)+1, adding x cannot make all p residues covered.
    # So we only need to check p <= len(points) + 1.
    
    k = len(points)
    relevant_primes = [p for p in primes if p <= k + 1]
    
    # empty_residues[p] = set of residues r such that no x in points has x % p == r
    empty_residues = {}
    for p in relevant_primes:
        residues_present = set()
        for x in points:
            residues_present.add(x % p)
        all_res = set(range(p))
        empty_residues[p] = all_res - residues_present
        
    # Candidates are points not in `points`
    # We should try to add them in a smart order?
    # Maybe points that are "less constrained"?
    # For now, let's just iterate through all candidates.
    # To speed up, we can group candidates by their "risk"?
    # Or just iterate. N=5000 is small.
    
    candidates = sorted(set(range(N + 1)) - points)
    
    for x in candidates:
        if time.time() > deadline:
            break
            
        new_k = len(points) + 1
        # We need to check primes p <= new_k
        # If new_k is prime, it might be a new relevant prime.
        
        valid = True
        # Check existing relevant primes
        for p in relevant_primes:
            if p > new_k:
                break
            r = x % p
            if r in empty_residues[p]:
                # If this is the only empty residue, adding x fills it -> invalid
                if len(empty_residues[p]) == 1:
                    valid = False
                    break
        
        # Check if new_k is a prime and if x violates it
        # If new_k is prime, we need to check if x fills the last hole for p=new_k.
        # But we don't have empty_residues[new_k] computed yet.
        # We can compute it on the fly if needed, or just check if new_k is in primes.
        
        if valid and new_k in primes:
            # new_k is a prime. We need to check if adding x makes new_k fully covered.
            # Current size is new_k - 1.
            # There is at least 1 empty residue for new_k.
            # If there is exactly 1 empty residue, and x falls in it, invalid.
            # If there are >= 2 empty residues, valid.
            
            # Count empty residues for new_k
            residues_present = set()
            for y in points:
                residues_present.add(y % new_k)
            all_res = set(range(new_k))
            empties = all_res - residues_present
            
            if len(empties) == 1 and x % new_k in empties:
                valid = False
        
        if valid:
            points.add(x)
            # Update empty_residues
            # For all p in relevant_primes (and potentially new prime new_k)
            for p in relevant_primes:
                if p > new_k:
                    continue
                r = x % p
                if r in empty_residues[p]:
                    empty_residues[p].remove(r)
                    
            # If new_k is prime, add it to relevant_primes and empty_residues
            if new_k in primes:
                relevant_primes.append(new_k)
                # We already computed empties for new_k in the check above if it was prime
                # But if it wasn't prime, we didn't.
                # Wait, the check `if valid and new_k in primes` computed `empties` locally.
                # We need to store it.
                # Let's recompute or store properly.
                # To avoid recomputation, let's just recompute if we added it.
                # Or better, structure the loop to compute it once.
                
                # Actually, if new_k is prime, we computed `empties` in the `if valid and new_k in primes` block.
                # But that block is inside the `for x` loop.
                # We can just recompute it here for simplicity, it's O(k).
                residues_present = set()
                for y in points:
                    residues_present.add(y % new_k)
                all_res = set(range(new_k))
                empty_residues[new_k] = all_res - residues_present

    return list(points)
# EVOLVE-BLOCK-END
